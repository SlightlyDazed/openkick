"""Draw the OpenSim model (its bones) from a baked motion, off screen, with VTK. No Qt here."""

from __future__ import annotations

from pathlib import Path

import numpy as np

BONE = (0.93, 0.90, 0.82)
BACKGROUND = (0.027, 0.035, 0.047)
GRID = (0.137, 0.157, 0.200)


class BoneScene:
    def __init__(self, baked: str | Path):
        import vtk
        self.vtk = vtk
        z = np.load(baked, allow_pickle=False)
        self.times, self.M = z["times"], z["M"]
        self.labels, self.data = [str(x) for x in z["labels"]], z["data"]
        gdir = Path(str(z["geometry"]))
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*BACKGROUND)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetShowWindow(False)
        self.win.SetMultiSamples(8)
        self.win.AddRenderer(self.ren)
        self.actors = []
        for name in z["files"]:
            f = gdir / str(name)
            reader = vtk.vtkXMLPolyDataReader() if f.suffix == ".vtp" else vtk.vtkSTLReader()
            reader.SetFileName(str(f))
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(reader.GetOutputPort())
            normals.SetFeatureAngle(60)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            p = actor.GetProperty()
            p.SetColor(*BONE)
            p.SetSpecular(0.15)
            p.SetAmbient(0.25)
            p.SetDiffuse(0.85)
            self.ren.AddActor(actor)
            self.actors.append(actor)
        self._floor = None
        self._frame = -1
        self._matrices = [vtk.vtkMatrix4x4() for _ in self.actors]
        for a, m in zip(self.actors, self._matrices):
            a.SetUserMatrix(m)
        self._grab = vtk.vtkWindowToImageFilter()
        self._grab.SetInput(self.win)
        self._grab.ReadFrontBufferOff()

    def frame_at(self, seconds: float) -> int:
        return int(np.abs(self.times - seconds).argmin())

    def angle(self, name: str, frame: int) -> float:
        return float(self.data[frame, self.labels.index(name)]) if name in self.labels else float("nan")

    def set_floor(self, cx: float, cz: float, half: float, step: float = 0.5):
        vtk = self.vtk
        if self._floor is not None:
            self.ren.RemoveActor(self._floor)
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        k = 0
        for u in np.arange(-half, half + 1e-6, step):
            for a, b in (((cx + u, 0, cz - half), (cx + u, 0, cz + half)), ((cx - half, 0, cz + u), (cx + half, 0, cz + u))):
                pts.InsertNextPoint(*a)
                pts.InsertNextPoint(*b)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, k)
                line.GetPointIds().SetId(1, k + 1)
                lines.InsertNextCell(line)
                k += 2
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        self._floor = vtk.vtkActor()
        self._floor.SetMapper(m)
        self._floor.GetProperty().SetColor(*GRID)
        self._floor.GetProperty().LightingOff()
        self.ren.AddActor(self._floor)

    def render(self, frame: int, size: tuple[int, int], center, up, toward, px_per_m: float) -> np.ndarray:
        """RGB image (h, w, 3) of `frame`, seen without perspective from the direction `toward`
        (unit vector from `center` to the viewer), `px_per_m` pixels per metre."""
        if frame != self._frame:
            for m, T in zip(self._matrices, self.M[frame]):
                for i in range(4):
                    for j in range(4):
                        m.SetElement(i, j, float(T[i, j]))
            self._frame = frame
        w, h = int(size[0]), int(size[1])
        if tuple(self.win.GetSize()) != (w, h):
            self.win.SetSize(w, h)
        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(np.asarray(center, float) + np.asarray(toward, float) * 20.0))
        cam.SetViewUp(*up)
        cam.SetParallelScale(h / 2 / px_per_m)
        self.ren.ResetCameraClippingRange()
        self.win.Render()
        self._grab.Modified()
        self._grab.Update()
        img = self._grab.GetOutput()
        iw, ih, _ = img.GetDimensions()
        from vtk.util.numpy_support import vtk_to_numpy
        arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(ih, iw, -1)[::-1, :, :3]
        return np.ascontiguousarray(arr)
