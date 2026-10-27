"""Render an OpenSim model driven by a .mot motion to images, offscreen, with VTK.
Bones are the model's own meshes (Pose2Sim's Geometry folder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def geometry_dir() -> Path:
    import Pose2Sim
    return Path(Pose2Sim.__file__).parent / "OpenSim_Setup" / "Geometry"


class OsimScene:
    def __init__(self, osim_path: Path, mot_path: Path, size=(480, 600), background=(0.043, 0.051, 0.067)):
        import opensim as osim
        import vtk
        self.osim, self.vtk = osim, vtk
        osim.Logger.setLevelString("error")
        self.model = osim.Model(str(osim_path))
        self.state = self.model.initSystem()
        table = osim.TimeSeriesTable(str(mot_path))
        self.times = np.array(table.getIndependentColumn())
        self.labels = list(table.getColumnLabels())
        self.data = np.array([[table.getRowAtIndex(i)[j] for j in range(len(self.labels))] for i in range(len(self.times))])
        self.coords = {}
        cs = self.model.getCoordinateSet()
        for i in range(cs.getSize()):
            c = cs.get(i)
            self.coords[c.getName()] = (c, c.getMotionType() == osim.Coordinate.Rotational)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*background)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetMultiSamples(8)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(*size)
        self.meshes = []
        gdir = geometry_dir()
        for comp in self.model.getComponentsList():
            mesh = osim.Mesh.safeDownCast(comp)
            if mesh is None:
                continue
            f = gdir / mesh.get_mesh_file()
            if not f.exists():
                continue
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
            p.SetColor(0.93, 0.90, 0.82)
            p.SetSpecular(0.15)
            p.SetAmbient(0.25)
            p.SetDiffuse(0.85)
            self.ren.AddActor(actor)
            s = mesh.get_scale_factors()
            self.meshes.append((actor, mesh.getFrame(), np.array([s.get(0), s.get(1), s.get(2)])))
        self.marker_actors = []
        self._floor()
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        light.SetPosition(0.5, 1, 1)
        light.SetIntensity(0.9)
        self.ren.AddLight(light)

    def _floor(self, half=3.0, step=0.5):
        vtk = self.vtk
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        k = 0
        for v in np.arange(-half, half + 1e-6, step):
            for a, b in (((v, 0, -half), (v, 0, half)), ((-half, 0, v), (half, 0, v))):
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
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.20, 0.23, 0.28)
        self.ren.AddActor(a)

    def set_markers(self, n: int, color=(0.776, 0.949, 0.306), radius=0.018):
        vtk = self.vtk
        for _ in range(n):
            src = vtk.vtkSphereSource()
            src.SetRadius(radius)
            src.SetThetaResolution(12)
            src.SetPhiResolution(12)
            m = vtk.vtkPolyDataMapper()
            m.SetInputConnection(src.GetOutputPort())
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(*color)
            self.ren.AddActor(a)
            self.marker_actors.append(a)

    def pose(self, frame: int, markers: np.ndarray | None = None):
        osim, vtk = self.osim, self.vtk
        row = self.data[frame]
        for name, v in zip(self.labels, row):
            if name in self.coords:
                c, rot = self.coords[name]
                c.setValue(self.state, float(np.radians(v) if rot else v), False)
        self.model.realizePosition(self.state)
        for actor, frame_obj, scale in self.meshes:
            T = frame_obj.getTransformInGround(self.state)
            R, p = T.R(), T.p()
            M = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    M.SetElement(i, j, R.get(i, j) * scale[j])
                M.SetElement(i, 3, p.get(i))
            actor.SetUserMatrix(M)
        if markers is not None:
            for a, m in zip(self.marker_actors, markers):
                a.SetVisibility(bool(np.isfinite(m).all()))
                if np.isfinite(m).all():
                    a.SetPosition(*m)

    def pelvis(self) -> np.ndarray:
        T = self.model.getBodySet().get("pelvis").getTransformInGround(self.state)
        return np.array([T.p().get(i) for i in range(3)])

    def shot(self, direction, focus, distance=4.2, up=(0, 1, 0)) -> np.ndarray:
        """RGB image looking at `focus` from `direction` (a unit vector from focus to camera)."""
        vtk = self.vtk
        cam = self.ren.GetActiveCamera()
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        cam.SetFocalPoint(*focus)
        cam.SetPosition(*(np.asarray(focus) + d * distance))
        cam.SetViewUp(*up)
        cam.SetViewAngle(30)
        self.ren.ResetCameraClippingRange()
        self.win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.win)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img = w2i.GetOutput()
        w, h, _ = img.GetDimensions()
        from vtk.util.numpy_support import vtk_to_numpy
        arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
        return np.ascontiguousarray(arr)
