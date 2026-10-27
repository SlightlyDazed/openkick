"""Two-camera 3D reconstruction (experimental).

The person is the calibration object: camera poses are solved from the 2D skeletons seen by
both cameras, and the user corrects wrong points by dragging them, exactly as in the 2D editor.
Optional dependencies: rtmlib + onnxruntime (pose), opencv (stabilization), Pose2Sim + opensim
(export to OpenSim) and vtk (rendering)."""

REQUIREMENTS = {"rtmlib": "rtmlib", "onnxruntime": "onnxruntime"}
OPENSIM_REQUIREMENTS = {"Pose2Sim": "pose2sim", "opensim": "opensim", "vtk": "vtk"}


def missing(requirements: dict | None = None) -> list[str]:
    """pip names of the optional packages that are not installed."""
    import importlib.util
    req = REQUIREMENTS if requirements is None else requirements
    return [pip for mod, pip in req.items() if importlib.util.find_spec(mod) is None]
