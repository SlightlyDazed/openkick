# Writing an OpenKick plugin

A plugin is a Python file (or a folder with a `plugin.py`) inside the user's plugins folder. In
OpenKick: *Plugins › Open plugins folder*.

Each plugin defines a `PLUGINS` list with instances of:

- `AnalysisPlugin`: computes something and returns tables (it appears in the *Plugins* menu).
- `ExportPlugin`: writes a file (it appears under *Export*).

## Example: maximum knee flexion per phase

Save this as `knee_flexion.py` in the plugins folder and reopen the Plugins menu:

```python
from openkick.plugins import AnalysisPlugin, Result, Table


class KneeFlexion(AnalysisPlugin):
    id = "knee_flexion"
    name = {"en": "Max knee flexion", "es": "Flexión máxima de rodilla"}
    description = {"en": "Minimum angle of the right knee in each phase."}
    version = "1.0"
    author = "Your name"

    def run(self, ctx):
        knee = ctx.find_angle("right_knee")
        if knee is None:
            return Result(summary="Add the right knee angle.")
        rows = []
        for phase in ctx.project.phases:
            s = ctx.stats(knee, phase)
            rows.append([phase.name, ctx.fmt(s.minimum, "°"), f"{s.t_min:.2f} s"])
        return Result(tables=[Table(["Phase", "Minimum angle", "When"], rows)])


PLUGINS = [KneeFlexion()]
```

## What `ctx` provides

| Attribute / method | What it is |
|---|---|
| `ctx.project` | The project: `video`, `pose`, `measurements`, `phases`, `notes`, `calibration` |
| `ctx.project.pose.xy` | Array `(frames, 33, 2)` in pixels (`NaN` = not detected) |
| `ctx.project.pose.visibility` | The detector's confidence `(frames, 33)` |
| `ctx.project.pose.source` | Origin of each point: 1 = detected, 2 = corrected by hand, 3 = smoothed |
| `ctx.project.pose.detector` | Which detector found it (`heavy`, `rtmpose-balanced`, …; empty in older projects, MediaPipe) |
| `ctx.times()` | Real time of each frame, in seconds |
| `ctx.series(m)` | Series of a measurement: `.value` (° / m/s / m), `.rate` (°/s for angles), `.xy` (trajectories) |
| `ctx.stats(m, phase=None)` | `minimum`, `maximum`, `range`, `mean`, `t_min`, `t_max` |
| `ctx.find_angle("right_knee")` | The angle whose vertex is that joint |
| `ctx.landmarks` | Names and indices of the 33 points (`ctx.landmarks.INDEX["left_hip"]`) |
| `ctx.ask_number(text, default)` | Asks the user for a number |
| `ctx.fmt(value, unit)` | Formats a number as the interface does |

Plugins must **read** the project, not modify it.

## Export plugin

```python
from openkick.plugins import ExportPlugin


class MyFormat(ExportPlugin):
    id = "my_format"
    name = "Export to my format"
    file_filter = "Text (*.txt)"
    suffix = ".txt"

    def export(self, ctx, path):
        with open(path, "w") as f:
            for m in ctx.project.measurements:
                f.write(f"{m.name}\n")
        return "Done."          # message for the user ("" = no message)


PLUGINS = [MyFormat()]
```

The built-in plugins (`openkick/plugins/builtin/`) are complete examples.

**Security:** a plugin runs code with the user's permissions. Only install plugins from sources you
trust.

## Share it: a repository, a ZIP and the plugin index

To share a plugin, start from the
[plugin template](https://github.com/SlightlyDazed/openkick-plugin-template) (**Use this template**
or fork it). It holds a working example, tests on a made-up rider and a `plugin.json`, the file
OpenKick reads to install it:

```json
{
  "id": "my-plugin",
  "name": {"en": "My analysis", "es": "Mi análisis"},
  "description": {"en": "What it measures."},
  "version": "1.0.0",
  "author": "Your name",
  "license": "GPL-3.0-or-later",
  "openkick": ">=0.1.0",
  "category": "Analysis",
  "kind": "plugin",
  "repository": "https://github.com/you/my-plugin",
  "templates": []
}
```

`id`, `name`, `version`, `author` and `openkick` are required. Next to `plugin.json` there must be
the `plugin.py` with the `PLUGINS` list.

**Installing.** On the repository's page, *Code › Download ZIP*; in OpenKick, *Plugins › Install
plugin (.zip)…*. OpenKick finds `plugin.json` (at the top of the ZIP or in the one folder GitHub
puts everything in), checks the fields and the OpenKick version it needs, and copies the plugin to
the plugins folder without its `tests/` and `.github/`. The templates listed in `templates` go to
the templates folder. Installing a plugin that is already there asks before replacing it.

**The index.** The community's plugins are listed at <https://slightlydazed.github.io/openkick/>
(the page is `site/` in this repository). To add one, open a pull request that adds its entry to
`site/plugins.json`; it is reviewed before it is merged.
