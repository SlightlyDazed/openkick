# Analysis templates

A template says what to measure in a movement and which phases it has. It is created without
programming from *Analysis › New analysis template…*, or with *Create your own* on the home screen.

When a video is opened with a template, OpenKick creates its angles and trajectories as soon as the
detection finishes. When you press *Mark phase here* (F), it suggests the template's phase names in
order.

## File

Templates are small JSON files you can share. They are stored in the `templates` folder of the user
data (`~/.local/share/openkick/templates` on Linux, `%APPDATA%\OpenKick\templates` on Windows). To
install a template someone sends you, copy it there.

```json
{
  "id": "kick-right",
  "name": {"en": "Kick · right leg", "es": "Patada · pierna derecha"},
  "description": {"en": "Knee and hip of the kicking leg", "es": "Rodilla y cadera de la pierna de golpeo"},
  "angles": [
    ["right_hip", "right_knee", "right_ankle"],
    ["right_shoulder", "right_hip", "right_knee"]
  ],
  "trajectories": ["right_ankle"],
  "phases": ["Run-up", "Plant", "Strike", "Follow-through"]
}
```

- `angles`: three points `[end, vertex, end]`.
- `trajectories`: points whose path and speed are tracked.
- `name`, `description` and each phase can be plain text or a dictionary by language code
  (`{"en": "...", "es": "..."}`).

Point names: `nose`, `left_`/`right_` + `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`, `heel`,
`foot_index` (tip of the foot), `eye`, `ear`, `index`, `pinky`, `thumb`.
