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
- `convention` (optional): how the standard joints (knee, hip, ankle, shoulder, elbow) read. Left out or
  `"clinical"`: 0° is neutral, with a sign (flexion +). `"interior"`: the angle inside the joint, 180° =
  straight, as the Cycling template uses. Angles between other points are always the angle inside.
- `hip_reference` (optional, clinical only): what the thigh and the upper arm are measured against. Left
  out or `"pelvis"`: the vertical tipped by three quarters of the trunk's lean for the thigh, the trunk's
  axis for the arm (best for squats, sit-to-stand and jumps). `"vertical"`: the vertical alone (best for
  gait, as the Gait template uses). See *Clinical angles* in
  [VALIDATION.md](VALIDATION.md).
- `cutoff_hz` (optional): the low-pass filter before speeds and accelerations, in Hz. Left out: 6 Hz,
  right for walking, squats and pedalling. Fast movements need more: the Vertical jump template uses 12,
  the Kick template 15. It is kept below half the video's frame rate.
- `min_fps` (optional): the frame rate the movement needs (60 in Jump and Kick). The velocities report
  warns when the video has fewer.

Point names: `nose`, `left_`/`right_` + `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`, `heel`,
`foot_index` (tip of the foot), `eye`, `ear`, `index`, `pinky`, `thumb`.
