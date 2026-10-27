# 3D from several videos (experimental)

This tool reconstructs the movement in 3D from two or more videos of the same movement, recorded at the
same time from different places. Nothing has to be calibrated and no markers are needed: OpenKick works
out where the cameras were from the body itself. It has two steps:

1. **Correct the points:** the videos side by side with their skeletons; whatever is wrong is dragged
   to its place, as in the normal editor.
2. **OpenSim model:** the 3D points next to the OpenSim bone model they produce.

It is in **Analysis › 3D from several videos (experimental)…** It opens in a separate window, closed
with the **X** at the top right.

## How to record

- At least two cameras and at most six, recording **at the same time** the same movement. They do not
  need to start recording together: the program synchronises the videos itself. If the phones record
  sound it uses it first (a clap at the start makes it safer); if the sound is not enough, it
  synchronises by the motion of the body.
- Different angles, for example one from the side and one from the front. The more similar the
  viewpoints, the worse the depth.
- The whole person inside the image in every camera during the whole movement.
- Preferably on a tripod. A hand-held camera is compensated, but the result is worse.
- No zoom, with the phone's main lens.
- The person's height has to be known: it gives the reconstruction its real size. If the thigh or the
  shank is also measured with a tape (see *Patient record* below), the size comes from those
  measurements, which are more accurate.
- Phones upright and straight. The program takes the cameras' vertical as "up": with a tilted phone,
  angles relative to the vertical (trunk, foot) come out tilted.

## Choosing the videos

Drag the files onto the box, all at once or one by one; dropping them anywhere in the window also
works. Or click the box to choose them. Each video becomes a card with its thumbnail; the card's **×**
removes it. The first video in the list is the reference: the others are synchronised and positioned
relative to it.

Type the height and choose the **Activity**:

| Activity | What is analysed |
|---|---|
| Automatic | The program tells by itself whether the person is pedalling or walking. |
| Cycling | Every pedal stroke of both legs (see *Cycling in 3D*). |
| Gait | Every stride with its eight phases (Perry). |
| A template (Kick, Squat, Vertical jump or your own) | The template's angles measured in 3D. |
| Other (correction only) | Nothing: only the reconstruction and its correction. |

**Patient record…** opens the person's data (see below). Everything is optional.

Press **Reconstruct in 3D**. The first time, the detection model is downloaded (about 150 MB).

## Patient record

It is opened before reconstructing (**Patient record…**) or afterwards, with the **Patient** button at
the top. At the top, **Saved patient** lets you choose someone already registered in another analysis:
their most recent record fills the form, and you only update what changed (today's weight, the saddle
height). A record from before the list existed, with the same name as a single saved patient, is linked
to them without changing what was written; when it is saved and the analysis is saved, that analysis
appears with the patient. It has three tabs:

- **Person:** name or code, sex, date of birth (computes the age), height, weight, dominant side,
  sport, reason for the visit, pain from 0 to 10 and where, and notes. With height and weight it
  computes the **body-mass index** and its WHO class.
- **Measured lengths:** what was measured on the person with a tape, left and right. Each measurement
  says from which point to which point it is taken:

  | Measurement | From → to |
  |---|---|
  | Thigh | greater trochanter → knee joint line (outer side) |
  | Shank | knee joint line → lateral malleolus |
  | Lower-limb length | ASIS (anterior superior iliac spine) → medial malleolus |
  | Foot | heel → tip of the longest toe |
  | Upper arm | acromion → lateral epicondyle |
  | Forearm | lateral epicondyle → radial styloid process |

  Also shoulder and pelvis width, and the inseam. If the two legs do not measure the same, it says by
  how much and warns that the tape itself has a ±1 cm error. **The measured thigh and shank give the 3D
  its real size**: the same bone measured with a tape and in the reconstruction.
- **Equipment:** what the person uses. In cycling: saddle height and setback, crank length, reach and
  drop to the handlebar, frame size, cleats or pedals. In gait: footwear, orthotic insoles and walking
  aid. There are three free rows for anything else.

The record is saved inside the `.ok3d` session (and inside the `.okick` project in 2D), appears in the
PDF report and in the comparison of analyses.

## Cycling in 3D

With the *Cycling* activity (or *Automatic* and a person pedalling), the program splits every pedal
stroke of each leg from the knee angle:

- The timeline at the bottom shows the pedal strokes, divided into **Recovery** (the foot goes up) and
  **Propulsion** (the foot goes down).
- The right-hand panel shows the cycling card: cadence, knee at the bottom and at the top, ankle and
  foot at the bottom.
- **See the pedal stroke analysis** opens the full report, with four tables: *Position*, *Pedalling*,
  *Left and right* and *Pedal stroke by pedal stroke*.

Besides what 2D measures, the 3D sees what happens sideways, which a side camera cannot:

| Measurement | What it is |
|---|---|
| Knee lateral sway | How far the knee moves in and out on each pedal stroke (cm) |
| Knee towards the bike | Which side of the hip–ankle line the knee travels on, on average: positive towards the frame, negative outwards (cm) |
| Pelvic roll | How much the line of the hips tilts from side to side on each pedal stroke (degrees) |
| Trunk lateral sway | How much the shoulders move from side to side relative to the pelvis (cm) |
| Hip bounce | How much the hip rises and drops on each pedal stroke (cm) |

And both legs at once: every left measurement next to the right one.

## Gait and other templates in 3D

- **Gait:** every stride with its eight Perry phases, as in 2D. Phases whose start is not clearly
  visible are placed at their usual position and marked as estimated.
- **Any other template:** its angles measured in 3D, with minimum, maximum, range, mean and when they
  occur. It opens with **See the angles in 3D**.

## Step 1 · Correct the points

1. The videos appear side by side, each with its skeleton, and on the right the 3D skeleton, which
   rotates when dragged.
2. Press **Next point to check** (or the Tab key). The program takes you to the next doubtful point and
   tells you why it is doubtful.
3. Look at that point in every video. If in any of them it is not on the body, **drag it to its place**.
   The 3D is recomputed immediately.
4. **Save** keeps the session with your corrections (`.ok3d` file).

| On screen | Meaning |
|---|---|
| Coloured skeleton | What the detector saw in that video. This is what you drag. |
| Thin white lines | The 3D skeleton seen from that camera. If it separates from the body, something is wrong. |
| Orange or red ring | Doubtful point (more than 6 cm; red, more than 12 cm). |
| Lime-green ring | A point you placed by hand. |
| Hollow grey point | Not used in that video: the detector was not sure, or the other videos contradict it. If you drag it into place, it is used. |
| Dashed line while dragging | In the other videos, the line on which that same point has to fall. |
| Bars at the bottom | How much doubt there is in each frame. Click to go to that moment. |

### Editing one video on its own

Each video has an **Edit this video** button at its top right. It opens it full size to correct it
calmly, as in the 2D editor:

- The ← → arrows, the previous and next buttons and the timeline move **frame by frame through that
  video**: each bar of the timeline is one of its frames, so none is skipped. With all the videos at
  once, by contrast, each step is a frame of video 1, and the other videos show the closest frame in
  time.
- The **shadow of the skeleton** of the previous frame (amber) and the next one (blue) is shown, to
  line the point up with its neighbours. It is turned off with **Shadow** or the G key.
- The 3D on the right and the doubtful points are updated with every correction.
- **Back to all the videos** (or Esc) returns to the joint view. Nothing has to be saved or loaded: the
  corrections are already part of the 3D session.

Only correcting happens here. 2D angles and trajectories are measured in the normal editor; the 3D ones
come from the OpenSim model in step 2.

A point is flagged as doubtful for three reasons:

- **The videos disagree:** the cameras place it in different spots. With three or more cameras at
  least two have to deviate from the result; a single one is corrected by the others.
- **A bone changes length:** a thigh cannot measure 43 cm in one frame and 52 in the next.
- **It jumps:** it is far from where the previous and the next frame place it.

The last two catch what the first cannot see: a point displaced exactly in the direction another camera
is looking.

Other corrections:

- **Right-click on a point:** go back to the automatic detection, do not use that point, or swap left
  and right if the detector confused the legs or the arms.
- **When moving a point › Also the neighboring frames:** the correction is spread, more and more
  softly, over the 3 previous and the 3 next frames.
- **When moving a point › The whole doubtful stretch:** the same correction reaches every frame of the
  stretch the program flagged as doubtful for that point, and fades out over 3 frames on each side. It
  is useful when the detector makes the same mistake for several frames in a row: you correct one and
  the stretch is corrected.
- **Synchronization:** choose a video and move it forwards or backwards by a quarter of a frame with −
  and +.
- **Ctrl+Z** undoes any correction.

### Thigh and shank lengths

With **Lengths** (at the top, on by default), when a thigh or a shank deviates 5 % or more from its
typical length in the video, each video shows which point to move and how far:

- **Red:** the bone is too long; the red line runs from the point to where it should be.
- **Dotted yellow:** the bone is too short; the line marks what is missing.
- The circle at the end of the line is **the spot to drag the point to**. It is computed in 3D and shown
  in every video, so in the video that is wrong the line is long and in the ones that are right, short.
- The point to move is the one the cameras see with the least confidence. If a knee makes the thigh and
  the shank fail at once, the mark is the spot that fixes both.
- When the point is selected or dragged, a label gives the measurement: "Right thigh 53.8 cm · typical
  40.5 · +33 %".

In the bar at the bottom, the **Lengths** strip colours each frame by how far the worst bone deviates
(yellow from 5 %, orange from 10 %, red from 15 %); with a point selected, only its bones. Hovering
shows how much.

If the line is equally short in every video even though the bone is far off, no video is clearly
wrong: the error comes from how the cameras are combined, and dragging helps little. This happens with
the right thigh of the 93 cm cycling session (4–6 cm in each video), and not with the left shank, where
the rear video is the one that fails (12 cm).

### List of points to check

On the *Review* card, **List of points to check…** opens a window with every doubtful stretch, from the
worst to the mildest. Each row is one point over a few consecutive frames:

| Column | What it says |
|---|---|
| Point | Which joint |
| Video | The video that disagrees most there |
| Frames | The exact frames of the stretch, in that video |
| Time and Worst | When, and how many centimetres in its worst frame |
| Reason | The videos disagree, a bone changes length, or the point jumps |
| Status | To check or fine |

Selecting a row shows that frame of the video: in orange, what the detector saw; in white, where the 3D
places it. **Go to it** (or double-click) takes the 3D window to that frame with the point selected,
ready to drag. **Edit it in its video** opens that video on its own, full size. If the point is right,
**It is fine** removes it from the list and from *Next point to check*; it is saved with the session and
can be undone with *Mark it to check again*.

A point that is wrong for several consecutive frames only "jumps" at the edges; that is why two warnings
of the same point less than 0.15 s apart are merged into a single stretch.

### Correcting does not move the cameras

A correction changes the 3D point, but it no longer repositions the cameras (before, each drag moved
them a little, and with them all the other points: fixing one video seemed to throw another off). To
reposition them with all your corrections: **Place the cameras again**, under *Synchronization*.

Under *Selected point*, **Bring it where its bones say, in every video** places the point, in every video
at once, where its thigh and shank recover their typical length (the circle of the length guides).
Ctrl+Z undoes it.

### The other videos follow your correction

Under *When moving a point*, **The other videos follow my correction** (on by default). When you release
a point in one video, in the others it moves to where your correction places it in 3D, so they do not
fall out of line:

- **Your point rules.** The 3D stays on the line that leaves that camera through your point; the other
  videos decide the depth. If they were right, they move very little (in the 93 cm session, about 5 px).
- **What you placed by hand never moves.** If you correct the same point in two videos, the 3D comes
  from those two and the third follows them.
- Points moved this way carry a **dotted ring**. A single Ctrl+Z undoes everything.
- **If another video does not agree with your correction** (it would have to move more than 3 % of the
  person's height in that video), it does not move there and the message says so: check the point in
  that video. It happens, for example, when moving a point to the length guide while the other two
  videos see it somewhere else.
- Points that follow do not count for repositioning the cameras nor for the "videos disagree" figure of
  the report: they agree because the program moved them.

### Learning a point's path over the pedal stroke

For a point the detector misplaces **always at the same moment of the pedal stroke** (in the 93 cm
session, the right hip slides about 7 cm towards the knee every time the leg goes up), correcting it
frame by frame is not feasible. Under *Selected point*, with a pedalling video:

1. Correct that point, in one video, in the 8 parts of **one** pedal stroke. The panel says how many you
   have done and which are missing (top, going down, bottom, going up).
2. **Learn its path over the pedal stroke and apply it to every stroke.** The program learns how wrong
   the detector is at each moment of the turn and corrects every pedal stroke of that video the same
   way. Your points are not touched; the ones corrected this way carry a dashed green ring. The other
   videos follow the correction if that is turned on. One Ctrl+Z undoes it.

Only repeat what you have checked. The program copies your correction; it does not know whether it is
right. In the 93 cm session, putting the hip "where the thigh measures the tape length, along the thigh"
doubles the pelvic roll (from 11.5° to 22–25°), so that is not where it belongs. **What is reliable is a
sticker on the greater trochanter** (and others on the lateral femoral condyle and the lateral
malleolus) when recording: that shows where the point really is.

The length guides now compare with the **tape measurement** when the patient record has it (the label
says "tape"), and otherwise with the typical length in the video. In pedalling the typical one can be
misleading: if the hip slides with the leg up, the median falls between the two values. Note: the
guide's circle moves the point along the bone, and for the hip in pedalling that is not always the right
direction.

### Rigid feet

Optional, below the bones: **Rigid, in one piece (stiff sole)**. Heel, toe and little toe keep one shape
(the typical one in the video) and rotate together about the ankle, like a shoe with a stiff sole. The
ankle is not touched, so the leg stays the same.

- **The foot still rotates as much as it does:** the ankle angle and the foot inclination are measured
  the same; only its shape stops changing. In the 93 and 95 cm saddle sessions, the change of the ankle
  between one and the other holds (+11.6° free, +11.2° rigid), and that of the toe goes from +9.1° to
  +14.0°, closer to what the side camera measures in 2D (+14.3°).
- **For cycling shoes or any stiff sole.** With soft shoes the forefoot bends when pushing the pedal, and
  a rigid foot would not show it.
- With rigid feet, the list of points to check no longer flags heel and toes because "a bone changes
  length": that shape is now fixed by the program. They are still flagged if they jump or if the videos
  disagree.

### 3D smoothing

At the bottom of the panel, **3D smoothing** cleans the 3D skeleton. It does two things: it removes jumps
of one or two frames (a point that goes far away and comes back) and the tremor faster than the movement
(Butterworth filter, the standard in biomechanics).

| Option | What for |
|---|---|
| Off (the points as triangulated) | The points as they come out of the cameras. |
| Light (12 Hz) | Fast movements: kick, jump. It only removes the tremor; it does not touch jumps, because in a very fast movement a real peak can look like one. |
| Normal (8 Hz) | Cycling, gait, squat. The default option. |
| Strong (6 Hz) | Slow movements with a lot of tremor. |

With the kick or jump template it starts at Light; with the others, at Normal. Smoothing does not change
the detection or your corrections, and doubtful points are still searched in the unsmoothed 3D: an error
the smoothing hides still appears in *Next point to check*. What is measured, exported and shown in the
3D is the smoothed version. It is saved with the session.

## Notes and report

- **Notes**, at the top: the assessor's name (remembered for next time) and their **observations and
  recommendations**. They are saved with the session and appear in its report.
- **Report**, next to it: creates the report on the spot, in the format chosen from the menu:
  - **PDF report…:** to read and hand over (described below).
  - **Text report (.txt)…:** the same report as plain text, with aligned tables, without images.
  - **Research data (.json)…:** every number unrounded and with its unit (every pedal stroke or stride
    with its measurements, the median and spread of each, the mean curves from 0 to 100 % of the cycle,
    the quality, the settings and the videos). The measurement keys are fixed and in English, so that a
    program reads them the same in any language; the `about` section explains each field.

  Below, in the same menu, what it includes (remembered): each pedal stroke or stride as an appendix,
  the images, the frame-by-frame values in the JSON, and **without what identifies the patient**
  (removes name, date of birth and notes; the age stays), to share the data in a study.

The PDF contains:

1. the patient record;
2. the assessor's observations, with their name;
3. the recording: each video, its size, frames per second, offset and whether the camera motion was
   compensated, and the settings (smoothing, bones, feet);
4. how much to trust it: the quality, how much the videos disagree and the bones change, the points
   corrected by hand and the doubtful stretches still to check; if a leg is doubtful, it says so;
5. what was found: the activity's tables and, in cycling, the mean pedal stroke of knee, hip and ankle of
   both legs;
6. one moment (in cycling, the bottom of a typical pedal stroke) in each video and in 3D;
7. as an appendix, every pedal stroke or stride (optional);
8. the limits of the 3D analysis.

The comparison has its own PDF (see *How to compare analyses* in [`OVERVIEW.md`](OVERVIEW.md)).

## Saving and opening again

- **Save** (Ctrl+S) writes the session to an `.ok3d` file: the points of each video, your corrections,
  the synchronisation and the camera positions. The first time it asks for a name; afterwards it
  overwrites that same file.
- The arrow next to Save offers **Save as…** (Ctrl+Shift+S) to keep another copy under another name. The
  previous file is not touched.
- The session's name is shown at the top, next to the title. A dot (•) next to it means there are
  unsaved corrections; when closing or opening another session the program asks before losing them.
- The videos are not copied into the file. If you move the whole folder (session and videos), the
  session finds them by itself. If you move or rename a single video, on opening it asks where it is and
  checks that it is the same one.
- A session is opened from **Open a saved 3D session…**, by dropping the `.ok3d` into the window, or
  from the OpenKick home screen (*Open project* or the recent list).
- What you export (the `.trc` and the OpenSim folder) takes the session's name.

## What more than two videos add

- A point one camera does not see (hidden by the body) is reconstructed from the others.
- If a camera places a point more than 12 cm away from where the others put it, its error is flagged. It
  is left out of the 3D point only when, with it, the point jumps off the trajectory given by the
  neighbouring frames; otherwise all of them are averaged, which is the most accurate. Discarding it as
  soon as it disagreed meant that, with one camera at the side, one behind and one in front, the side one
  was lost in single frames and the point jumped (the rear and the front cameras easily agree with each
  other, and what they see worst is exactly forwards and backwards).
- Left/right confusions of one video are detected by contrast with the rest.

## Step 2 · OpenSim model

Press **2 · OpenSim model**. The program sends the 3D points to OpenSim, which scales the model to the
person and computes the inverse kinematics. It takes a few seconds.

- On the left are the 3D points; on the right, the bone model. Both rotate together when either is
  dragged, and they follow playback.
- The panel shows the knee, hip and ankle angles of the current frame.
- If you go back to step 1 and correct points, the model is marked as out of date: **Compute the model
  again** brings it up to date.
- **Save the OpenSim files…** writes the model (`.osim`), the motion (`.mot`), the markers (`.trc`) and the
  bone shapes to a folder, ready to open in the OpenSim application.
- **Export to OpenSim**, at the top, saves only the 3D points (`.trc`).
- **Export video** saves an MP4 of the whole movement. There is one button per format, and each lets you
  choose real speed or slow motion (half or a quarter):
  - **3D points + OpenSim model:** the two halves of the screen, side by side, seen from the angle you
    have at that moment.
  - **Your videos + OpenSim model:** one column per video; at the top the video and below it the OpenSim
    model seen from where that camera was, with the reconstructed points in green. It is useful for
    checking by eye that the model does the same as the person.
  If you corrected points after computing the model, it has to be recomputed before exporting.

## Limits

- If a point is well placed in every video and still flagged, the cause is a very fast movement between
  frames (a hand in a serve, at 30 frames per second). Dragging does not fix it; recording at 60 frames
  per second or more does.
- Checked against a motion-capture lab only in squats, sit-to-stand, drop jumps and gait, with identical
  phones on tripods. The numbers, with two, three and five cameras, are in
  [`VALIDATION.md`](VALIDATION.md).
- How much the result improves with manual corrections is not measured.
- **3D cycling is not checked against a lab.** Only with two sessions of the same rider (saddle at 93 and
  at 95 cm), in which the 3D and the 2D side camera agree on what changed. With one camera behind and
  another in front (about 140° between them), the hip of the far side comes out worse: the comparison
  flags as doubtful the leg whose thigh or shank changes length by more than 10 %.
- In the OpenSim model, knee and ankle come out with 3–6° of error against the lab, but hip flexion comes
  out 9 to 16° too high in squats, sit-to-stand and jumps.
- Required components: `rtmlib` and `onnxruntime` for step 1; `pose2sim`, `opensim` and `vtk` for step 2.
  They are not yet included in the Windows installer.
