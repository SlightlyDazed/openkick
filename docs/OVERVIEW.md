# OpenKick: what it is and how to use it

OpenKick is a free, open-source desktop program for analysing human movement in video. It detects the
person's skeleton automatically in every frame, measures angles and trajectories, and lets you fix by
hand whatever the detection gets wrong. It is designed for health and sports professionals who do not
program.

Developed by Gabriel Alexander Laurente (SlightlyDazed). GPLv3 license. Current version: 0.1. Status as
of 26 October 2026.

## The problem it solves

The most widely used program for this is Kinovea, which is also free and open source and is the
reference in many clinics and sports teams. In Kinovea the user places the measurement points by hand
and the program follows them from one frame to the next, which gives full control over every point.

OpenKick takes a complementary approach: a neural network detects the whole body in every frame, and
every measurement is attached to that skeleton. The user does not have to place points; they only
correct the ones the detector got wrong. When a joint is corrected, every angle and trajectory that uses it is recomputed
on its own.

## What it can do today

| Part | What it does | Status |
|---|---|---|
| 2D analysis | One video: automatic skeleton, angles, trajectories, distances, phases, notes, results with charts | Works; the core of the program |
| 3D analysis | Two to six videos of the same movement: reconstructs it in 3D without calibrating the cameras | Experimental |
| Assisted 3D correction | List of points to review, bone-length guides, the other videos follow your fix, learn the error of a point over the pedal stroke | New |
| Cycling and gait | Every pedal stroke or stride with its phases, in 2D and 3D; in 3D, both legs and the sideways motion | 3D cycling not validated in a lab |
| Comparison | Several analyses side by side (two saddle heights, before and after, two people), in 2D and 3D | New |
| Patient record and list | Personal data, body-mass index, lengths measured with a tape and the equipment; patients saved to pick them in another analysis | New |
| Reports | Assessor notes and a PDF, text or JSON report, for 3D and for comparisons | New |
| OpenSim model | Turns the 3D motion into a bone model with joint angles (step 2; what is measured is still the detection) | Experimental |
| Compatibility | Reads and writes Kinovea `.kva` files | Works |
| Extensions | Plugin store, analysis templates without code and Python plugins | Works; the store does not download anything yet |

## How to use the 2D analysis

1. **Open a video.** Drag it into the window or choose it with *Open video*. Skeleton detection starts
   by itself.
2. **Review the skeleton.** The timeline marks doubtful frames in orange; the Tab key jumps to the next
   one.
3. **Correct.** Drag the misplaced point to its place. The correction can apply to that frame only, be
   spread over the neighbouring frames or be applied to a whole phase. While correcting, the skeleton
   of the previous and the next frame is shown as a ghost, to keep the point aligned with them.
4. **Measure.** With the *Angle* tool, click a knee, hip, elbow, shoulder or ankle. With *Trajectory*,
   follow any joint. With *Marker*, mark a point that is not on the skeleton, such as a ball.
5. **Mark the phases** of the movement (for example run-up, plant, strike) on the timeline.
6. **Calibrate** with a known length if you want speeds in metres per second.
7. **View the results** and export them.

To avoid repeating these steps every time there are **templates**: Cycling, Gait, Kick, Squat and
Vertical jump already include their angles, trajectories and phases. You can also create your own
templates in the program, without programming. All of them work in 3D as well.

The person's data go in **Analysis › Patient record…**: height, weight (with the body-mass index),
reason for the visit, pain, the lengths measured with a tape and the equipment (for example the saddle
height). They appear in the PDF report and in the comparison. A patient already registered is chosen
at the top, under **Saved patient**, and their record fills the form.

## How to use the 3D analysis

It is in the home sidebar (*3D analysis*) and in *Analysis › 3D from several videos (experimental)…*.

1. **Drop the videos** into the window, two to six, type the person's height and choose the activity:
   cycling, gait, a template, or *Automatic* to let the program tell whether the person is pedalling or
   walking. This is also where a **saved patient** is chosen or their record is filled in.
2. **Reconstruct.** The program detects the skeleton in each video, synchronises the videos with each
   other (by the sound if there is any, otherwise by the motion) and works out where each camera was
   from the body itself. No calibration board or markers are needed.
3. **Step 1, correct the points.** The videos are shown side by side with their skeletons. The program
   flags the doubtful points and says why (the videos disagree, a jump, a bone that changes length).
   Each video can be opened on its own to correct it frame by frame. The correction tools are described
   in the next section.
4. **Step 2, OpenSim model.** The program scales a bone model to the person and computes their joint
   angles. It is shown next to the 3D points.
5. **Save and export:** the session, the assessor's **notes**, a **report** in PDF, text or JSON
   (patient, results, how much to trust them, images and those notes), the OpenSim files or a video.

With an activity chosen, the timeline at the bottom shows every pedal stroke or stride with its phases,
and the right-hand panel shows its numbers. In cycling, the 3D also measures what a side camera cannot
see: the knee moving in and out, the pelvis rocking, the trunk swaying and both legs at once.

The right-hand panel also holds **3D smoothing** (removes one- or two-frame jumps and the jitter;
*Light* or *Off* for fast movements) and rigid feet — **Rigid, in one piece (stiff sole)** — optional, for hard soles.

### Tools for correcting in 3D

- **List of points to check…** Every doubtful point in stretches, from worst to best, with a cropped
  view of the video: *Go to it*, *Edit it in its video*, *It is fine* (stops being flagged) and *Next point to check*.
- **Scope of the correction:** that frame only, the neighbours too, or the whole doubtful stretch.
- **The cameras do not move while correcting.** Before, every drag repositioned them and threw the
  other videos off. They are repositioned only with *Place the cameras again*.
- **The other videos follow my correction** (on by default): when a point is released, in the other
  videos it moves the minimum needed to agree with it in 3D. What was placed by hand never moves. If
  another video would have to move too much (it disagrees), it stays and the message says so.
- **Length guides.** If a thigh or a shank deviates 5 % or more from its length (the tape measure if it
  is in the record, otherwise the typical one in the video), a red line (too long) or a yellow one (too
  short) runs from the point to where it should be, and a strip at the bottom marks the frames.
  *Bring it where its bones say, in every video* places it there.
- **Learn its path over the pedal stroke.** For a point the detector always misplaces at the same
  moment of the stroke: correct it in the 8 parts of one pedal stroke and the program applies that
  correction to all the others.
- Marks on the points: light-green ring, placed by hand; dotted, moved to follow another correction;
  dashed green, placed by the learned path. Ctrl+Z undoes any correction, including the ones that
  spread on their own.

The full guide to this part is [`MULTIVIEW_3D.md`](MULTIVIEW_3D.md).

## How to compare analyses

It is in the home sidebar (*Comparative analysis*) and in *Analysis › Compare analyses…*

1. **Add analyses…** or drop the files into the window: 2D projects (`.okick`) and 3D sessions
   (`.ok3d`), mixed if needed, as many as you like.
2. Name each one on its card ("Saddle 93", "Before"...). The first one is the reference; the ◀ ▶ arrows
   change the order.
3. Choose **People:** *The same person* (two settings, before and after) or *Different people*. With
   different people, lengths are given as a percentage of each person's height.
4. Read, leg by leg, the table of measurements and the mean knee, hip and ankle curves over the cycle,
   with their spread. A difference is marked **●** when it is larger than what changes from one pedal
   stroke (or stride) to the next.
5. **Save…** stores the comparison in an `.okcmp` file; **Export CSV…** writes every table for a
   spreadsheet; **Notes…** stores the assessor's name and observations (in the `.okcmp`); **Report ▾**
   creates the PDF, text or JSON report with the analyses, the patients, the tables with the marked
   differences, the mean curves and those observations. The JSON carries every number unrounded and
   whether each difference exceeds the cycle-to-cycle variation.

## Patients

In the home sidebar (*Patients*) and in *Analysis › Patients…*. Every record saved in an analysis goes
into this list, **on this computer only** (it is not sent anywhere). For each patient you see their most
recent record and the analyses they appear in (2D and 3D):

- **Open** an analysis, or **Compare them** all in the comparison window.
- **Edit the record…** to bring it up to date, or **New patient…**
- **Take off the list:** the patient leaves the list; their analyses are not touched.

Each analysis keeps its own copy of the record as it was that day: if the patient's weight changes,
earlier analyses keep the weight of then. The list lives in the OpenKick data folder (`patients.json`).

## Plugin store

In the home sidebar (*Plugin store*) and in *Plugins › Manage plugins…*. Each package brings one kind of
analysis: its plugin, its template and, if it has one, its 3D. For now they all come installed with
OpenKick: **Remove** hides a package and **Install** shows it again. Later they will be downloaded from an
online store. The basic reports (range of motion, speeds, export to OpenSim) are part of the program
and cannot be removed.

## How to record so that it works

- The whole person inside the image during the whole movement.
- Camera on a tripod, no zoom, with the phone's main lens.
- Preferably at 60 frames per second, especially for fast movements.
- **For 2D:** the camera perpendicular to the movement, that is, side-on. The further it moves from
  side-on, the larger the measurement error.
- **For 3D:** all cameras recording at the same time, from different angles. They do not need to start
  recording at the same instant.

## How accurate it is

It was measured against a marker-based motion-capture lab using a public dataset (OpenCap, Stanford
University): 9 people, phones on tripods at 60 frames per second, doing squats, sit-to-stand, drop
jumps and walking. Everything measured is the automatic result, without any manual correction.

Mean error in degrees against the lab:

| Joint | 2D, one camera | 3D, two cameras | OpenSim model |
|---|---|---|---|
| Knee | 5 to 10 | 4.3 | 3.7 |
| Hip | 7 to 10 | 7.6 | 9.4 in flexion |
| Ankle | 13 to 20 | 5.1 | 6.3 |
| Shoulder | not measured | 5.3 | not measured |
| Elbow | not measured | 14.4 | not measured |

What these numbers say:

- **The knee is the most reliable** in all three cases.
- **The ankle can only be measured with two cameras.** With one it is not reliable.
- **The elbow is not valid yet:** it reads about 14° less, almost constantly.
- **The OpenSim model's hip flexion is too high**, by 9 to 16° in squats, sit-to-stand and jumps.
- **For gait it cannot compare two people with each other:** the error is the same size as the
  differences between them.
- In 2D the program tends to measure less flexion than the real one, and the error grows when the camera
  is not side-on.

What is not measured:

- How much the result improves with manual correction, which is the program's central function.
- Fast sports movements, such as kicks, serves or throws.
- Hand-held cameras, other phones and real 30 frames-per-second recordings.
- **Cycling.** There is no public lab data with pedalling video. In a real session (three phones), the
  right hip point slides about 7 cm towards the knee every time the leg goes up, the same in all three
  videos. Measuring it needs a reference of its own: stickers on the greater trochanter, the lateral
  femoral condyle and the lateral malleolus, or a goniometer with the pedal stopped.

**About the reference.** In that data, the lab's "knee" and "ankle" are skin markers outside the joint.
Against the angles the lab computes with its own OpenSim (the gold standard), the OpenKick 3D knee is
off by 4.0–4.7° and the OpenCap two-camera knee by 3.1–3.4°.

**What needs validating and what does not.** MediaPipe, RTMPose, Pose2Sim and OpenSim have their own
studies and are cited. What has to be measured is what OpenKick adds: calibrating with the body,
synchronising, combining the cameras, smoothing, correcting. And as a tool for therapists: whether two
assessors get the same result (reliability), whether it repeats on another day, how long it takes and
whether they find it easy to use.

Tried and removed: measuring with the OpenSim model instead of the detection (worse knee and ankle;
worse still with the tape measurements), fixing bone lengths by moving the points along them (makes the
pelvic roll explode) and leaving out a camera whenever the others agree (caused one-frame jumps).

Every measurement is detailed in [`VALIDATION.md`](VALIDATION.md).

## Files it handles

| File | What it is |
|---|---|
| `.okick` | OpenKick 2D project: skeleton, corrections, measurements, phases and notes |
| `.ok3d` | 3D session: the points of each video, the corrections, the camera positions, the activity and the patient record |
| `.okcmp` | Comparison: which analyses are compared and under which name (it does not copy the analyses) |
| `.json`, `.txt` | Report of a 3D session or a comparison: the numbers unrounded (JSON) or the report as text |
| `patients.json` | The patient list of this computer, in the OpenKick data folder |
| `.kva` | Kinovea annotations; OpenKick reads and writes them |
| `.csv` | Table with the measurements frame by frame, for Excel or statistics |
| `.pdf` | Report with charts, ranges of motion and a per-phase table |
| `.mp4` | Video with the drawings on top, or the comparison of the 3D with the OpenSim model |
| `.trc`, `.osim`, `.mot` | Markers, model and motion to open in OpenSim |

The original videos are not copied into the projects: the project stores where they are.

## How to install and open it

It needs Python 3.10 to 3.12. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate              # on Windows: .venv\Scripts\activate
pip install -e ".[multiview,opensim]"  # without the brackets only the 2D part is installed
openkick
```

- The first time each part is used, its detection model is downloaded: about 30 MB for 2D and about
  150 MB for 3D.
- There is a Windows installer, but it is from an earlier version: it does not include the 3D or the
  latest improvements.
- The interface is available in 10 languages: Spanish, English, German, French, Italian, Portuguese
  (Brazil), Russian, Japanese, Korean and Simplified Chinese.

## Where the project stands

Done: the full 2D analysis, the multi-video 3D, the step to OpenSim, the validation against a lab with
public data, cycling and gait in 2D and 3D, the comparison of analyses, the patient record and list,
notes and reports (PDF, text, JSON), 3D smoothing, the 3D correction tools and the plugin store (no
downloads yet).

Still to do, in order of importance:

1. Validate cycling with a reference of its own (stickers on the trochanter, knee and ankle, or a
   goniometer with the pedal stopped).
2. Measure the effect of manual corrections, and check the learned pedal-stroke path with real
   corrections by a physiotherapist on a pedal stroke it has not seen.
3. A study with therapists: inter-rater and test–retest reliability, analysis time and ease of use,
   compared with Kinovea or the goniometer. It needs a supervising professor and ethics-committee
   approval.
4. Validate sports movements, which requires access to a lab.
5. Update the Windows installer and test the program on modest computers.

## Words used in the program

- **Frame:** each of the images of the video.
- **Skeleton:** the body points the program detects, joined by lines.
- **Detection engine:** the neural network that finds those points. 2D uses MediaPipe (33 points); 3D
  uses RTMPose (26 points, with the feet).
- **Phase:** a named stretch of the movement, such as "stance" or "strike".
- **Range of motion:** the difference between the largest and the smallest angle of a joint.
- **Calibrate:** in 2D, telling the program how many pixels make a metre. In 3D it is not needed: the
  person's height is enough.
- **OpenSim:** free software from Stanford University for simulating the musculoskeletal system.
- **Inverse kinematics:** the calculation that finds which angles a bone model must have to match the
  measured points.

## More detail

All documents are in the `docs/` folder:

| Document | Content |
|---|---|
| [`OVERVIEW.md`](OVERVIEW.md) | This summary: what it is, how to use it and where it stands |
| [`MULTIVIEW_3D.md`](MULTIVIEW_3D.md) | Guide to the 3D analysis, its correction tools and the OpenSim model |
| [`TEMPLATES.md`](TEMPLATES.md) | How to create analysis templates |
| [`PLUGINS.md`](PLUGINS.md) | How to write a Python plugin |
| [`VALIDATION.md`](VALIDATION.md) | Every accuracy measurement and how to repeat it (scripts and results are in `validation/`) |
