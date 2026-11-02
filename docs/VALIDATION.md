# Validating OpenKick against lab data

The `validation/` folder measures how wrong OpenKick's angles are, compared with a motion-capture lab. It
uses the public OpenCap dataset (Stanford University): 10 people recorded at the same time with 5 phones
and with lab markers. **9 of them are used**, because one did not consent to publishing their videos.

## Summary

Mean absolute error in degrees against the lab, automatic output only (no manual correction), 9
participants, 72 recordings (squat, sit-to-stand, drop jump, gait):

| Joint | 2D · one side camera (MediaPipe) | 3D · two cameras [95 % CI] | OpenSim model, two cameras |
|---|---|---|---|
| Knee | 5.3–10.2 | 4.3 [3.5 to 5.3] | 3.7 |
| Hip | 6.7–9.9 (trunk–thigh) | 7.6 [6.6 to 8.6] (trunk–thigh) | 9.4 (flexion) |
| Ankle | 12.8–19.5 | 5.1 [4.5 to 5.8] | 6.3 |
| Shoulder | — | 5.3 [4.7 to 5.8] | — |
| Elbow | — | 14.4 [12.4 to 16.6], **not valid** | — |

- **Sources.** 2D: the range over the four movements in *Results (1 October 2026)*. 3D: all 72 recordings
  pooled, from *Statistics: intervals, agreement and reliability*, which also gives the bias, Bland–Altman
  limits of agreement, correlation and ICC. The per-movement 3D tables further down give 3.8–4.8° for the
  knee and 4.4–6.7° for the ankle; weighted by number of recordings they give the pooled value. OpenSim: *OpenSim angles
  against the lab*.
- **Dates.** The pooled 3D figures were computed on 2 October 2026. The reconstruction changes of 25 October
  moved the per-movement results by less than 1° (see that section).
- **Clinical angles since 1 November 2026.** The app now reports the joints with a sign from 0°, and the
  error is measured against the lab's OpenSim angles: 2D hip in gait 5.5° (it was 16.5° inside the joint),
  2D ankle 9–20° (it was 27–43°), 3D hip 4.2–7.3°, 3D ankle 4.3–7.5°. See *Clinical angles, gait events
  and speeds*.
- **Cycling is not validated.** It has only a consistency test: one rider, two saddle heights, phone videos
  compressed by a messaging app, no lab reference.
- **Tuning on the same data.** Three settings were checked on held-out participants (*Tuning on some
  participants and measuring on others*): no sign of inflated numbers. The later thresholds (12 cm to drop a
  camera, 4 cm off the trajectory, 8 Hz smoothing) were chosen on the cycling sessions and then checked on
  all OpenCap participants, not re-run on a held-out split.

The rest of this document is the detailed record, section by section and dated, of each measurement and
of what changed between them.

## How to reproduce it

1. Download `LabValidation_withVideos.zip` from `https://simtk.org/projects/opencap` (free account) and
   unzip it into `datos/opencap/` next to the `openkick/` folder (`datos` is the folder name the scripts
   look for by default). If it is somewhere else, set the folder in the `OPENKICK_OPENCAP` environment
   variable.
2. From the repository root:

```bash
python -m validation.opencap --engine mediapipe --trials squats1 --plots
python -m validation.opencap --engine rtmpose-balanced --trials squats1
```

Available engines: `mediapipe` (the one in the app), `rtmpose-lightweight`, `rtmpose-balanced` and
`rtmpose-performance` (these need `pip install rtmlib`). Results go to `validation/results/` and the
detections are cached, so repeating a run is immediate.

## What is compared

- **Reference.** The same angles OpenKick measures, computed from the lab's 3D markers projected onto the
  subject's sagittal plane. Knee: thigh–shank. Hip: trunk–thigh. Ankle: shank–foot.
- **OpenKick.** The angles obtained from the video of the two cameras that see the subject from the side
  (Cam0 and Cam4), using the leg closest to each camera.
- **Timing.** Video and lab are only roughly synchronised: an offset of up to 0.18 s remains in squat,
  sit-to-stand and jump, and of up to 0.56 s in gait. It is measured in 3D, by fitting the body
  reconstructed with two cameras to the lab markers (`python -m validation.multiview --batch ...` saves it
  to `validation/results/mocap_lag.json`), and corrected before computing the error.

## Results (1 October 2026)

9 participants (one did not consent to publishing their videos), 2 side cameras each, 4 movements:
squats (1 recording per person), sit-to-stand (1), drop jump (3) and gait (3). In total 144 videos per
engine. The full table is regenerated with `python -m validation.report` and is written to
`validation/results/summary.txt`.

Mean error in degrees against the lab, after correcting the time offset.

**Knee**

| Engine | Squat | Sit-to-stand | Jump | Gait |
|---|---|---|---|---|
| MediaPipe (current) | 6.7 | 7.9 | 5.3 | 10.2 |
| RTMPose lightweight | 6.9 | 8.5 | 5.2 | 7.9 |
| RTMPose balanced | 5.8 | 8.1 | 4.7 | 6.6 |
| RTMPose performance | 6.3 | 8.8 | 5.3 | 6.4 |

**Hip** (trunk–thigh angle)

| Engine | Squat | Sit-to-stand | Jump | Gait |
|---|---|---|---|---|
| MediaPipe (current) | 6.7 | 9.7 | 7.2 | 9.9 |
| RTMPose lightweight | 9.1 | 12.3 | 8.3 | 5.0 |
| RTMPose balanced | 7.9 | 11.0 | 7.8 | 5.1 |
| RTMPose performance | 7.9 | 11.9 | 8.3 | 5.0 |

**Ankle**

| Engine | Squat | Sit-to-stand | Jump | Gait |
|---|---|---|---|---|
| MediaPipe (current) | 19.5 | 19.0 | 16.8 | 12.8 |
| RTMPose lightweight | 15.8 | 15.8 | 12.1 | 11.8 |
| RTMPose balanced | 18.5 | 17.8 | 12.9 | 10.6 |
| RTMPose performance | 13.9 | 11.8 | 10.8 | 8.9 |

**Speed and size** (CPU only, 12-thread machine; RTMPose looking for the person every 10 frames):
MediaPipe 25 fps and 30 MB; RTMPose lightweight 48 fps and 43 MB; balanced 26 fps and 157 MB;
performance 8 fps and 596 MB.

What these numbers say:

- **For the knee in squat, sit-to-stand and jump, the four engines tie:** they differ by 1° or less. The
  largest model does not measure better.
- **In gait RTMPose wins:** knee 6.4–7.9° against MediaPipe's 10.2°, and hip 5° against 9.9°. The median
  correlation with the lab is 0.97 for the two larger RTMPose models and 0.83 for MediaPipe.
- **For the hip MediaPipe wins** in squat, sit-to-stand and jump (1 to 3° less error).
- **Gait is still the movement with the most failures:** between 17 % and 28 % of the recordings have more
  than 10° of knee error, because of the frames in which one leg hides the other.
- **There is an error common to all of them:** they measure less flexion than the lab (2–8° at the knee)
  and, in squat, sit-to-stand and jump, a range of motion 5 % to 12 % smaller.
- **That error grows the further the camera is from a true side view** with MediaPipe: with the camera
  less than 30° from the side view the knee error is 5.1°; between 40 and 50°, 7.6°. The camera angle is
  our own estimate from the apparent separation of the hips; it is not checked against the real
  calibration.
- **The ankle is not reliable with any of them:** 9 to 20° of error, largely a constant offset.

**Correction of 1 October 2026.** The first version of this table gave 12–14° of knee error in gait for
every engine. It was a bug in the test bench, not in the engines: it aligned the video with the lab by
looking for the best fit of the knee curve within ±0.25 s, but in 12 of the 27 gait recordings the real
offset is larger, and since gait repeats, it fitted half a cycle off. The offset is now measured in 3D,
which does not have that ambiguity. Squat, sit-to-stand and jump change by 0.2° or less.

## Limits of this measurement

- The dataset's "side" cameras are not quite side views: they sit about 30–40° from the sagittal view in
  squat, sit-to-stand and jump. A therapist who places the camera well should get less error than
  measured here; this is not verified.
- There are no kicks or other sports movements.
- The participants wear markers and straps on their legs, which may affect detection.
- The data are the raw output of each engine, without smoothing or manual correction, which is exactly
  what OpenKick adds on top.
- The cause of the common error is not fully established: the camera angle explains part of it.
- The thresholds of the method were tuned looking at this same dataset. Three of them were later checked
  on held-out participants (see *Tuning on some participants and measuring on others*); the ones added on
  25 October were not.

## 3D reconstruction with two cameras (1 October 2026)

`python -m validation.multiview --batch squats STS DJ walking`

This measures OpenKick's *3D from several videos* tool (`openkick/multiview/`), here with two cameras, the
way a person uses it: it receives only the videos and the height. It is compared with a ceiling:

- **Ceiling:** triangulation with the calibration that comes with the dataset (checkerboard).
- **OpenKick:** no calibration. The offset between videos, the camera positions and the scale are deduced
  from the two skeletons and the height. The lens is assumed to be a phone's main lens (26 mm
  equivalent). No manual correction at all.

The 3D points are scored with 3D joint angles against the lab markers. 9 participants, cameras Cam0 and
Cam4 (about 130° between them, 6 m apart). Mean error in degrees (knee / hip / ankle):

| Movement | Recordings | Ceiling | OpenKick | Camera orientation error | Synchronisation error (median) |
|---|---|---|---|---|---|
| Squat | 9 | 3.7 / 8.4 / 4.8 | 3.9 / 8.6 / 5.3 | 5.0° | 4 ms |
| Sit-to-stand | 9 | 3.4 / 7.9 / 5.9 | 3.8 / 8.3 / 6.7 | 5.4° | 2 ms |
| Drop jump | 27 | 4.4 / 8.0 / 5.8 | 4.1 / 7.5 / 5.4 | 6.3° | 3 ms |
| Gait | 27 | 4.8 / 7.1 / 4.7 | 4.8 / 7.1 / 4.4 | 8.6° | 4 ms |

For comparison, a single camera in 2D with the same engine (RTMPose balanced), in the same order of
movements: knee 5.8 / 8.1 / 4.7 / 6.6°, hip 7.9 / 11.0 / 7.8 / 5.1°, ankle 18.5 / 17.8 / 12.9 / 10.6°.

- **Calibrating with the body matches checkerboard calibration** in all four movements: the difference
  in the angles is 1° or less.
- **At the knee, two cameras measure better than one:** 3.8–4.8° against 4.7–8.1°.
- **With two cameras the ankle becomes measurable** (4–7° against 11–18° with one).
- **The hip stays at 7–9°**, with no clear improvement over one camera (better in sit-to-stand, worse in
  gait).
- The camera orientation is recovered with 5 to 9° of mean error (maximum 17°) and the distance between
  cameras within 1 to 4 %, and the joint angles still come out the same.
- The offset between the two videos is recovered with a maximum error of 18 ms (one frame lasts 17 ms).
- The tool's quality indicator said "good" in 69 of the 72 recordings and "acceptable" in 3, all of them
  gait.
- The 3D points end up about 6 cm from the lab markers after aligning position and orientation (this
  includes the difference between a marker on the skin and the joint centre).
- Letting the program fit the focal length makes things worse (tested earlier: 14° of knee error). It is
  kept fixed.

**What failed before and why.** The first gait test gave 16.5° of knee error and 51° of camera
orientation error. There were two causes, and neither was pose detection:

1. Synchronisation between videos used the vertical motion of the body, which barely exists in gait. In 6
   of 27 recordings it was off by about half a second. Now it looks for the offset at which the two
   skeletons fit a single camera pair, and refines it with the mismatch between cameras.
2. The test bench aligned the video with the lab badly (see the correction above).

Left/right leg confusions, the initial suspicion, hardly appear: 1 frame across the 27 gait recordings.

**Limits.** Two cameras, a single pair, far apart and on tripods; all the phones are the same model, at
60 frames per second; direct geometric angles, not yet through OpenSim; how much manual corrections help
is not measured.

### With three and with five cameras

`python -m validation.multiview --batch squats walking --cams Cam0 Cam2 Cam4`
`python -m validation.multiview --batch squats walking --cams Cam0 Cam1 Cam2 Cam3 Cam4`

Squat (9 recordings) and gait (27). Mean error in degrees (knee / hip / ankle), with the ceiling
(checkerboard calibration) and with uncalibrated OpenKick:

| Cameras | Squat, ceiling | Squat, OpenKick | Gait, ceiling | Gait, OpenKick |
|---|---|---|---|---|
| 2 (Cam0, Cam4) | 3.7 / 8.4 / 4.8 | 3.9 / 8.6 / 5.3 | 4.8 / 7.1 / 4.7 | 4.8 / 7.1 / 4.4 |
| 3 (Cam0, Cam2, Cam4) | 4.5 / 6.5 / 3.8 | 4.6 / 6.5 / 3.9 | 6.3 / 7.9 / 5.3 | 6.3 / 7.9 / 4.9 |
| 5 (all) | 4.4 / 6.5 / 4.0 | 4.3 / 6.3 / 3.8 | 6.3 / 7.7 / 5.1 | 6.1 / 7.4 / 4.5 |

- **Uncalibrated OpenKick matches the ceiling** with any number of cameras (0.5° or less difference). The
  more cameras, the better they are placed: 5.0° of orientation error with two cameras in squat, 2.7°
  with five.
- **More cameras is not better for everything.** In squat the hip (from 8.6 to 6.3°) and the ankle (from
  5.3 to 3.8°) improve, but the knee gets slightly worse (from 3.9 to 4.3–4.6°). In gait the knee gets
  worse by more (from 4.8 to 6.1–6.3°).
- That degradation **appears equally with checkerboard calibration**, so it does not come from
  calibrating with the body. The frontal cameras add it, since they see knee flexion poorly. It is left
  as is: it is the real error of the method with these cameras.
- **Synchronisation in gait:** in 12 of the 27 recordings, the offset OpenKick computes for some camera
  differs by more than 50 ms from the reference one (in 3, by more than half a second). With two cameras
  this did not happen (maximum 18 ms). It is not established which of the two is wrong: gait repeats, and
  the reference is also obtained by looking for the best fit. The angles come out the same as the ceiling
  in those recordings.
- **Dropping the camera that disagrees:** with three or more cameras, leaving out the one that deviates
  by a few centimetres made the angles worse (knee 5.3° against 4.7° in a test with 7 recordings). That is
  why a camera is only dropped if it places the point more than 12 cm from where the others put it. With
  that rule, in 39 recordings the angles stay the same as always using every camera (0.07° or less
  difference) and the points flagged as doubtful are halved.

### Doubtful points

The tool tells the user which points to check. Three signals, with a 6 cm threshold:

- the two cameras place the point in different spots;
- every bone that meets at the point departs from its usual length;
- the point is far from where the previous and the next frame place it.

The last two are needed because the first cannot see a point displaced in the direction the other camera
is looking: in a test with a synthetic person, a knee moved by 50 px (about 25 cm in the image) moved the
3D point by 19 cm while the cameras only disagreed by 2 cm. With the three signals that point, and only
that point, is flagged (`tests/test_multiview.py`).

### Changes to the reconstruction (25 October 2026)

The cycling sessions recorded with phones (one camera at the side, one behind and one in front, see
below) brought to light five failures this dataset did not show. What was changed:

1. **Camera motion:** it was measured by chaining each frame to the previous one, and the error
   accumulated: up to 17.5 px on a phone that was not moving. Now each frame is compared directly with the
   first one.
2. **Synchronisation by sound:** pedalling repeats, and the motion can fit one pedal stroke off (it
   happened: 0.46 s against a true 0.42 s, one crank revolution). If both videos have sound and the match
   is clear, the offset comes from the sound (GCC-PHAT correlation); otherwise from the motion, as before.
   The videos in this dataset have no sound, so it is not tested here.
3. **Calibrating a pair with several hypotheses:** with one camera at the side and another behind, the fit
   could give angles between them from 33° to 76° with the same fit quality. Now several starting
   positions are tried and the one that gives the most constant bone lengths is kept.
4. **Third camera and beyond:** they are placed from the 3D points of the first pair (resection, PnP)
   instead of by loose pairs, and the starting pair is the one with the most constant bones. The final
   joint adjustment penalises the thigh and shank changing length. Without this, the three cycling cameras
   gave a knee of 172° (actually 140°).
5. **Scale from tape measurements:** if the person's thigh or shank is measured, the scale comes from
   those measurements instead of the height.

Result against the lab, same test bench (knee / hip / ankle, uncalibrated OpenKick):

| Cameras | Squat, before | Squat, now | Gait, before | Gait, now |
|---|---|---|---|---|
| 2 (Cam0, Cam4) | 3.9 / 8.6 / 5.3 | 3.7 / 8.3 / 4.7 | 4.8 / 7.1 / 4.4 | 4.8 / 7.1 / 4.4 |
| 3 (Cam0, Cam2, Cam4) | 4.6 / 6.5 / 3.9 | 4.7 / 6.5 / 3.9 | 6.3 / 7.9 / 4.9 | 6.2 / 7.9 / 4.7 |

- Nothing gets worse by more than 0.1°; the two-camera squat improves in all three joints.
- The camera orientation error with two cameras in squat goes up from 5.0° to 6.0° (maximum 17°) without
  the angles noticing; in gait it stays the same (8.6°).
- **Gait synchronisation with three cameras is unchanged:** in 12 of 27 recordings the offset differs by
  more than 50 ms from the reference, and in 3 by more than half a second (half a cycle or a whole one).
  It is the same problem as before: a repeating movement fools motion-based synchronisation. A synthetic
  person walking with identical cycles reproduces it
  (`tests/test_multiview_gait.py::test_gait_through_the_whole_3d_reconstruction`: one camera ends up three
  strides off). Sound avoids it when available.

**In the afternoon: one-frame jumps and smoothing.** In the 93 cm cycling session, with its three cameras,
the 3D skeleton showed one-frame jumps: a leg that bent, a foot that deformed, the head moving without
moving in the videos. The 2D detections in those frames barely moved (0.4–2 px); the cause was the rule
that leaves out the camera that disagrees. The rear and the front cameras easily agree with each other
(both see poorly forwards and backwards), so the one that lost out, in single frames, was the side camera.
Changes:

1. A camera that disagrees by more than 12 cm is left out of the 3D point only when, with it, the point
   leaves the trajectory given by the neighbouring frames (their median over 0.15 s) by more than 4 cm and,
   without it, returns to that trajectory. Otherwise all of them are averaged. The disagreement still
   counts towards doubtful points. (Deciding over 0.3 s stretches was also tried: it gave more jumps.)
2. **3D smoothing:** it removes one- or two-frame jumps (more than 3 cm, or 4 times that point's own
   tremor, away from its trajectory) and applies a zero-lag 4th-order Butterworth filter, 8 Hz by default
   (12 Hz and without removing jumps for kick and jump). Doubtful points are searched before smoothing.
3. A speed-up from the same morning was undone: trying the calibration's starting positions with a third
   of the frames. In one gait recording (subject7, walking3) it left the camera with 50° of error (28°
   before refinement), against 10° with all of them.

In the two cycling sessions: jumps of more than 6 cm, from 848 and 827 to 0; head tremor (above 2 Hz), from
3.5 and 3.0 cm to 0.6; doubtful points, from about 590 to 87 and 144. The pedal stroke angles change by less
than 1°.

Against the lab, with all of this (knee / hip / ankle):

| Cameras | Squat, morning | Squat, final | Gait, morning | Gait, final |
|---|---|---|---|---|
| 2 (Cam0, Cam4) | 3.7 / 8.3 / 4.7 | 3.7 / 8.3 / 4.6 | 4.8 / 7.1 / 4.4 | 4.8 / 7.1 / 4.2 |
| 3 (Cam0, Cam2, Cam4) | 4.7 / 6.5 / 3.9 | 4.6 / 6.4 / 3.9 | 6.2 / 7.9 / 4.7 | 6.3 / 7.9 / 4.7 |

The same or slightly better, except the knee in gait with three cameras (+0.1°). Gait synchronisation with
three cameras is as it was (12 of 27 recordings more than 50 ms off).

**Fixed-length thighs and shanks (optional).** Each bone is brought to its typical length in the video by
moving its ends along it, mostly the one the cameras see worse. Two cameras, 9 squats and 9 gait trials
(one per person), the same sessions with and without fixing:

| | Free | Fixed | Fixed, hip with double weight |
|---|---|---|---|
| Squat | 3.68 / 8.30 / 4.63 | 3.62 / 8.38 / 4.38 | 3.64 / 8.40 / 4.47 |
| Gait | 4.70 / 6.89 / 4.21 | 5.08 / 6.90 / 4.15 | 5.03 / 6.92 / 4.17 |

It does not bring the angles closer to the lab (the knee in gait gets 0.3–0.4° worse), and in the 93 cm
cycling session the pelvic roll went from 11.5° to 23–26° and the knee at the bottom changed by 3–4°: the
hip, moved along the thigh, goes up and down with every pedal stroke. It remains an option, off by default
and with that warning.

**Rigid feet (optional).** Heel, toe and little toe with a single shape (the typical one in the video) that
rotates about the ankle, which is not touched. Two cameras, 9 squats and 27 gait trials:

| | Free | Rigid |
|---|---|---|
| Squat | 3.68 / 8.30 / 4.63 | 3.68 / 8.30 / 4.74 |
| Gait | 4.75 / 7.12 / 4.23 | 4.75 / 7.12 / 4.15 |

Neutral: knee and hip identical, ankle ±0.1°, with participants in regular trainers. In the cycling
sessions the shape of the foot stops varying (from 4–25 % to 0.1–0.3 %), the change of the ankle from 93 to
95 cm holds (+11.6° free, +11.2° rigid) and that of the toe goes from +9.1° to +14.0°, closer to the 2D
side camera (+14.3°). The doubtful points of the 93 session drop from 87 to 56. It remains an option for
stiff soles.

### Skeleton: detection or OpenSim model, frame by frame (25 October 2026)

(Options 2 and 3 were removed from the program on 26 October: the skeleton is again the detection only.
The measurement remains, and the scripts need the version with `openkick/multiview/model_fit.py`.)

```
PYTHONPATH=. python validation/skeleton_frames_collect.py <folder>
PYTHONPATH=. python validation/skeleton_frames_analyse.py <folder> validation/results/esqueleto_fotograma
```

The three *Skeleton* options (1 raw detection; 2 OpenSim model with the proportions of the 3D; 3 model with
the tape measurements, here the lab's as a perfect tape), two cameras (Cam0, Cam4), 9 squats and 9 gait
trials. For each frame, the angles and 3D points of the three are saved, together with the lab markers, the
lab's OpenSim angles and those of OpenCap with two cameras (HRNet). Figures and full report in
`validation/results/esqueleto_fotograma/`.

**What it is compared against.** Until now, against geometric angles from the lab markers. But `r_knee` and
`r_ankle` are skin markers, on the outside of the knee and the ankle (6–8 cm and 3–5 cm outside the joint
centre), not centres. The gold standard is the lab's OpenSim angles (`OpenSimData/Mocap/IK`).

Mean error in degrees (knee / hip / ankle), against the markers, as in the tables above:

| | 1 Detection | 2 Model, proportions | 3 Model, measurements |
|---|---|---|---|
| Squat | 3.7 / 8.3 / 4.6 | 5.0 / 5.5 / 7.1 | 6.6 / 6.8 / 13.8 |
| Gait | 4.7 / 6.9 / 4.2 | 8.2 / 4.8 / 8.0 | 11.4 / 5.3 / 14.8 |

The knee OpenKick shows, against the lab's OpenSim knee:

| | Lab markers | 1 Detection | 2 Model | 3 Measurements | OpenCap (its OpenSim) |
|---|---|---|---|---|---|
| Squat | 4.5 | 4.7 | **2.5** | 4.9 | 3.1 |
| Gait | 5.5 | **4.0** | 5.0 | 7.7 | 3.4 |

The OpenSim angles of our model against the lab's (knee / hip flexion / ankle): squat 2.7 / 9.4 / 3.3
(OpenCap 3.1 / 5.7 / 3.1); gait 4.3 / 3.5 / 5.3 (OpenCap 3.4 / 7.4 / 4.1). This matches the 1 October
measurement.

- **They are not isolated spikes.** With each recording's offset removed, the frames above 10° are 0 to 5 %,
  in stretches of up to 0.4 s. The error is almost all a **fixed offset** or a **gain** (the range of motion
  stretched or compressed). Without the offset, the knee of all three options is at 2–5°.
- **The lab's marker geometry compresses the knee** (gain 0.89 against its own OpenSim: +6.7° near a
  straight leg, −3.3° at the bottom of the squat). **The detection has the same defect** (gain 0.90: +4.1°
  and −5.3°). That is why comparing against markers favoured the detection.
- **Squat: the model is the best knee.** It stays within ±1.4° of the lab from 0 to 120° of flexion.
- **Gait: the model shows less knee flexion**, the more so the more it bends (−1.8° with the leg almost
  straight, −7.2° in swing, between 45 and 75°). The detection does better (+2.6° to −3.0°).
- **Hip (trunk–thigh):** the detection shows 6.6–8.1° less, consistently (spread between people
  1.7–2.4°). In squat, its hip point sits 6 cm outside and 2 cm above the lab's hip centre. With a fixed
  correction learnt from the other participants, the error would drop from 8.3 to 4.8° in squat and from
  6.9 to 3.6° in gait. The model places the hip 2 cm from the centre and the offset is halved (−4°).
- **Model ankle: fixed offset of +7–8°**, 1.7–3.6° without the offset. The model's heel sits 2–3 cm further
  forward and up to 3 cm higher than the lab's heel marker, and that tilts the foot line. The model's
  OpenSim ankle is close to OpenCap's (3.3 and 5.3° against 3.1 and 4.1°).
- **The detector places the knee 1.2–1.8 cm higher** than the lab. That is why, in 3D, the thigh comes out
  shorter (−0.8 to −1.0 cm) and the shank longer (+1.0 to +1.7 cm). The same happens in the 93 cm cycling
  session against the tape (thigh −1.1 to −3.0 cm; shank +1.0 to +2.1 cm).
- **A perfect tape (option 3) makes the angles worse:** knee −4.6° in squat and −7.5° in gait. Those
  lengths do not fit where the detector places the points, and the model compensates by bending the knee
  less. For the angle, what matters is that the model's proportions fit the detected points, not that they
  are the true ones. Tape measurements are useful for scaling and for checking (the *Tape − detection*
  column), not for forcing the model.
- **The model's pelvic tilt in squat is not reliable:** correlation 0.75 with the lab and a different
  offset for each person (spread 13°), against 0.98 for OpenCap. It goes with the +8° of hip flexion in
  OpenSim (see 1 October).
- **The default stays the raw detection:** the model wins for the knee in squat and for the hip, and loses
  for the knee in gait and for the ankle. None of this is measured in cycling.

### Cycling: saddle at 93 and at 95 cm (25 October 2026)

There is no lab to compare with: this is a consistency test. A cyclist recorded in two sessions, with the
saddle at 93 cm and at 95 cm, with three phones: at the side (832×464, 60 fps), behind (480×854, 60 fps)
and in front (854×480, 30 fps), videos forwarded through Telegram (compressed). Height 1.685 m. No manual
corrections. Both sessions come out with "acceptable" quality; offsets (from the sound) 0 / 0.416 /
2.249 s in the 93 session and 0 / −0.047 / 1.849 s in the 95 one.

Pedal strokes measured: 26 left and 27 right with the saddle at 93; 32 and 31 at 95. Median of each
measurement (before the afternoon fix for jumps, which changes them by less than 1°):

| Leg | Measurement | 93 | 95 | Change |
|---|---|---|---|---|
| Right | Knee, bottom | 140.3° | 138.6° | −1.6 |
| Right | Ankle, bottom | 106.9° | 118.6° | +11.6 ● |
| Right | Toe down, lowest point | 0.6° | 11.1° | +10.5 ● |
| Left | Knee, bottom | 138.7° | 141.8° | +3.0 ● |
| Left | Hip, top | 77.4° | 83.1° | +5.7 ● |
| Left | Knee lateral sway | 8.9 cm | 11.3 cm | +2.4 ● |
| Left | Knee towards the bike | −7.7 cm | −10.3 cm | −2.6 ● |

(● = the change exceeds the variation from one pedal stroke to the next.)

- **The 3D and the 2D side camera tell the same story:** with the saddle 2 cm higher, the right knee
  hardly changes (2D: 138.6° → 139.0°; 3D: 140.3° → 138.6°) and the ankle absorbs it by extending (2D:
  +19.7° ankle, +14.3° foot; 3D: +11.6° and +10.5°). The left leg, which the side camera does not see,
  extends the knee (+3.0°), opens the hip at the top (+5.7°) and carries the knee further out (−2.6 cm).
- The absolute values of what only the 3D sees (knee sway, pelvic roll, 11.5–11.8° in both sessions) are
  not checked against a lab: what counts here is the difference between sessions recorded the same way.
- **The right thigh is doubtful:** its length varies by ±15–18 % (the hip slides along the thigh), even
  though the knee angle is stable. The comparison flags it as doubtful.
- With the rear and the front cameras about 140° apart, the hip on the far side is triangulated poorly. The
  front camera records at 30 fps.

## OpenSim angles against the lab (1 October 2026)

`python -m validation.opensim_ik --types squats STS DJ walking`

This is the full chain: two videos → OpenKick 3D points (no calibration, no manual corrections) → OpenSim
scaling and inverse kinematics with Pose2Sim (simple model, no marker augmentation) → model angles. They are
compared with the angles OpenSim gives from the lab markers (`OpenSimData/Mocap/IK`). 72 recordings, 9
participants, cameras Cam0 and Cam4. Mean error in degrees; bias in brackets.

| Movement | Knee | Hip, flexion | Hip, adduction | Hip, rotation | Ankle |
|---|---|---|---|---|---|
| Squat (9) | 2.9 (+0.9) | 9.7 (+8.7) | 3.6 (+3.0) | 7.2 (+5.1) | 3.4 (+0.9) |
| Sit-to-stand (9) | 2.9 (+1.4) | 16.2 (+15.8) | 3.5 (+2.7) | 5.9 (+4.3) | 4.1 (−0.6) |
| Drop jump (27) | 3.4 (+1.9) | 12.8 (+12.5) | 3.0 (+1.9) | 5.9 (+4.2) | 8.6 (+8.3) |
| Gait (27) | 4.6 (−2.7) | 3.7 (+0.7) | 6.3 (−3.3) | 11.1 (+0.4) | 5.8 (+4.1) |
| **All (72)** | **3.7** | **9.4** | **4.4** | **8.0** | **6.3** |

As a yardstick, the result OpenCap publishes for these same recordings with two cameras (HRNet,
checkerboard-calibrated cameras, with its marker augmentation), scored with the same code: knee 3.9; hip
flexion 6.6; adduction 2.3; rotation 3.9; ankle 5.8. It is not the same camera pair, so the comparison is
indicative.

- **Knee and ankle are on a par with OpenCap** (3.7 against 3.9 and 6.3 against 5.8).
- **Hip flexion comes out too high** in squat, sit-to-stand and jump: 9 to 16° of bias, with a range of
  motion about 23° larger than the lab's. It does not happen in gait (0.7°).
- **The cause is the pelvis, not the legs.** The model flexes the lumbar spine 15 to 24° less than the lab
  and loads that flexion onto the hip. If the full trunk–thigh angle (hip plus lumbar) is measured, the
  error drops to 6.5 / 7.8 / 9.1° in squat, sit-to-stand and jump. With only two hip points, pelvic tilt
  cannot be seen; OpenCap infers it with a network that adds anatomical markers.
- **Hip adduction and rotation** come out worse than OpenCap's (4.4 and 8.0 against 2.3 and 3.9), for the
  same reason.
- The model markers end up 3.5–4.1 cm from the 3D points after inverse kinematics.

### With Pose2Sim's marker network

`python -m validation.opensim_ik --types squats STS DJ walking --augment`

Pose2Sim ships the same network OpenCap uses (it adds 43 anatomical markers from the detected points).
Turned on as it comes, with the Pose2Sim model, **it does not fix the hip and makes the whole worse**:

| | Knee | Hip, flexion | Hip, adduction | Hip, rotation | Ankle |
|---|---|---|---|---|---|
| Detected points only (default) | 3.7 | 9.4 | 4.4 | 8.0 | 6.3 |
| With the marker network | 5.0 | 15.0 | 3.4 | 7.2 | 7.0 |
| OpenCap, two cameras | 3.9 | 6.6 | 2.3 | 3.9 | 5.8 |

- Hip flexion now has an almost constant excess of 11 to 18° in all four movements, gait included. In
  exchange its shape improves: with that excess removed, the spread drops from 8–10° to 5–6°, and the range
  of motion is 13–15° too large instead of 23–25°.
- In gait, adduction (correlation with the lab from 0.16 to 0.87) and hip rotation improve a lot.
- Knee and ankle get worse: the knee range falls 6–9° short and the ankle shifts by 8–10° in squat and
  sit-to-stand.

A constant excess points to how the pelvis is defined in the Pose2Sim model and how it is scaled, not to the
3D points. OpenCap uses its own model and scales it with a standing recording of the person; that is not
tested here. For now the export stays without the network.

## Additional studies (2 October 2026)

`python -m validation.study baseline fps30 holdout pairs` and then `python -m validation.stats`. All the
tables, with their intervals, are in `validation/results/study_summary.txt`.

### Tuning on some participants and measuring on others

Several settings of the method were decided by looking at these same recordings, which can give optimistic
results. To check this, the participants were split: settings are chosen only with participants 2, 3, 4 and
5, and the result is measured on participants 7, 8, 9, 10 and 11. The split was made after developing the
method, not before; the right thing would have been to make it from the start.

3D with two cameras (Cam0 and Cam4), all four movements. Mean error in degrees:

| | Knee | Hip | Ankle |
|---|---|---|---|
| Tuning participants, default settings | 4.5 | 7.8 | 5.5 |
| Test participants, default settings | 4.2 | 7.5 | 4.8 |
| Test participants, settings chosen on the tuning ones | 4.3 | 7.1 | 4.6 |

- On the participants who took part in no tuning, the error is the same or lower. There is no sign that the
  numbers in this document are inflated by tuning on the same data.
- Of the three settings tried, two change nothing: the minimum confidence of a point (0.3, 0.5 or 0.7) and
  the threshold for accepting a match (2, 3 or 5 px) move the error by 0.1° or less.
- **The assumed lens does matter.** Assuming 22, 26 or 30 mm equivalent gives 5.5 / 5.9 / 6.5° of mean
  error on the tuning participants. 22 mm does slightly better here, but on the test participants the
  improvement is not clear (better in squat and sit-to-stand, worse for the knee in gait). 26 mm is kept,
  which is usual for a phone. If the real lens is known, it is better to use it.

### Statistics: intervals, agreement and reliability

3D with two cameras, 72 recordings, 9 participants. The 95 % interval is obtained by resampling
participants, not recordings. The limits of agreement (Bland–Altman) are over all frames.

| Joint | Mean error [95 % CI] | Bias | Limits of agreement | Correlation |
|---|---|---|---|---|
| Knee | 4.3° [3.5 to 5.3] | −3.1° | −10.6 to +4.4° | 0.99 |
| Hip (trunk–thigh) | 7.6° [6.6 to 8.6] | −8.0° | −18.2 to +2.2° | 0.91 |
| Ankle | 5.1° [4.5 to 5.8] | +4.4° | −6.0 to +14.8° | 0.92 |
| Shoulder (arm–trunk) | 5.3° [4.7 to 5.8] | +0.3° | −15.6 to +16.1° | 0.86 |
| Elbow | 14.4° [12.4 to 16.6] | −14.0° | −30.9 to +2.9° | 0.80 |

- **What is new here is the shoulder and the elbow.** The shoulder is on a par with the knee and the ankle.
  **The elbow is not valid:** it measures 14° less flexion almost constantly, which suggests that the point
  the network detects does not coincide with the elbow centre defined by the markers.
- **Reliability within one movement.** To know whether the method tells one person from another, the range
  of motion and the maximum value were compared recording by recording with the lab's (intraclass
  correlation, ICC; 1 is perfect agreement):

| Movement | Knee, range | Knee, maximum | Hip, range | Hip, maximum | Ankle, range |
|---|---|---|---|---|---|
| Squat | 0.84 | 0.95 | 0.40 | 0.97 | 0.84 |
| Sit-to-stand | 0.70 | 0.89 | 0.30 | 0.93 | 0.90 |
| Drop jump | 0.82 | 0.99 | 0.87 | 0.94 | 0.68 |
| Gait | 0.32 | 0.45 | 0.24 | 0.13 | 0.41 |

- In squat, sit-to-stand and jump, the **knee and hip maximum** is reliable (0.89 to 0.99). The **hip range**
  is not, in squat and sit-to-stand (0.30 to 0.40): it comes out 13 to 17° larger than the lab's.
- **In gait reliability is low for everything** (0.13 to 0.45). People are very alike when walking, and an
  error of 4 to 7° is the same size as the differences between them. This method cannot compare people by
  their gait.
- If all four movements are pooled, the ICC rises to 0.95–0.99, but that number is misleading: it only says
  that a squat can be told apart from a step.
- The same statistics for 2D and for the OpenSim angles are in `study_summary.txt`. In 2D with one camera,
  the knee range has an ICC of 0.43 to 0.81 in squat, sit-to-stand and jump, and of 0.09 to 0.22 in gait.

### At 30 frames per second

The same 72 recordings using one frame out of two. Mean error (knee / hip / ankle): 4.3 / 7.6 / 5.1° at
60 fps and 4.3 / 7.6 / 5.1° at 30 fps. Synchronisation does not get worse either (3 ms against 4 ms median
error).

Two limits of this test: these movements are not fast, and dropping frames from a 60 fps video is not the
same as recording at 30, because each frame keeps the sharpness of 60. It says nothing about a serve or a
kick at 30 fps.

### Other camera pairs

Squat (9 recordings) and gait (27), with six camera pairs. Mean error in degrees (knee / hip / ankle), with
checkerboard calibration and with uncalibrated OpenKick:

| Pair | Angle between cameras | Squat, checkerboard | Squat, OpenKick | Gait, checkerboard | Gait, OpenKick |
|---|---|---|---|---|---|
| Cam0 + Cam1 | 30° | 4.9 / 6.1 / 6.8 | 5.1 / 6.5 / 9.5 | 6.2 / 6.3 / 6.4 | 6.3 / 6.3 / 6.2 |
| Cam2 + Cam4 | 65° | 5.8 / 7.0 / 3.9 | 5.9 / 7.0 / 4.9 | 7.3 / 8.1 / 7.2 | 9.8 / 7.8 / 7.4 |
| Cam0 + Cam2 | 65° | 5.2 / 6.5 / 3.6 | 5.4 / 6.7 / 4.7 | 6.7 / 6.7 / 6.5 | 6.7 / 6.8 / 5.8 |
| Cam1 + Cam3 | 80° | 4.7 / 6.8 / 4.1 | 4.9 / 6.8 / 4.0 | 6.4 / 6.6 / 6.1 | 6.4 / 6.3 / 7.1 |
| Cam0 + Cam3 | 105° | 4.6 / 7.3 / 4.4 | 4.8 / 7.4 / 5.4 | 5.8 / 7.3 / 5.2 | 5.8 / 7.1 / 4.9 |
| Cam0 + Cam4 | 130° | 3.7 / 8.4 / 4.8 | 3.9 / 8.6 / 5.3 | 4.8 / 7.1 / 4.7 | 4.8 / 7.1 / 4.4 |

- **It works with any pair, including cameras only 30° apart.** In 11 of the 12 cases OpenKick stays within
  1° of checkerboard calibration for knee and hip.
- **Two worse cases:** the knee in gait with Cam2 + Cam4 (9.8 against 7.3°) and the ankle in squat with
  Cam0 + Cam1 (9.5 against 6.8°).
- **The pair used in the rest of this document (Cam0 + Cam4) is the one that measures the knee best.** With
  the other pairs, the knee comes out 1 to 2.5° worse even with checkerboard calibration. The knee numbers
  in the main tables are therefore the favourable case.
- Synchronisation is less precise when one of the cameras is the frontal one (Cam2): 7 to 37 ms median
  error, against 2 to 6 ms for the other pairs.
- The quality indicator said "good" in all 54 squats and in 131 of the 162 gait trials. With the cameras
  30° apart, only in 13 of 27 gait trials.

### Other public datasets (search of 2 October 2026)

We looked for a dataset with sports movements, several cameras and a marker-based lab reference. None was
found that has all three. Candidates, with what could be checked on their pages:

| Dataset | Movements | Cameras | Reference | Access |
|---|---|---|---|---|
| BioCV (University of Bath, 2024) | Gait, running, countermovement jump, two-footed hops; 15 people | 9, at 1920 × 1080 and 200 fps, calibrated | Markers (.c3d files) and force plates | CC BY 4.0 licence; registration required |
| SportsPose (DTU, 2023) | 5 sports activities; 24 people | 7, at 90 fps | Markerless: triangulated detections; 34.5 mm error against markers | Academic licence, on request |
| AthleticsPose (2025) | Start, sprint, running, race walking, hurdles, shot put, discus, javelin; 23 or 24 athletes, depending on the source | 8, HD, at 30 or 60 fps, outdoors | Markerless (Theia3D) | Public on GitHub |
| AthletePose3D (2025) | 12 sports actions | 4 to 12, at 60 or 120 fps | Markerless | Not checked |

- **BioCV is the natural next step:** it is the only one with a marker reference, and it adds running and
  jumps. It has no kicks or throws.
- The three with sports movements use another markerless system as reference. They are useful to see
  whether OpenKick agrees with them, not to say how wrong it is.
- For SportsPose, its page did not confirm which the five activities are or whether the videos of every
  camera are provided.

## From 3D to OpenSim

`openkick/multiview/export.py` writes the 3D points as markers (.trc) in OpenSim's axes and runs Pose2Sim's
scaling and inverse kinematics. `validation/opensim_render.py` draws the resulting model, with its bones,
without needing the OpenSim application. `openkick/multiview/detect.py` compensates for the motion of a
hand-held camera using the background.

## Clinical angles, gait events and speeds (1 November 2026)

A biomechanics review of the program (ten points) found that the angles were reported as the angle
inside the joint (180° = straight), which folds over at 180°: a hip in extension read like one in
flexion, and its speed showed false peaks where the curve folded. It also asked for the gait events, the
speeds and the stride length to be checked. Each point was answered by measuring several ways of doing
it on the OpenCap data and keeping the one with the least error. The scripts are not in the repository;
they reuse `validation/opencap.py` and `validation/multiview.py`.

**Reference.** Here the reference is the lab's own OpenSim inverse kinematics (`hip_flexion`,
`knee_angle`, `ankle_angle`, `arm_flex`), which is the clinical convention. The tables in *Results (1
October 2026)* compared against angles built from the lab markers the way OpenKick built its own, so the
numbers are not comparable one to one.

**2D, one side camera (MediaPipe as the app runs it).** 9 participants, Cam0 and Cam4, squat (18 videos),
sit-to-stand (18), drop jump (54) and gait (54; only the trials whose time offset is known from 3D). Mean
absolute error in degrees, near leg / far leg:

| Joint, way of measuring | Gait | Squat | Sit-to-stand | Drop jump |
|---|---|---|---|---|
| Hip, inside the joint (before) | 16.5 / 17.8 | 7.3 / 7.5 | 6.1 / 9.8 | 7.0 / 7.8 |
| Hip, thigh against the trunk's axis | 9.5 / 12.3 | 7.0 / 7.8 | 6.7 / 8.9 | 7.0 / 7.6 |
| Hip, thigh against the vertical | **5.5 / 6.0** | 24.8 / 20.9 | 15.6 / 14.4 | 18.1 / 17.7 |
| Hip, vertical tipped by 0.75 × trunk lean | 9.1 / 10.9 | **8.4 / 6.0** | **6.0 / 6.8** | **5.4 / 5.4** |
| Knee, inside the joint (180 − angle) | 8.2 / 7.3 | 9.2 / 4.3 | 9.7 / 6.4 | 7.0 / 5.5 |
| Knee, with a sign | 8.1 / 8.6 | 9.4 / 4.9 | 9.7 / 6.5 | 7.0 / 5.5 |
| Ankle, foot from ankle to toes (before) | 29.8 / 27.1 | 43.1 / 31.8 | 42.2 / 35.4 | 32.0 / 27.5 |
| Ankle, foot from heel to toes | **10.9 / 9.4** | **20.4 / 11.3** | **19.1 / 14.5** | **13.0 / 11.2** |

- **Hip.** In gait the folded angle was unusable (correlation with the lab 0.45 near, 0.29 far); against
  the vertical it is 0.90 and 0.96. Where the trunk bends (squat, sit-to-stand, jump) the vertical alone
  misses the pelvis tilting with it, and taking three quarters of the trunk's lean as that tilt is best:
  over both legs 7.2° against 7.4° before in the squat, 6.4° against 8.0° in sit-to-stand, 5.4° against
  7.4° in the jump. In the frames where the lab hip is in extension the error drops from 10–12° to 5°
  (squat) and from 24–28° to 5–6° (gait). Other values of the share (0.5, 1.0) and a slowly filtered lean
  (0.5, 1 and 2 Hz) were also tried and were no better.
- **Knee.** With or without a sign the error is the same within 0.6°, except the far leg in gait (1.3°
  more with a sign: on a healthy knee near straight the noise now shows on both sides of 0°). The sign is
  kept so that a hyperextended knee shows as negative.
- **Ankle.** The foot from heel to toes removes most of the error. What remains is mostly a constant
  offset (−4 to −20°) that changes with the movement, so it is not corrected.
- **Shoulder** (MediaPipe heavy, which keeps the arms; participants 2 to 9, 103 videos). Near side /
  far side: inside the joint 24.3 / 19.9 in gait, 14.8 / 9.9 squat, 19.9 / 10.7 sit-to-stand, 31.8 /
  23.7 jump. Against the vertical in gait: **5.1 / 9.7**. Against the trunk's axis in the others: **10.6 /
  7.5, 14.6 / 7.9, 23.3 / 18.6**. The shoulder in the jump stays unreliable (the arms swing fast and above
  the head).

**3D, two cameras (Cam0 and Cam4).** 9 participants, one trial of each movement, both legs (n = 18 per
cell), with the vertical the app itself finds:

| Joint, way of measuring | Gait | Squat | Sit-to-stand | Drop jump |
|---|---|---|---|---|
| Hip, inside the joint (before) | 12.8 | 7.6 | 8.9 | 7.6 |
| Hip, thigh against the vertical, in the pelvis's side plane | **4.3** | 16.9 | 12.0 | 9.2 |
| Hip, vertical tipped 0.75 of the way to the trunk | 4.2 | **6.0** | **7.3** | **5.3** |
| Knee, flexion | 4.0 | 4.6 | 5.0 | 4.3 |
| Ankle, foot from ankle to toes (before) | 17.4 | 25.5 | 26.9 | 18.8 |
| Ankle, foot from heel to toes | **4.3** | **6.1** | **7.5** | **7.3** |

The hip's adduction was also measured in the same pelvis estimate: 4.9° in gait but with a correlation
of only 0.40, and a 10° offset in the squat; it is not reported. Rotation cannot be measured from these
points.

**What the app does now.** Knee, hip, ankle, shoulder and elbow read clinically, in 2D and 3D (see
`openkick/core/joints.py`): the Gait template measures the hip and the shoulder against the vertical,
every other template against the vertical tipped by three quarters of the trunk's lean (the shoulder
against the trunk's axis), the ankle with the foot from heel to toes. Cycling keeps the angle inside
the joint, the language of bike fitting: when pedalling the hip and the knee never reach 180°, so they do
not fold. A template can choose (`convention`, `hip_reference`, see [TEMPLATES.md](TEMPLATES.md)). The
gait tables gained *Hip, most extended*. Projects already saved show the new angles when opened.

**Gait events against the force plates.** The OpenCap gait trials have force plates (2000 Hz): an
initial contact is the vertical force rising through 20 N, a toe off falling through it. 2D, 60 fps, 9
participants, both feet; mean (+ = late) and mean absolute error in ms:

| Method | Contact, near foot | Toe off, near foot | Contact, far foot | Toe off, far foot |
|---|---|---|---|---|
| Zeni et al. 2008 (the app's) | +13, 17 (45/56 found) | −31, 32 (47/53) | +5, 12 (43/56) | −25, 28 (47/53) |
| Foot velocity (O'Connor et al. 2007) | −135, 135 (2/56) | −26, 53 (7/53) | none found | −62, 62 (19/53) |
| Lowest heel / toes lifting | +110, 121 (18/56) | +8, 52 (14/53) | +126, 126 (28/56) | −13, 56 (17/53) |

Zeni's method is kept. Its toe off comes early. Moving it by the median learned on the other
participants (leaving each one out in turn) took the error from 29.7 to 21.2 ms. The shift was 21–26 ms
whoever was left out, so 2D toe offs are now moved 23 ms later (`TOE_OFF_DELAY_2D` in
`openkick/core/gait.py`). For the contact the same correction gained 1 ms (14.9 to 13.8), so it is left
alone. In 3D (10 contacts, 9 toe offs: the trials are short) the toe off was only 9 ms early
(contact +5 ms, abs 15; toe off −9 ms, abs 10), and the shift would make it worse. 3D is not shifted.

**Speeds.** A detector jump, a point that leaves for a frame or two and comes back, became a false
peak of speed (the review found an ankle at 5.86 m/s before the first stride). Before
differentiating, the app now drops the values further than 3 robust standard deviations from the median
of the 0.2 s around them. This is a Hampel filter, with a floor: the value must also be more than 5° away
for an angle, or 2 % of the leg for a point. Against the lab (2D, the same 144 videos):

| | Gait | Squat | Sit-to-stand | Drop jump |
|---|---|---|---|---|
| Knee angular speed, error in °/s, before → after | 114 → 89 | 25 → 25 | 29 → 28 | 52 → 53 |
| Knee, peak speed error in °/s | 222 → 189 | 51 → 51 | 50 → 50 | 59 → 60 |
| Hip angular speed, error in °/s | 48 → 42 | 21 → 21 | 27 → 26 | 44 → 44 |
| Ankle speed (lab marker), error in m/s | 0.70 → 0.56 | 0.01 → 0.01 | 0.02 → 0.02 | 0.33 → 0.31 |

The 2D ankle speed was scaled to the lab's metres by one factor per trial, median against median.
Without the floor, the filter took real speed out of the jumps (knee 52 → 57 °/s). Median filters of 3
and 5 frames, and a test for out-and-back jumps above 600 or 1000 °/s, gained less. The same filter
runs before the gait plugin's angles (gait knee angle error, once each trial's own offset is removed,
6.7 → 5.2°). With the Gait template, the
velocities plugin reports its peaks over *All strides* instead of the whole clip, so walking into and
out of the picture is left out.

**Stride length from one camera.** A calibration holds on one line, so someone walking nearer or further
changes size in the picture and their stride with it. The gait plugin now flags *Stride length steady*
when the longest and the shortest stride differ by more than 15 % of the median. Walking itself varies
2–4 % (coefficient of variation) from stride to stride. When the leg's size in the picture also changes
by more than 10 %, the message gives that as the cause. These two thresholds are not validated: the
OpenCap gait trials hold one or two strides.

### The back's shape from the person's outline (1 November 2026)

Prompted by a deadlift demo that scores how rounded the back is, a prototype read the back's edge from
MediaPipe's person mask (heavy model, `output_segmentation_masks`). The edge was taken along the line
from the mid-hips to the mid-shoulders, on the same 144 side videos. The edge sat on the back in every
frame checked, standing and at the bottom of the squat. Several measures of its shape were tested: the
bend between its lower and upper halves, the bulge from the straight line, and the tilt of its lower and
upper halves. Each was checked as a predictor of the lab's `lumbar_extension` and `pelvis_tilt`, with a
linear fit learned on the other participants. Error in degrees on the participant left out:

| | Squat | Sit-to-stand | Drop jump | Gait |
|---|---|---|---|---|
| Lumbar flexion, from the trunk's lean | 5.5 | 10.0 | 6.6 | 2.9 |
| Lumbar flexion, lean + best shape measure | 5.5 | 8.8 | 6.4 | 2.8 |
| Pelvis tilt, from the trunk's lean | 4.3 | 7.7 | 4.5 | 2.6 |
| Pelvis tilt, lean + best shape measure | 4.3 | 6.8 | 4.5 | 2.6 |
| Hip flexion, lean + shape (against lean alone) | 5.2 (5.0) | 4.4 (4.4) | 4.9 (4.9) | 5.6 (6.0) |

- **The shape adds almost nothing to what the trunk's lean tells.** It gains about 1° only in
  sit-to-stand.
- **The roundness measures do not follow the lab's lumbar flexion within a trial:** the bend and the
  bulge correlate at r = −0.14 to 0.32.
- **This does not prove the outline is useless for a rounded back.** The lab model has a single lumbar
  joint, so it measures the angle between the pelvis and the thorax, not the curve. The participants are
  healthy, wear tight tops and do not round their backs. Nothing was measured on cycling. **It is not added
  to the app as a measure.**
- **A constant offset would help the hip.** Fitting the hip as thigh + a·lean + b gives smaller errors
  than the app's fixed 0.75 (squat 7.2 → 5.0°, sit-to-stand 6.5 → 4.4°). Most of that is the constant
  b, which changes with the movement and the camera. Like the ankle's offset, it is left out.

## Every measure against the literature (1 November 2026)

After the hip, every measure the program reports was compared with how the literature defines it. Seven
things were changed. Each is measured where there was something to measure it against.

**1. The low-pass filter before speeds.** Everything was smoothed at 6 Hz, Winter's value for walking.
Against the OpenCap lab (lab angles at 100 Hz, filtered at 15 Hz), peak angular speeds of the 2D knee and
hip, mean of the trials:

| | Lab peak | 6 Hz | 12 Hz | Winter's residual analysis (median cutoff) |
|---|---|---|---|---|
| Drop jump, knee | 855 °/s | −25 % | −6 % | −15 % (8 Hz) |
| Drop jump, hip | 534 °/s | −17 % | −4 % | −11 % (8 Hz) |
| Squat, knee | 223 °/s | +11 % | +32 % | +17 % (7 Hz) |
| Gait, knee | 418 °/s | +13 % | +88 % | +17 % (7 Hz) |

- **Fast movements lose their peaks at 6 Hz; slow ones gain noise at higher cutoffs.**
- **A kick is faster than anything in OpenCap.** A foot reaching 20 m/s in a 70 ms swing, filtered at
  6 Hz, reads 14.5 m/s at 60 fps and 13.6 at 30 fps, whatever the frame rate. At 15 Hz it reads 18.9 m/s
  at 60 fps. At 30 fps it cannot go above 16.5 m/s even unfiltered.
- **Residual analysis chose 7–8 Hz everywhere** and only halved the loss in the jump.
- **What the app does now.** The cutoff belongs to the template (`cutoff_hz`): 6 Hz by default, 12 Hz in
  Vertical jump, 15 Hz in Kick, always below half the frame rate. The velocities report says which
  cutoff was used. It warns when the video has fewer frames per second than the template needs
  (`min_fps`, 60 in both).

**2. Acceleration.** The velocities plugin reported the rate of change of the speed. That is only the
part of the acceleration along the path: a foot going round a circle at a steady speed came out at 0.
It now reports the size of the second derivative of the position, as the literature defines it.

**3. 3D scale of someone who never stands up.** Without tape measures, the 3D is scaled so that the top
of the head to the heels, in the most upright frames, equals the height. That holds for anyone who
stands at some moment: in the 36 OpenCap trials the shank comes out 0.973–1.078 of the lab markers'
(median 1.009). A rider on a trainer never stands: in the sessions at 93 and 95 cm the head-to-heels
distance is 0.89 of the height. Every length in centimetres would come out 11–12 % too big (checked
against the same sessions scaled with the tape).

Now, when the head-to-heels distance never reaches 0.90 of the height the legs suggest (riders 0.86–0.87,
standing people 0.945 or more), the scale comes
from the legs instead. The detected thigh plus shank is 0.468 of the height in the OpenCap participants
standing, against 0.491 anatomically (Winter), because the detector puts the hip and the knee a little
closer.
- **The two sessions:** −2.9 % and −3.3 % from the tape (they were +12.4 % and +10.6 %).
- **Standing people:** never trigger it. Their ratio was 0.945 or more, and none of the 36 OpenCap
  trials changed.
- **Legs for everyone was tried too:** standing people then came out 7 % off on average.

**4. Height in the 2D OpenSim export.** Without a calibration the export took nose-to-ankle as 94 % of
the height. In the OpenCap 3D, standing upright, it is 87.6 % (27 trials, 84.4–88.5 %), which matches
Winter's proportions (nose ≈ 0.91, ankle 0.039 of the height). Everything came out about 7 % small. It
now uses 0.876.

**5. Body mass index under 18.** The WHO adult classes were shown at any age. Under 18 the BMI is read
on percentiles for age and sex (WHO growth references), which the program does not hold. The value is
still shown, but without a class.

**6. Cycling's "bottom" and "top".** The program measures the knee at its most stretched and most bent
while pedalling, a dynamic value. Bike-fitting ranges are often given for a goniometer with the crank
held at the bottom (Holmes et al. 1994: 25–35° of knee flexion), which reads differently from the
dynamic value (Ferrer-Roca et al. 2012 recommend 30–40° for dynamic measures). Recovery and propulsion
are split at these two moments, not at the crank's position. The reports now say so.

**7. "Beyond the variation" in comparisons.** The ● compares a difference with how much the measure
changes from one cycle to the next within each recording. Between two recordings, for example two
saddle heights on two days, it does not include the error of placing the camera again. The note under
every comparison now says so.

**Checked and left as they were:**
- the centre of mass: de Leva's 1996 fractions, with the trunk from the mid-shoulders;
- Perry's eight phases (their shares and events);
- cadence, stance, stride and step length and step width;
- the BMI cut-offs for adults;
- leg length with a tape (ASIS to medial malleolus);
- the cycling trunk angle (shoulder–hip line above the horizontal);
- the closed hip angle (trunk–thigh), as bike fitting measures it.
