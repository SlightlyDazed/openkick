"""MediaPipe BlazePose landmark definitions, skeleton topology and angle presets."""

from __future__ import annotations

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
COUNT = len(NAMES)
INDEX = {n: i for i, n in enumerate(NAMES)}

# Human readable names (English source strings, translated through i18n).
DISPLAY = {
    "nose": "Nose", "left_eye_inner": "Left eye (inner)", "left_eye": "Left eye",
    "left_eye_outer": "Left eye (outer)", "right_eye_inner": "Right eye (inner)",
    "right_eye": "Right eye", "right_eye_outer": "Right eye (outer)",
    "left_ear": "Left ear", "right_ear": "Right ear",
    "mouth_left": "Mouth (left)", "mouth_right": "Mouth (right)",
    "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder",
    "left_elbow": "Left elbow", "right_elbow": "Right elbow",
    "left_wrist": "Left wrist", "right_wrist": "Right wrist",
    "left_pinky": "Left pinky", "right_pinky": "Right pinky",
    "left_index": "Left index", "right_index": "Right index",
    "left_thumb": "Left thumb", "right_thumb": "Right thumb",
    "left_hip": "Left hip", "right_hip": "Right hip",
    "left_knee": "Left knee", "right_knee": "Right knee",
    "left_ankle": "Left ankle", "right_ankle": "Right ankle",
    "left_heel": "Left heel", "right_heel": "Right heel",
    "left_foot_index": "Left toe", "right_foot_index": "Right toe",
}


def side(index: int) -> str:
    """'R', 'L' or 'C' (center) for a landmark index."""
    name = NAMES[index]
    if name.startswith("right") or name == "mouth_right":
        return "R"
    if name.startswith("left") or name == "mouth_left":
        return "L"
    return "C"


# Bones drawn for the skeleton overlay. Face details are left out on purpose:
# at sports-video resolution they add noise, not information.
BONES = [
    (11, 12), (11, 23), (12, 24), (23, 24),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),              # arms
    (15, 19), (16, 20),                                  # hands
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),    # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),    # right leg
]
# Joints drawn as handles (editable points).
JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Clicking one of these joints with the angle tool creates its standard angle:
# vertex -> (proximal, vertex, distal)
ANGLE_PRESETS = {
    INDEX["right_knee"]: ("right_hip", "right_knee", "right_ankle"),
    INDEX["left_knee"]: ("left_hip", "left_knee", "left_ankle"),
    INDEX["right_hip"]: ("right_shoulder", "right_hip", "right_knee"),
    INDEX["left_hip"]: ("left_shoulder", "left_hip", "left_knee"),
    INDEX["right_elbow"]: ("right_shoulder", "right_elbow", "right_wrist"),
    INDEX["left_elbow"]: ("left_shoulder", "left_elbow", "left_wrist"),
    INDEX["right_shoulder"]: ("right_elbow", "right_shoulder", "right_hip"),
    INDEX["left_shoulder"]: ("left_elbow", "left_shoulder", "left_hip"),
    INDEX["right_ankle"]: ("right_knee", "right_ankle", "right_foot_index"),
    INDEX["left_ankle"]: ("left_knee", "left_ankle", "left_foot_index"),
}

# OpenPose BODY_25 order used by Kinovea's "OpenPose" posture tool.
# Each entry lists the BlazePose landmarks averaged to build that point.
BODY25 = [
    ("nose",),
    ("right_shoulder", "left_shoulder"),   # neck
    ("right_shoulder",), ("right_elbow",), ("right_wrist",),
    ("left_shoulder",), ("left_elbow",), ("left_wrist",),
    ("right_hip", "left_hip"),             # mid hip
    ("right_hip",), ("right_knee",), ("right_ankle",),
    ("left_hip",), ("left_knee",), ("left_ankle",),
    ("right_eye",), ("left_eye",), ("right_ear",), ("left_ear",),
    ("left_foot_index",),                  # left big toe
    ("left_foot_index",),                  # left small toe (not in BlazePose)
    ("left_heel",),
    ("right_foot_index",),                 # right big toe
    ("right_foot_index",),                 # right small toe (not in BlazePose)
    ("right_heel",),
]
