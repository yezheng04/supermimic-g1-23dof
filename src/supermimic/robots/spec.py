"""Single source of truth for the native23 tensor contract (no padded 29-slot state)."""
JOINT_NAMES = ('left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint')
NJ = len(JOINT_NAMES)
NQ = 7 + NJ
NV = 6 + NJ
STATE_DIM = NQ + NV
FEAT_DIM = 13 + 2 * NJ
DELTA_DIM = 12 + 2 * NJ
XI_DIM = 1 + NJ + 3
GOAL_FEAT_DIM = 2 * NJ + 10
OBS_DIM = 3 * NJ + 6
STATE_SLICES = (slice(0,3), slice(3,7), slice(7,NQ), slice(NQ,NQ+3), slice(NQ+3,NQ+6), slice(NQ+6,STATE_DIM))
DELTA_SLICES = (slice(0,3), slice(3,6), slice(6,6+NJ), slice(6+NJ,9+NJ), slice(9+NJ,12+NJ), slice(12+NJ,DELTA_DIM))


def split_state(state):
    return tuple(state[..., s] for s in STATE_SLICES)
