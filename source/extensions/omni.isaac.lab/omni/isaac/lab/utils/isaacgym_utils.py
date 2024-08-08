def reorder_joints_from_isaaclab_to_isaacgym(isaaclab_joints):
    # IsaacLab
    # FL hip,   FR hip,   RL hip,   RR hip
    # FL thigh, FR thigh, RL thigh, RR thigh
    # FL calf,  FF calf,  RL calf,  RR calf
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # IsaacGym
    # FL hip, FL thigh, FL calf
    # FR hip, FR thigh, FR calf
    # RL hip, RL thigh, RL calf
    # RR hip, RR thigh, RR calf
    # [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

    isaacgym_joint_indices = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    isaacgym_joint = isaaclab_joints[:, isaacgym_joint_indices]

    return isaacgym_joint


def reorder_joints_from_isaacgym_to_isaaclab(isaacgym_joints):
    # IsaacGym
    # FL hip, FL thigh, FL calf
    # FR hip, FR thigh, FR calf
    # RL hip, RL thigh, RL calf
    # RR hip, RR thigh, RR calf
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # IsaacLab
    # FL hip,   FR hip,   RL hip,   RR hip
    # FL thigh, FR thigh, RL thigh, RR thigh
    # FL calf,  FF calf,  RL calf,  RR calf
    # [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    isaaclab_joint_indices = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    isaaclab_joint = isaacgym_joints[:, isaaclab_joint_indices]

    return isaaclab_joint
