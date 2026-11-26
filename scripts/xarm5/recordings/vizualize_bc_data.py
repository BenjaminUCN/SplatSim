import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_trajectory(traj_folder, traj_name):
    traj_path = os.path.join(traj_folder, traj_name)

    files = sorted(
        f for f in os.listdir(traj_path)
        if f.endswith(".pkl")
    )

    joint_positions = []
    actions = []

    for file in files:
        with open(os.path.join(traj_path, file), "rb") as f:
            data = pickle.load(f)

        joint_positions.append(data["joint_positions"])
        actions.append(data["action"])

    print(f"length of trajectory: {len(joint_positions)}")
    print(f"length of actions: {len(actions)}")
    return (
        np.asarray(joint_positions),
        np.asarray(actions)
    )


def visualize_trajectories(traj_folder):
    trajs = sorted(
        d for d in os.listdir(traj_folder)
        if os.path.isdir(os.path.join(traj_folder, d))
    )

    traj_idx = 0

    joints, actions = load_trajectory(traj_folder, trajs[traj_idx])

    print(f"joints shape: {joints.shape}")
    print(f"actions shape: {actions.shape}")
    n_steps, n_joints = joints.shape
    

    fig, axes = plt.subplots(
        n_joints,
        1,
        figsize=(12, 2 * n_joints),
        sharex=True
    )

    if n_joints == 1:
        axes = [axes]

    plt.subplots_adjust(bottom=0.15)

    joint_lines = []
    action_lines = []

    x = np.arange(n_steps)

    for j, ax in enumerate(axes):
        l1, = ax.plot(
            x,
            joints[:, j],
            label="joint_positions"
        )

        l2, = ax.plot(
            x,
            actions[:, j],
            "--",
            label="action"
        )

        ax.set_ylabel(f"J{j}")
        ax.grid(True)

        if j == 0:
            ax.legend()

        joint_lines.append(l1)
        action_lines.append(l2)

    axes[-1].set_xlabel("Timestep")

    slider_ax = plt.axes([0.15, 0.03, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        "Trajectory",
        0,
        len(trajs) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        idx = int(slider.val)

        joints, actions = load_trajectory(
            traj_folder,
            trajs[idx]
        )

        n_steps = len(joints)
        x = np.arange(n_steps)

        for j in range(n_joints):
            joint_lines[j].set_data(x, joints[:, j])
            action_lines[j].set_data(x, actions[:, j])

            axes[j].relim()
            axes[j].autoscale_view()

        fig.suptitle(trajs[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.suptitle(trajs[0])
    plt.show()
    
if __name__ == "__main__":
    traj_folder = os.path.expanduser("~/data/bc_data/gello")
    visualize_trajectories(traj_folder)