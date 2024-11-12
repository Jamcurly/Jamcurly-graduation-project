import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time

def dh_transformation_matrix(theta, d, a, alpha):
    """Calculate the Denavit-Hartenberg transformation matrix"""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,             np.sin(alpha),                  np.cos(alpha),                 d],
        [0,             0,                              0,                             1]
    ])

def forward_kinematics(joint_angles):
    """Compute the forward kinematics and collect points for visualization"""
    # Define the DH parameters for each joint (example values, adjust as needed)
    dh_parameters = [
        {"theta": joint_angles[0], "d": 0.1, "a": 0,    "alpha": np.pi/2},
        {"theta": joint_angles[1], "d": 0,   "a": 0.5,  "alpha": 0},
        {"theta": joint_angles[2], "d": 0,   "a": 0.3,  "alpha": 0},
        {"theta": joint_angles[3], "d": 0.2, "a": 0,    "alpha": np.pi/2},
        {"theta": joint_angles[4], "d": 0,   "a": 0,    "alpha": -np.pi/2},
        {"theta": joint_angles[5], "d": 0.1, "a": 0,    "alpha": 0}
    ]

    # Initial transformation matrix (identity matrix)
    t_matrix = np.eye(4)

    # Store points for visualization
    points = [t_matrix[:3, 3].copy()]  # Start with the base

    # Calculate the transformation matrix through each joint
    for i in range(6):
        dh = dh_parameters[i]
        t_matrix = np.dot(t_matrix, dh_transformation_matrix(dh["theta"], dh["d"], dh["a"], dh["alpha"]))
        points.append(t_matrix[:3, 3].copy())  # Store the position of each joint

    return points

def plot_robot_arm(joint_angles):
    """Plot the 3D visualization of the robotic arm"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Turn on interactive mode for real-time plotting

    while True:
        # Compute the positions of each joint
        arm_points = forward_kinematics(joint_angles)

        # Clear previous plot
        ax.clear()

        # Unpack x, y, z coordinates
        x = np.array([point[0] for point in arm_points])
        y = np.array([point[1] for point in arm_points])
        z = np.array([point[2] for point in arm_points])

        # Plot the arm
        ax.plot(x, y, z, '-o', markersize=8, linewidth=2)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('6-DOF Robotic Arm Visualization')
        ax.grid(True)

        # Set equal scaling
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.draw()
        plt.pause(0.05)  # Pause for a brief moment to update the plot

def update_joint_angles(joint_angles):
    """Simulate joint angles update in a separate thread"""
    while True:
        for i in range(len(joint_angles)):
            joint_angles[i] += 0.01  # Incrementally change the joint angles
        time.sleep(0.05)  # Control the speed of the updates

# Initialize the robotic arm with joint angles (example values in radians)
joint_angles = [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, 0]

# Start the angle updating thread
thread = threading.Thread(target=update_joint_angles, args=(joint_angles,))
thread.daemon = True  # This will allow the thread to exit when the main program exits
thread.start()

# Plot the robotic arm
plot_robot_arm(joint_angles)
