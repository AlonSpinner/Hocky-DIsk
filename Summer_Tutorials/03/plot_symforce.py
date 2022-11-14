import symforce.symbolic as sf
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d as a3


#copied and edited shamelessly from https://github.com/borglab/gtsam/blob/develop/python/gtsam/utils/plot.py

def plot_pose3_on_axes(axes : plt.Axes, pose : sf.Pose3, axis_length=0.1) -> None:
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.
    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
    """

    # get rotation and translation (center)
    gRp = pose.R.to_rotation_matrix().to_numpy()
    origin = pose.t.to_numpy().squeeze()

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')


def set_axes_equal(ax):

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def plot_line(ax : plt.Axes, p1, p2, color = 'black') -> None:
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    ax.plot(x,y,z,color = color)
