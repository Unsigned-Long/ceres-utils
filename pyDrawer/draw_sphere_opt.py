import matplotlib.pyplot as plt
import numpy as np

points_file_path = '/home/csl/CppWorks/artwork/ceres-utils/output/sphere_opt.txt'
figure_save_path = '/home/csl/CppWorks/artwork/ceres-utils/img/sphere_opt.png'

target = [1, 1, 1]

# setting
config = {
    # "text.usetex": True,
    "font.family": 'serif',  # sans-serif/serif/cursive/fantasy/monospace
    "font.size": 12,  # medium/large/small
    'font.style': 'normal',  # normal/italic/oblique
    'font.weight': 'normal',  # bold
    "mathtext.fontset": 'cm',
    "font.serif": ['cmb10'],
    "axes.unicode_minus": False,
}


def read_points(filename):
    points = []
    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
        items = line.split(',')
        points.append([float(items[0]), float(items[1]), float(items[2])])

    return points


def plot_globe(a, b, c, r, color, ax, dense=1000):
    t1 = np.linspace(0, np.pi, dense)
    t2 = np.linspace(0, np.pi * 2, dense)
    t1, t2 = np.meshgrid(t1, t2)
    x = a + r * np.sin(t1) * np.cos(t2)
    y = b + r * np.sin(t1) * np.sin(t2)
    z = c + r * np.cos(t1)
    ax.plot_surface(x, y, z, color=color, alpha=0.2)


if __name__ == '__main__':
    plt.rcParams.update(config)
    plt.rcParams['figure.figsize'] = (10, 10)
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X($m$)')
    ax.set_ylabel('Y($m$)')
    ax.set_zlabel('Z($m$)')
    ax.set_box_aspect((1, 1, 1))
    arrow_length = 1
    line_width = 3
    mark_size = 5
    marker = '8'
    # draw x, y, z axis
    ax.plot(
        [0, arrow_length], [0, 0], [0, 0], linestyle='-', ms=mark_size, marker=marker, c='red', lw=line_width
    )
    ax.plot(
        [0, 0], [0, arrow_length], [0, 0], linestyle='-', ms=mark_size, marker=marker, c='green', lw=line_width
    )
    ax.plot(
        [0, 0], [0, 0], [0, arrow_length], linestyle='-', ms=mark_size, marker=marker, c='blue', lw=line_width
    )
    ax.plot(
        [0], [0], [0], linestyle='-', ms=mark_size, marker=marker, c='black', lw=line_width
    )

    radius = np.linalg.norm(target)
    ax.scatter(
        [target[0]], [target[1]], [target[2]], c='blue', marker='X', s=100
    )
    ax.plot(
        [0, target[0]], [0, target[1]], [0, target[2]], linestyle='-', c='blue', lw=2
    )
    plot_globe(0, 0, 0, radius, 'blue', ax)

    points = read_points(points_file_path)

    ax.plot(
        [elem[0] for elem in points], [elem[1] for elem in points], [elem[2] for elem in points],
        linestyle='-', c='red', lw=2, marker='X', mec='green', mfc='green'
    )
    plot_globe(0, 0, 0, np.linalg.norm(points[0]), 'red', ax)

    plt.tight_layout()
    plt.savefig(figure_save_path)
    plt.show()
