import matplotlib.pyplot as plt
import numpy as np

a = 0.5
b = 10
c = -5

ea = 0.493902
eb = 9.9255
ec = -4.56218

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
    lines = []
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        elems = line.split(',')
        points.append([float(elems[0]), float(elems[1])])
    return points


if __name__ == '__main__':
    plt.rcParams.update(config)
    plt.rcParams['figure.figsize'] = (8.0, 6.0)

    # ground truth
    mid = -b / (2 * a)
    x = np.linspace(start=mid - 10, stop=mid + 10, num=50)
    y = a * x * x + b * x + c
    plt.plot(x, y, 'r-', lw=2, alpha=0.5, label=r"$y=ax^2+bx+c$")

    # points
    points = read_points("/output/points.txt")
    plt.scatter([elem[0] for elem in points], [elem[1] for elem in points],
                marker='o', c='g', alpha=0.5, label=r"points")

    # estimate
    mid = -eb / (2 * ea)
    x = np.linspace(start=mid - 10, stop=mid + 10, num=50)
    y = ea * x * x + eb * x + ec
    plt.plot(x, y, 'b--', lw=2, alpha=0.5, label=r"$y=\hat{a}x^2+\hat{b}x+\hat{c}$")

    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()
    plt.title("Parabolic Fitting")
    plt.grid(ls='-.', alpha=0.5)
    plt.savefig("/home/csl/CppWorks/artwork/ceres-utils/src/img/parabolic_fitting.png")
    plt.show()
