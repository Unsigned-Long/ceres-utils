import json
import matplotlib.pyplot as plt
import numpy as np

filename = '../output/equation_before.json'
savename = '../img/equation_before.png'
# filename = '/home/csl/ros_ws/LIC-Calib/src/lic_calib/output/data_sim2/lm_equ_graph/batch_opt_0.json'
# savename = '../img/batch_opt_0_equation.png'

# ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
color_name = 'coolwarm'

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


def read_equation(filename):
    file = open(filename, "r")
    lines = file.readlines()
    content = ''
    for line in lines:
        content += line
    array_buffer = json.loads(content)
    h_matrix = array_buffer.get("h_matrix")
    b_vector = array_buffer.get("b_vector")
    param_info = []
    for elem in array_buffer.get("param_blocks"):
        name_dime = array_buffer.get("param_blocks").get(elem)
        param_info.append([name_dime['name'], name_dime['dime']])
    return [h_matrix, b_vector, param_info]


if __name__ == '__main__':
    plt.rcParams.update(config)
    plt.rcParams['figure.figsize'] = (12.0, 10)
    [h_matrix, b_vector, param_info] = read_equation(filename)

    # construct the equation
    equation = h_matrix
    for idx in range(len(b_vector)):
        equation[idx].append(0.0)
        equation[idx].append(b_vector[idx])
    equation = np.array(equation)

    # mapping
    equation = equation / np.abs(equation) * np.log10(np.abs(equation) + 1)

    # find value lim
    max = np.max(equation)
    min = np.min(equation)
    val_max = np.max([np.abs(min), np.abs(max)])

    # figure setting
    ax = plt.axes()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.pcolormesh(np.array(equation), cmap=color_name, vmin=-val_max, vmax=val_max)
    plt.colorbar()
    # count = 0
    # for idx in range(len(param_info) + 1):
    #     plt.axvline(count, color='w', linestyle='-', linewidth=2)
    #     plt.axhline(count, color='w', linestyle='-', linewidth=2)
    #     if idx == len(param_info):
    #         continue
    #     count += param_info[idx][1]
    # for idx in range(len(h_matrix)):
    #     plt.axvline(idx, color='w', linestyle='--', linewidth=1)
    #     plt.axhline(idx, color='w', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(savename)
    plt.show()
