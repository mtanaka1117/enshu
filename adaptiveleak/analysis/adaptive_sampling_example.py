import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, LINE_WIDTH

Sample = namedtuple('Sample', ['x', 'y'])

SEQUENCE_ONE = [1, 1.5, 2, 2, 2, 2, 1, 1, 1]
UNIFORM_ONE = Sample(x=[0, 3, 6], y=[1, 2, 1])
ADAPTIVE_ONE = Sample(x=[0, 2, 5, 6], y=[1, 2, 2, 1])

SEQUENCE_TWO = [1, 1.5, 2, 2, 2, 2, 2, 2, 2]
UNIFORM_TWO = Sample(x=[0, 3, 6], y=[1, 2, 2])
ADAPTIVE_TWO = Sample(x=[0, 2], y=[1, 2])


# Parameters
sample_one = UNIFORM_ONE
sample_two = UNIFORM_TWO

label='Uniform'
color='orange'

output_path = '../plots/2021-07-26/uniform.pdf'

#sample_one = ADAPTIVE_ONE
#sample_two = ADAPTIVE_TWO
#
#label='Optimal'
#color='green'
#
#output_path = '../plots/2021-07-26/optimal.pdf'


true_one = np.array(SEQUENCE_ONE)
true_two = np.array(SEQUENCE_TWO)


with plt.style.context(PLOT_STYLE):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(PLOT_SIZE[0] * 0.75, PLOT_SIZE[1] * 0.75), sharex=True)

    xs = list(range(len(true_one)))

    rec_one = np.interp(x=xs, xp=sample_one.x, fp=sample_one.y)
    rec_two = np.interp(x=xs, xp=sample_two.x, fp=sample_two.y)

    # Plot the first sequence
    ax1.plot(xs, true_one, label='True', linewidth=LINE_WIDTH)
    ax1.plot(xs, rec_one, label=label, linewidth=LINE_WIDTH, color=color)
    
    ax1.scatter(xs, true_one, marker='x', linewidth=LINE_WIDTH)
    ax1.scatter(sample_one.x, sample_one.y, marker='o', linewidth=LINE_WIDTH, color=color)

    # Plot the second sequence
    ax2.plot(xs, true_two, label='True', linewidth=LINE_WIDTH)
    ax2.plot(xs, rec_two, label=label, linewidth=LINE_WIDTH, color=color)
    
    ax2.scatter(xs, true_two, marker='x', linewidth=LINE_WIDTH)
    ax2.scatter(sample_two.x, sample_two.y, marker='o', linewidth=LINE_WIDTH, color=color)

    # Set the labels
    ax1.set_ylabel('Signal Value', fontsize=AXIS_FONT)
    ax1.set_title('Sequence One', fontsize=TITLE_FONT)

    ax2.set_xlabel('Time Step', fontsize=AXIS_FONT)
    ax2.set_ylabel('Signal Value', fontsize=AXIS_FONT)
    ax2.set_title('Sequence Two', fontsize=TITLE_FONT)

    ax1.legend(fontsize=LEGEND_FONT)
    ax2.legend(fontsize=LEGEND_FONT)

    error_one = np.average(np.abs(rec_one - true_one))
    error_two = np.average(np.abs(rec_two - true_two))

    ax1.text(6, 1.5, s='MAE: {0:.3f}'.format(error_one), fontsize=LEGEND_FONT)
    ax1.text(6, 1.35, s='# Collected: {0}'.format(len(sample_one.x)), fontsize=LEGEND_FONT)

    ax2.text(6, 1.65, s='MAE: {0:.3f}'.format(error_two), fontsize=LEGEND_FONT)
    ax2.text(6, 1.5, s='# Collected: {0}'.format(len(sample_two.x)), fontsize=LEGEND_FONT)


    plt.savefig(output_path, bbox_inches='tight', transparent=True)
