
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


import numpy as np
import pytest
import matplotlib.pyplot as plt

from mne.viz import plot_connectivity_circle, circular_layout


def test_plot_connectivity_circle():
    """Test plotting connectivity circle."""
    node_order = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff']
    
    label_names = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff']

    group_boundaries = [0, 2, 4]
    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    con = np.random.RandomState(0).randn(6, 6)
    con[con < 0.7] = 0
    
    fig, ax = plot_connectivity_circle(con, label_names, n_lines=60,
                             node_angles=node_angles, title='test',
                             colormap='RdBu_r', vmin=0, vmax=2, linewidth=.5,facecolor='k')
    plt.show()

    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[-1])
    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[20, 0])
    # plt.close('all')


if __name__ == "__main__":
    test_plot_connectivity_circle()