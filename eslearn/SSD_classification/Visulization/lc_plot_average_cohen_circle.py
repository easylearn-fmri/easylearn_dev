cr# -*- coding: utf-8 -*-
"""
This script is used to plot cohen'd using circle format.
"""

import numpy as np
import pytest
import scipy.io as sio
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle, circular_layout


def test_plot_connectivity_circle():
    """
    Test plotting connectivity circle.
    """
    # node_order = ['Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'];
    # label_names = ['Amyg', 'BG', 'Tha', 'Hipp', 'Limbic', 'Visual', 'SomMot', 'Control', 'Default', 'DorsAttn',  'Sal/VentAttn'];

    node_order = [str(i) for i in range(246)];
    label_names = [str(i) for i in range(246)];

    # group_boundaries = [0, 2, 4, 6, 8, 10]

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    
    con_medicated = sio.loadmat(r'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_medicated1.mat')
    con_unmedicated = sio.loadmat(r'D:\WorkStation_2018\SZ_classification\Data\Stat_results\cohen_feu1.mat')
    con_medicated = con_medicated['cohen_medicated']
    con_unmedicated = con_unmedicated['cohen_feu']
    con_medicated[np.abs(con_medicated) <= 0.5] = 0
    con_unmedicated[np.abs(con_unmedicated) <= 0.8] = 0
    
    figs, ax = plt.subplots(1,2, facecolor ='k')
    
    n_lines = np.sum(con_medicated[:] != 0)
    plot_connectivity_circle(con_medicated, label_names, n_lines=n_lines,
                             node_angles=node_angles, title='test',
                             colormap='RdBu_r', vmin=-1, vmax=1, linewidth=2,
                             fontsize_names=12, textcolor='k', facecolor='w', 
                             subplot=121, fig=figs, colorbar=True,)

    n_lines = np.sum(con_unmedicated[:] != 0)
    plot_connectivity_circle(con_unmedicated, label_names, n_lines=n_lines,
                         node_angles=node_angles, title='test',
                         colormap='RdBu_r', vmin=-1, vmax=1, linewidth=1.5,
                         fontsize_names=12, textcolor='k', facecolor='w',
                         subplot=122, fig=figs, colorbar=True)

    # plt.tight_layout()
    plt.subplots_adjust(wspace = 0.2, hspace = 0)   
    

    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[-1])
    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[20, 0])
    # plt.close('all')


if __name__ == "__main__":
    test_plot_connectivity_circle()