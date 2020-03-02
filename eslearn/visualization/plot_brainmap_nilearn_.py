# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:49:33 2019

@author: Chao Li
Email: lichao19870617@gmail.com
"""

"""
Making a surface plot of a 3D statistical map
=============================================
project a 3D statistical map onto a cortical mesh using
:func:`nilearn.surface.vol_to_surf`. Display a surface plot of the projected
map using :func:`nilearn.plotting.plot_surf_stat_map`.
"""

##############################################################################
# Get a statistical map
# ---------------------

from nilearn import datasets
from nilearn import plotting
import matplotlib.pyplot as plt
import sys
sys.path.append(r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
import Utils.lc_niiProcessor as NiiProc
img = r'D:\WorkStation_2018\WorkStation_CNN_Schizo\Data\Atalas\sorted_brainnetome_atalas_3mm.nii'

niiproc = NiiProc.NiiProcessor()
stat_img_data, stat_img = niiproc.read_sigle_nii(img)

plotting.plot_roi(stat_img, cmap=plotting.cm.bwr, colorbar=True)
plt.show()
# motor_images = datasets.fetch_neurovault_motor_task()
# stat_img = motor_images.images[0]


##############################################################################
# Get a cortical mesh
# -------------------

# fsaverage = datasets.fetch_surf_fsaverage()

# ##############################################################################
# # Sample the 3D data around each node of the mesh
# # -----------------------------------------------

# from nilearn import surface

# texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)

# ##############################################################################
# # Plot the result
# # ---------------



# plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
#                             title='Surface right hemisphere', colorbar=True,
#                             threshold=3., bg_map=fsaverage.sulc_right)

# ##############################################################################
# # Plot 3D image for comparison
# # ----------------------------

# plotting.plot_glass_brain(stat_img, display_mode='r', plot_abs=False,
#                           title='Glass brain', threshold=3.)

# plotting.plot_stat_map(stat_img, display_mode='z', threshold=3.,
#                        cut_coords=range(-30, -10, 3), title='Slices')

# plotting.plot_stat_map(stat_img, display_mode='z', threshold=3.,
#                        cut_coords=range(-10, 10, 3), title='Slices')

# plotting.plot_stat_map(stat_img, display_mode='z', threshold=3.,
#                        cut_coords=range(10, 30, 3), title='Slices')

# plotting.plot_stat_map(stat_img, display_mode='z', threshold=3.,
#                        cut_coords=range(30, 50, 3), title='Slices')


# ##############################################################################
# # Plot with higher-resolution mesh
# # --------------------------------
# #
# # `fetch_surf_fsaverage` takes a "mesh" argument which specifies
# # wether to fetch the low-resolution fsaverage5 mesh, or the high-resolution
# # fsaverage mesh. using mesh="fsaverage" will result in more memory usage and
# # computation time, but finer visualizations.

# big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
# big_texture = surface.vol_to_surf(stat_img, big_fsaverage.pial_right)

# plotting.plot_surf_stat_map(big_fsaverage.infl_right,
#                             big_texture, hemi='right', colorbar=True,
#                             title='Surface right hemisphere: fine mesh',
#                             threshold=1., bg_map=big_fsaverage.sulc_right)


# plotting.show()


# ##############################################################################
# # 3D visualization in a web browser
# # ---------------------------------
# # An alternative to :func:`nilearn.plotting.plot_surf_stat_map` is to use
# # :func:`nilearn.plotting.view_surf` or
# # :func:`nilearn.plotting.view_img_on_surf` that give more interactive
# # visualizations in a web browser. See :ref:`interactive-surface-plotting` for
# # more details.

# view = plotting.view_surf(fsaverage.infl_right, texture, threshold='90%',
#                           bg_map=fsaverage.sulc_right)

# # In a Jupyter notebook, if ``view`` is the output of a cell, it will
# # be displayed below the cell
# view

# ##############################################################################

# # uncomment this to open the plot in a web browser:
# # view.open_in_browser()

# ##############################################################################
# # We don't need to do the projection ourselves, we can use view_img_on_surf:

# view = plotting.view_img_on_surf(stat_img, threshold='90%')
# # view.open_in_browser()

# view
