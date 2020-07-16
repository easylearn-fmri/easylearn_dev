# -*- coding: utf-8 -*-`
"""This example demonstrates how to fully customize violin plots.

This code is copy and revised from https://cloud.tencent.com/developer/article/1486970

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ViolinPlot(object):
    """This class is used to plot violin
    """

    def plot(
            self, 
            data, 
            xlabel=None, 
            ylabel=None, 
            xlabelsize=10, 
            ylabelsize=10, 
            xticklabel=None, 
            yticklabel=None, 
            ticklabelfontsize=10,
            xticklabel_rotation=45
    ):
        """Plot

        Parameters:
        ----------
            data: list
                data to plot
            xlabel: str
                xlabel
            ylabel: str
                ylabel
            ....
        """

        fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6), sharey=True)
        
        # ax.set_title('')
        data = list([np.array(d) for  d in data])
        sns.violinplot(data=data, linewidth=1, ax=self.ax)
        # set style for the axes
        self.set_axis_style(xlabel, ylabel, xlabelsize, ylabelsize, xticklabel, yticklabel, ticklabelfontsize, xticklabel_rotation)
        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        # plt.show()

    def set_axis_style(self, xlabel, ylabel, xlabelsize, ylabelsize, xticklabel, yticklabel, ticklabelfontsize, xticklabel_rotation):
        self.ax.get_xaxis().set_tick_params(direction='out')
        self.ax.xaxis.set_ticks_position('bottom')
        if xticklabel:
            self.ax.set_xticks(np.arange(0, len(xticklabel) ))
            self.ax.set_xticklabels(xticklabel, rotation=xticklabel_rotation, ha="right", fontsize=ticklabelfontsize)
            # self.ax.set_xlim(0.25, len(xlabel) + 0.75)
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=xlabelsize)
        if yticklabel:
            self.ax.set_yticks(np.arange(0, len(yticklabel)))
            self.ax.set_yticklabels(yticklabel)
            # self.ax.set_ylim(0.25, len(ylabel) + 0.75)
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=ylabelsize)

class ViolinPlotMatplotlib(object):
    """ 
    ViolinPlot customization 

    """

    def plot(self, data, **kwargs):
        """ Plot violin

        Parameters:
        ----------
            data: list
                each item in list is a array
            **kwargs: matplotlib kwargs
        """

        # Sorted data so that adjacent_values method can get sorted array
        data = [sorted(d) for d in data]

        perc = np.array([np.percentile(data_, [25, 50, 75]) for data_ in data])
        quartile1, medians, quartile3 = perc[:,0], perc[:,1], perc[:,2]
        
        whiskers = np.array([
            self.adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        if 'positions' in kwargs.keys():
            inds = kwargs['positions']
        else:
            inds = np.arange(1, len(medians) + 1)
        
        # Plot violin
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False, positions=inds)
        
        # Set color
        if 'facecolor' in kwargs.keys():
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(kwargs['facecolor'][i])
        if 'edgecolor' in kwargs.keys():
            for i, pc in enumerate(parts['bodies']):
                pc.set_edgecolor(kwargs['edgecolor'][i])
        
        plt.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    
    def adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

if __name__ == "__main__":
    np.random.seed(666)
    data = [np.random.randn(100,), np.random.randn(100,), np.random.randn(100,), ]
    # violin = ViolinPlot()
    # violin.plot(data, xticklabel=['1111','',''])
    # plt.show()
    ViolinPlotMatplotlib().plot(data, positions=[0, 1, 2], facecolor=['r', 'g', 'b'])
    # plt.grid(axis='y')
    # ViolinPlotMatplotlib().plot([data[1]], positions=[1])
    # plt.grid(axis='y')
    # ViolinPlotMatplotlib().plot([data[2]], positions=[2])
    # plt.grid(axis='y')
    plt.show()
