# -*- coding: utf-8 -*-`
"""This example demonstrates how to fully customize violin plots.

This code is copy and revised from https://cloud.tencent.com/developer/article/1486970

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ViolinPlot(object):
    """This class is used to plot violin
    """

    def plot(self, data, xlabel=None, ylabel=None, xlabelsize=10, ylabelsize=10, xticklabel=None, yticklabel=None, xticklabel_rotation=90):
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
        self.set_axis_style(xlabel, ylabel, xlabelsize, ylabelsize, xticklabel, yticklabel, xticklabel_rotation)
        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        # plt.show()

    def set_axis_style(self, xlabel, ylabel, xlabelsize, ylabelsize, xticklabel, yticklabel, xticklabel_rotation):
        self.ax.get_xaxis().set_tick_params(direction='out')
        self.ax.xaxis.set_ticks_position('bottom')
        if xticklabel:
            self.ax.set_xticks(np.arange(0, len(xticklabel) ))
            self.ax.set_xticklabels(xticklabel, rotation=xticklabel_rotation, ha="right")
            # self.ax.set_xlim(0.25, len(xlabel) + 0.75)
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=xlabelsize)
        if yticklabel:
            self.ax.set_yticks(np.arange(0, len(yticklabel)))
            self.ax.set_yticklabels(yticklabel)
            # self.ax.set_ylim(0.25, len(ylabel) + 0.75)
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=ylabelsize)


if __name__ == "__main__":
    np.random.seed(666)
    data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
    violin = ViolinPlot()
    violin.plot(data)
    plt.show()
