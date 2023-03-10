# -*- coding: utf-8 -*-`
"""This example demonstrates how to fully customize violin plots.

This code is copy and revised from https://cloud.tencent.com/developer/article/1486970

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import mplcyberpunk


cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

cnames = {
    'whitesmoke':           'r',
    'yellow':               'g',
    'yellowgreen':          'b'
}

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
        
        # ax.set_title('')
        fig = plt.figure(figsize=(4,5), facecolor='k', edgecolor='k')
        self.ax = plt.gca()
        data = list([np.array(d) for  d in data])
        parts = sns.violinplot(data=data, linewidth=1, width=1, axes=self.ax, palette="Set2")
        sns.swarmplot(data=data, color="white", alpha=0.5)
        
        # Set backgroud axis and spines
        sns.set(rc={'axes.facecolor':'black', 
                    'figure.facecolor':'black',
                    'figure.edgecolor':'white',
                    'axes.grid' : False})
        
                 
        plt.tick_params(axis='x',colors='w', labelsize=15)
        plt.tick_params(axis='y',colors='w', labelsize=15)
        self.ax.spines['top'].set_visible(False) 
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('w')
        self.ax.spines['bottom'].set_color('w')
        self.ax.spines['left'].set_linewidth(2)
        self.ax.spines['bottom'].set_linewidth(2)
        # plt.grid('off')
        
        # set style for the axes
        # self.set_axis_style(xlabel, ylabel, xlabelsize, ylabelsize, xticklabel, yticklabel, ticklabelfontsize, xticklabel_rotation)
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
        
        # Set backgroud axis and spines
        plt.figure(figsize=(4,5), facecolor='k', edgecolor='w')
        plt.axes(facecolor='k')
        plt.tick_params(axis='x',colors='w', labelsize=15)
        plt.tick_params(axis='y',colors='w', labelsize=15)
        ax = plt.gca()
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('w')
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Plot violin
        parts = plt.violinplot(data,
                               showmeans=False, 
                               showmedians=False, 
                               showextrema=False, 
                               positions=inds)
        sns.swarmplot(data=data, color="white", alpha=0.5)
        
        # Set plt
        self.set_plt(kwargs, parts)
        
        plt.scatter(inds, medians, marker='o', color='k', s=10, zorder=3)
        plt.vlines(inds, quartile1, quartile3, color='w', linestyle='-', lw=5)
        plt.vlines(inds, whiskersMin, whiskersMax, color='w', linestyle='-', lw=1)
        mplcyberpunk.add_glow_effects()

    
    def adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    
    @ staticmethod
    def set_plt(kwargs, parts):
        ckey = list(cnames.keys())
        idex = np.random.permutation(len(cnames))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cnames[ckey[i]])
            pc.set_edgecolor(cnames[ckey[i]])
            pc.set_linewidth(3)
            pc.set_alpha(0.6)
                
        keys = set(kwargs.keys()) - set(["positions"])
        if "facecolor" not in kwargs.keys(): 
            keys = set(keys) - set(["facecolor"])
        
        for key in keys:
            for i, pc in enumerate(parts['bodies']):
                cmd = f"pc.set_{key}(kwargs['{key}'][{i}])"
                eval(cmd)
                
                
    
if __name__ == "__main__":
    np.random.seed(666)
    data = [np.random.randn(110,), np.random.randn(100,), np.random.randn(100,)]
    violin = ViolinPlot()
    f1 = violin.plot(data, xticklabel=['1111','',''])
    plt.title("ViolinPlotSeaborn", color="k")
    plt.show()
    
    ViolinPlotMatplotlib().plot(data, 
                                positions=np.arange(0,len(data)), 
                                facecolor=['r', 'g', 'b'],
                                edgecolor=['r', 'g', 'b'],
                                alpha=[.5,.5,.5])
    plt.title("ViolinPlotMatplotlib", color="w")
    plt.show()

