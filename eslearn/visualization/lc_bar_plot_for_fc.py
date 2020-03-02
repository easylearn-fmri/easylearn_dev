# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:01:21 2019

@author: lenovo
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(
    r'D:\My_Codes\LC_Machine_Learning\lc_rsfmri_tools\lc_rsfmri_tools_python')
from Utils.lc_read_write_Mat import read_mat
from Plot.lc_barplot import BarPlot


class ExtractFC():
    """
    Backgrounds: For plot bar of my paper about dynamic FC

    Goal: Extract fc from fc matrix according to a given mask

    How: Extract data in mask from 2D fc matrix to a 1D vector for each fc file
    Identify group and 2 roi name for each fc each extracted data vector

    NOTE: All fc file in a folder

    Attrs:
        fc_folder: directory of folder that contain fc files
        mask: mask used to extract fc vector from fc matrix
        roiname: excel file that contain each node name in network
        roiname_col: roi name in which column, e.g. 3
        group_name: give a group name to the fc vector. e.g. 'SZ' or 'BD' or 'MDD'

    Returns:
        extracted_data:
            A 2D matrix, with each row for one fc vector extracted from one fc file
            dim = N*(M+1), N = number of fc files, M = how many 'Trues' in mask,
            '1' means the groups column
    """
    def __init__(sel, fc_folder=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\state\allState17_4\state4_all\state1\state1_HC',
                 mask=r'D:\WorkStation_2018\WorkStation_dynamicFC\Data\zDynamic\state\allState17_4\state4_all\state1\result1\shared_1and2and3_fdr.mat',
                 roiname=r'D:\My_Codes\Github_Related\Github_Code\Template_Yeo2011\17network_label.xlsx',
                 group_name='HC'):

        sel.fc_folder = fc_folder
        sel.mask = mask
        sel.roiname = roiname
        sel.group_name = group_name
        sel._huename = 'Groups'

    def _get_all_data(sel):
        # get all fc data
        path = os.listdir(sel.fc_folder)
        path = (os.path.join(sel.fc_folder, pt) for pt in path)
        print('loading fc matrix of all fc files\n')
        sel._data = (read_mat(pt) for pt in path)

        # get mask data
        sel._maskdata = read_mat(sel.mask)

    def _extract(sel):
        sel._get_all_data()
        print('extracting data from mask\n')
        extracted_data = [np.array(data)[np.array(
            sel._maskdata == 1)] for data in sel._data]
        sel._extracted_data = pd.DataFrame(extracted_data)

    def _add_roipairname(sel):
        """
        add roi paire name to each fc
        FIX: auto identify wihich hemisphere and roi name? case-sensitive!
        FIX:col 2 and col 1
        """
        # extract roi pair name from excel
        roi = pd.read_excel(sel.roiname, header=None)
        which_hemisphere = roi.iloc[:, 0]
        which_hemisphere = [
            hem.split('_')[1][0] + ' ' for hem in which_hemisphere]
        roiname = roi.iloc[:, 1]
        roiname = which_hemisphere + roiname
        roiname = [n.strip() for n in roiname]
        # make each fc name is unique so that sns.barplot plot every fc
        roiname = [n + '(' + str(i) + ')' for i, n in enumerate(roiname)]

        idxi, idxj = np.where(sel._maskdata == 1)
        roipairname = [roiname[i] + '--' + roiname[j]
                       for i, j in zip(idxi, idxj)]
        sel._extracted_data.columns = roipairname
        return sel._extracted_data

    def _add_groupname(sel):
        groupname = pd.DataFrame(
            [sel.group_name] * np.shape(sel._extracted_data)[0])
        extracted_data = pd.concat([groupname, sel._extracted_data], axis=1)
        # change columns name
        colname = list(extracted_data.columns)
        colname[0] = sel._huename
        extracted_data.columns = colname
        return extracted_data

    def _extract_one(sel, rootpath, whichstate, group_name):
        """
        Equal to main function of ExtractFC class
        Only for _extract_all
        """
        sel = ExtractFC(fc_folder=os.path.join(rootpath, whichstate, whichstate + '_' + group_name),
                        mask=os.path.join(rootpath, whichstate,
                                          'result1', 'shared_1and2and3_fdr.mat'),
                        roiname=r'D:\WorkStation_2018\Workstation_dynamic_FC_V2\Data\Network_and_plot_para\17network_label.xlsx',
                        group_name=group_name)
        sel._extract()
        sel._add_roipairname()
        data = sel._add_groupname()
        return data

    def _extract_all(sel,
                     rootpath=r'D:\WorkStation_2018\WorkStation_dynamicFC\Workstation_dynamic_fc_baobaoComputer\Data\Dynamic',
                     whichstate='state1'):
        """
        Extract all groups fc data for one state, and concat them
        Only use for my paper
        """

        # hc
        group_name = 'HC'
        data_hc = sel._extract_one(rootpath, whichstate, group_name)

        # mdd
        group_name = 'MDD'
        data_mdd = sel._extract_one(rootpath, whichstate, group_name)

        # bd
        group_name = 'BD'
        data_bd = sel._extract_one(rootpath, whichstate, group_name)

        # sz
        group_name = 'SZ'
        data_sz = sel._extract_one(rootpath, whichstate, group_name)

        # concat
        data_all = pd.concat([data_hc, data_mdd, data_bd, data_sz], axis=0)
        data_all.index = np.arange(0, np.shape(data_all)[0])

        return data_all

    def _group_fc_accordingto_fctype(sel, data_all):
        """
        group fc according to intra- or inter-network fc
        """
        colname = pd.Series(data_all.columns)
        colname = colname[1:]
        fcname = [n.split('--') for n in colname]

        idx = []
        for i, nn in enumerate(fcname):
            pn = [n.split(' ')[1] for n in nn]
            if pn[0] == pn[1]:
                idx.append(i)
        colname_intra = colname.iloc[idx]
        colname_intre = list(set(data_all.columns) - set(colname_intra))
        colname_intra = list(colname_intra.iloc[:])
        colname_intra.append(sel._huename)

        data_all_intra = data_all[colname_intra]
        data_all_inter = data_all[colname_intre]
        return data_all_intra, data_all_inter

    def _order_fc_accordingto_networkname(sel, data):
        """
        Order the fc columns name according network name
        Make the bar sorted according network name
        """
        colname = list(data.columns)
        fcname = [name.split(' ')[1] if len(name.split(' '))
                             > 1 else name.split(' ')[0] for name in colname]
        # sorted by the first item of each str
        idx = [i for i, v in sorted(enumerate(fcname), key=lambda x:x[1])]
        sorted_colname = [colname[id] for id in idx]
        sorted_data = data[sorted_colname]
        return sorted_data


class BarPlotForFC(BarPlot):
    """
    plot bar for my paper about dynamic FC
    group bar according to intra- and inter-network
    """
    def __init__(sel, x_location=np.arange(1, 39), savename='fig.tiff'):
        super().__init__()
        sel.x_location = x_location
        sel.hue_name = 'Groups'
        sel.hue_order = None
        sel.if_save_figure = 0
        sel.savename = savename
        sel.x_name = 'FC'
        sel.y_name = 'Z value'
        sel.if_save_figure = 1,

    def prepdata(sel, df):
        df = sel.data_preparation(df)
        return df

    def _plot(sel, data):
        sel.plot(data)


if __name__ == '__main__':
    sel = ExtractFC()
    data_all = sel._extract_all(
        rootpath=r'D:\WorkStation_2018\Workstation_dynamic_FC_V2\Data\Dynamic',
        whichstate='state4')
    
    data_all = sel._group_fc_accordingto_fctype(data_all)
    data_intra = data_all[0]
    data_inter = data_all[1]
    data_intra = sel._order_fc_accordingto_networkname(data_intra)

    # plot
    loc = list(set(data_intra.columns) - set(['Groups']))
    # original order
    loc.sort(key = list(data_intra.columns).index)
    sel = BarPlotForFC(x_location=loc, savename=r'D:\WorkStation_2018\Workstation_dynamic_FC_V2\Figure\Supp\Bar\bar_intranetwork_s4.tiff')
    prepdata = sel.prepdata(data_intra)
    # sel._plot(prepdata)

    loc = list(set(data_inter.columns) - set(['Groups']))
    loc.sort(key = list(data_inter.columns).index)
    sel=BarPlotForFC(x_location=loc, savename=r'D:\WorkStation_2018\Workstation_dynamic_FC_V2\Figure\Supp\Bar\bar_internetwork_s4.tiff')
    prepdata=sel.prepdata(data_inter)
    # sel._plot(prepdata)
    