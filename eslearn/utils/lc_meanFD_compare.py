# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:33:44 2019

@author: lichao
"""
import sys
sys.path.append(r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python')
import pandas as pd
from Statistics.lc_anova import oneway_anova


class CompareMean(object):
    """compare the mean FD value between SZ, BD, MDD and HC using ANOVA
    """

    def __init__(sel):
        sel.sz_id = r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\SZ.xlsx'
        sel.bd_id = r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\BD.xlsx'
        sel.mdd_id = r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\MDD.xlsx'
        sel.hc_id = r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\HC.xlsx'
        sel.meanvalue = r'D:\WorkStation_2018\WorkStation_dynamicFC\Scales\meanFD.xlsx'

        # concat all id
        sel.all_id_path = [sel.sz_id, sel.bd_id, sel.mdd_id, sel.hc_id]

    def _loadexcel(sel):
        sel.id = []
        for idxpath in sel.all_id_path:
            sel.id.append(sel.loadexcel(idxpath))

        sel.meanvalue = pd.read_excel(sel.meanvalue)

    def loadexcel(sel, idxpath):
        idx = pd.read_excel(idxpath, header=None)
        return idx

    def _fetch_intersectionID(sel):
        """ There may be some subject lack mean FD
        So, fetch intersection between id and meanvalue values
        """
        sel.all_meanFD_id = [list(
            set(list(idx.iloc[:, 0])) & set(sel.meanvalue['ID'])) for idx in sel.id]

    def _group_meanvalue(sel):
        """
        return variable for anova
        """
        sel.grouped_meanvalue = [sel.extract_meanvalue_accord_id(
            idx).values for idx in sel.all_meanFD_id]

    def extract_meanvalue_accord_id(sel, idx):
        meanvalue = sel.meanvalue[sel.meanvalue['ID'].isin(idx)].iloc[:, 1:]
        return meanvalue

    def _anova(sel):
        f, p = oneway_anova(*sel.grouped_meanvalue)
        return f, p

    def _anova_abs(sel):
        """
        Given that head motions have direction, 
        but actually there is no need to distinguish direction in statistic analysis.
        So, I perform anova for abselute mean value
        """
        grpmva = [abs(gmv) for gmv in sel.grouped_meanvalue]
        (f, p) = oneway_anova(*grpmva)
        return (f, p)


if __name__ == '__main__':
    sel = CompareMean()
    sel._loadexcel()
    sel._fetch_intersectionID()
    sel._group_meanvalue()
    f, p = sel._anova_abs()
