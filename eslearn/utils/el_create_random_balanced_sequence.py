# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 07:50:18 2020

@author: Chao Li
Email: lichao19870617@gmail.com
"""

import numpy as np
import time
class CreateSequence(object):
    """
    This class is used to create random and balanced sequences for k stimulation
    This task must meet the following three conditions:
        1. There are no s continuous stimulus types.
        2. Each types of stimulation has the same number of stimulation.
        3. All stimulation occurs randomly.


    Parameters:
    -----------
        data: dict, 
            Keys are stimulation label (int); items are stimulation id for each stimulation label (array).
            
        s: int
            Continuous upper limit
            
        rt: int
        
            Repeat times for each category of stimulation.
            
        rand_seed: int
            random seed

    return
    -----------
        random_squence: numpy array
            created random and balanced sequence
    """

    def __init__(sel, data={1:np.arange(0, 60, 1), 2:np.arange(60, 90, 1), 3:np.arange(90, 120)}, s=5, rt=30, rand_seed=0):
        # Debug: 生成三个类别，1，2，3。第一个类别有60个不同的具体刺激，后两者分别有30个。
        sel.data = data 
        sel.s = s
        sel.rt = rt
        sel.rand_seed = rand_seed 
        # ---------
        sel.category_of_created_stimutations = list(data.keys())
        sel.n = len(sel.category_of_created_stimutations) * sel.rt  # totle sequence length
        sel.key_choose_all = np.repeat(sel.category_of_created_stimutations, sel.rt, 0)
        sel.key_to_store = np.arange(-sel.s, 0, 1)  # 避免在循环中判断i>=sel.s, 节省时间。切记完成循环后删除前s个元素。
        sel.random_squence = np.zeros([sel.n + sel.s, ])  # 切记完成循环后删除前s个元素。
        np.random.seed(sel.rand_seed)
        
    def main(sel):
        for i in np.arange(sel.s, sel.n + sel.s, 1):
            # 判断 np.arange(-1, -sel.s, -1)个数是否连续: 即n个数前面的sel.s - 1 个数是否相同，相同则第n个数不能再重复。
            # 否
            if len(np.unique(sel.key_to_store[np.arange(-1, -sel.s, -1)])) > 1:
                id_selected = np.random.randint(len(sel.key_choose_all))
                key_choose = sel.key_choose_all[id_selected]
                sel.key_choose_all = np.delete(sel.key_choose_all, id_selected)  
                sel.key_to_store = np.append(sel.key_to_store, key_choose)
                sel.random_squence[i] = sel.pick_one_stimulation(key_choose)
                
            # 是
            else:
                key_available_pick_point = list(set(sel.category_of_created_stimutations) - set(sel.key_to_store[[-1, -1]]))
                key_choose = key_available_pick_point[np.random.randint(len(key_available_pick_point))]
                
                loc_bool = np.where(np.isin(sel.key_choose_all == key_choose, sel.key_choose_all))[0]
                id_selected = loc_bool[np.random.randint(len(loc_bool))]
                sel.key_choose_all = np.delete(sel.key_choose_all, id_selected)  

                sel.key_to_store = np.append(sel.key_to_store, key_choose)
                sel.random_squence[i] = sel.pick_one_stimulation(key_choose)
        
        # 删除前s个
        return np.delete(sel.random_squence, np.arange(0, sel.s, 1), axis=0), np.delete(sel.key_to_store, np.arange(0, sel.s, 1), axis=0)

    def pick_one_stimulation(sel, key_choose):
        """
        Pick one stimulation from the one stimulation category.
        """
        return sel.data[key_choose][np.random.randint(len(sel.data[key_choose]))]


if __name__ == "__main__":
    st = time.time()
    sel = CreateSequence()
    rand_sequ, category_of_created_stimutations = sel.main()
    et = time.time()
    print(et - st)
    print(f"Repeat times for each category of stimutation is {sel.rt}")
    print(f"Stimutation category are {sel.category_of_created_stimutations}")
    print("--"*20)
    for i in np.unique(category_of_created_stimutations):
        print(f"Number of created {i}th category = {np.sum(category_of_created_stimutations == i)} ")
        
        
