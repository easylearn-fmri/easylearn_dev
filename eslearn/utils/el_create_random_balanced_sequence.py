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
    This class is used to create random and balanced sequences for k stimulations
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
            
        initial_rand_seed: int
            initial random seed
            
        tolerance: int
            How many iteration if you can not get accept results, given that available keys is not in key_all_for_picking

    return
    -----------
        random_squence: numpy array
            created random and balanced sequence
    """

    def __init__(sel, data={1:np.arange(0, 60, 1), 2:np.arange(60, 90, 1), 3:np.arange(90, 120)}, 
                 s=3, rt=50, initial_rand_seed=5, tolerance=1000):
        # Debug: 生成三个类别，1，2，3。第一个类别有60个不同的具体刺激，后两者分别有30个。
        sel.data = data 
        sel.s = s
        sel.rt = rt
        sel.initial_rand_seed = initial_rand_seed 
        sel.tolerance = tolerance
        # ---------
        sel.category_of_stimutations = list(data.keys())
        sel.n = len(sel.category_of_stimutations) * sel.rt  # totle sequence length
        np.random.seed(sel.initial_rand_seed)
        
    def main(sel):
        for tol in range(sel.tolerance):
            sel.key_to_store = np.arange(-sel.s, 0, 1)  # 避免在内部循环中判断 i >= sel.s, 节省时间。切记完成循环后删除前s个元素。
            sel.random_squence = np.zeros([sel.n + sel.s, ])  # 切记完成循环后删除前s个元素。
            sel.key_all_for_picking = np.repeat(sel.category_of_stimutations, sel.rt, 0)
        
            for i in np.arange(sel.s, sel.n + sel.s, 1):
                # 判断 np.arange(-1, -sel.s, -1)个数是否连续: 即n个数前面的sel.s - 1 个数是否相同，相同则第n个数不能再重复。
                # 否
                if len(np.unique(sel.key_to_store[np.arange(-1, -sel.s, -1)])) > 1:
                    id_selected = np.random.randint(len(sel.key_all_for_picking))
                    key_choose = sel.key_all_for_picking[id_selected]
                    sel.key_all_for_picking = np.delete(sel.key_all_for_picking, id_selected)  
                    sel.key_to_store = np.append(sel.key_to_store, key_choose)
                    sel.random_squence[i] = sel.pick_one_stimulation(key_choose)
                    
                # 是
                else:
                    key_available_pick_point = np.array(list(set(sel.category_of_stimutations) - set(sel.key_to_store[[-1, -1]])))
                    # Chose those keys that exist in sel.key_all_for_picking
                    is_available_keys = np.isin(key_available_pick_point , sel.key_all_for_picking)
                    # If have at least one availabel key in sel.key_all_for_picking
                    if any(is_available_keys):
                        key_available_pick_point = key_available_pick_point[is_available_keys]
                        key_choose = key_available_pick_point[np.random.randint(len(key_available_pick_point))]
                        loc_bool = np.where(sel.key_all_for_picking == key_choose)[0]
                        id_selected = loc_bool[np.random.randint(len(loc_bool))]
                        sel.key_all_for_picking = np.delete(sel.key_all_for_picking, id_selected)  
                        sel.key_to_store = np.append(sel.key_to_store, key_choose)
                        sel.random_squence[i] = sel.pick_one_stimulation(key_choose)
                        
                    # If have no availabel key in sel.key_all_for_picking, then go to next tol and rand_seed.
                    else:
                        break

            else:  # for-else pair: All iteration reached: succeed
            # Delete the first sel.s items
                return (np.delete(sel.random_squence, np.arange(0, sel.s, 1), axis=0), 
                        np.delete(sel.key_to_store, np.arange(0, sel.s, 1), axis=0), 
                        sel.initial_rand_seed)
                
            # Not all iteration reached: failed
            print(f'Failed!\nThe {tol}th Try...')
            sel.rand_seed = tol
            np.random.seed(sel.rand_seed)
            continue
            


    def pick_one_stimulation(sel, key_choose):
        """
        Pick one stimulation from the one stimulation category.
        """
        return sel.data[key_choose][np.random.randint(len(sel.data[key_choose]))]


if __name__ == "__main__":
    st = time.time()
    sel = CreateSequence()
    rand_sequ, category_of_stimutations, rand_seed = sel.main()
    et = time.time()
    print(f"Running time = {et - st}")
    print(f"Repeat times for each category of stimutation is {sel.rt}")
    print(f"Stimutation category are {sel.category_of_stimutations}")
    print("--"*20)
    for i in np.unique(category_of_stimutations):
        print(f"Number of created {i}th category = {np.sum(category_of_stimutations == i)} ")
        
        
