# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:01:44 2018
used for select target files
refer to and thank [https://www.jianshu.com/p/91453c76dbc3]
@author: Li Chao
"""
from nipype import SelectFiles, Node


def selectFile(
        rootPath=r'I:\Data_Code\insomnia\workstation_MVPA_2018_05\FunImgARW1'):
    templates = {'path': '*\\sw*.nii'}

    # Create SelectFiles node
    sf = Node(SelectFiles(templates),
              name='selectfiles')

    # Location of the dataset folder
    sf.inputs.base_directory = rootPath

    # Feed {}-based placeholder strings with values
#    sf.inputs.subject_id1 = '00[1,2]'
#    sf.inputs.subject_id2 = '01'
#    sf.inputs.ses_name = "retest"
#    sf.inputs.task_name = 'covertverb'
    path = sf.run().outputs.__dict__['path']
    return path
