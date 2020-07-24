# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:07:39 2018

@author: lenovo
"""
# import
from nipype import SelectFiles, Node
# def


def selectFile(rootPath=r'D:\其他\老舅财务\allData'):
    templates = {'path': '*.mat'}

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
