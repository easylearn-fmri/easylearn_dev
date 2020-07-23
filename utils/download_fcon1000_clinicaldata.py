# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:47:55 2019
# download clinical info of fcon1000
# Author: Li Chao 
"""

import os
import urllib


def get_sitename():
	"""
	get all site name
	"""
	root_dir = r'F:\Data\fcon_1000'
	sitename = os.listdir(root_dir)
    return sitename

def get_url(sitename):
	"""
	get all url neet to download
	"""
	root_path = 'https://fcp-indi.s3.amazonaws.com/data/Projects/FCON1000/'
	file_url = [''.join(root_path + one_sitename + '/participants.tsv') for one_sitename in sitename]
    return file_url

def download_info(file_url):
    '''
    down load file to each folder
    '''
    root_dir = r'F:\Data\fcon_1000'
    
    nf = len(file_url)
    for i, fu in enumerate(file_url):
        print(f'downloading {i+1}/{nf}\n')
        save_file = os.path.basename(os.path.dirname(fu))
        save_file = os.path.join(root_dir, save_file,'participants.tsv')
        
        if os.path.exists(save_file):
            continue
        else:
            try:
                urllib.request.urlretrieve(fu, save_file)
            except:
                print(f'no {fu}')
                continue

	# Print all done
	print ('Done!')


# Make module executable
if __name__ == '__main__':
	pass
