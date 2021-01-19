# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:54:28 2021

@author: Li Chao
Email: lichao19870617@163.com
"""

import re
import requests
r = requests.get('https://github.com/easylearn-fmri/easylearn_dev/blob/dev/eslearn_news.txt')
text = r.text

s = "__version__ = (\d+.\d+.\d+)##endLabel##"
pattern = re.compile(s, re.I)  # s, I表示忽略大小写
version = pattern.findall(text)
