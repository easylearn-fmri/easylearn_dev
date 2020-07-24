#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import json

def argparse_(configuration_file):
    with open(configuration_file, 'r', encoding='utf-8') as config:
                configuration = config.read()
    configuration = json.loads(configuration)

    return configuration
    

if __name__ == '__main__':
    configuration = argparse_(configuration_file=r'F:\Python378\Lib\site-packages\eslearn\GUI\test\configuration_file.json')