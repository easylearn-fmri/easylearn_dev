#-*- coding:utf-8 -*-

"""
Created on 2020/02/29
------
@author: LI Chao
Email: lichao19870617@gmail.com or lichao19870617@163.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='easylearn',
    version='1.1.1.20200229_alpha',
    description=(
        'This project is mainly used for machine learning in resting-state fMRI field'
    ),
    long_description=long_description,
    author='Chao Li',
    author_email='lichao19870617@gmail.com',
    maintainer='Chao Li +',
    maintainer_email='lichao19870617@gmail.com',
    license='MIT License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/lichao312214129/lc_rsfmri_tools_python',
    classifiers=[
        'Development Status :: 1.1.1.20200229_alpha',
        'Intended Audience :: Researcher/Student',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Resting-state fMRI',
        'Topic :: Neuroimaging',
        'Topic :: Machine learning',
    ],

    install_requires=[
        'sklearn',
        'skrebate >= 0.6',
        'numpy >= 1.17.4',
        'pandas',
        'nibabel==3.0.0',
        'nilearn==0.6.0',
        'imblearn',
        'collections',
        'sys',
    ]
)