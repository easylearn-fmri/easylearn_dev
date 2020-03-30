HYDRA

  Section of Biomedical Image Analysis
  Department of Radiology
  University of Pennsylvania
  Richard Building
  3700 Hamilton Walk, 7th Floor
  Philadelphia, PA 19104

  Web:   https://www.med.upenn.edu/sbia/
  Email: sbia-software at uphs.upenn.edu

  Copyright (c) 2018 University of Pennsylvania. All rights reserved.
  See https://www.med.upenn.edu/sbia/software-agreement.html or COPYING file.

Author:
Erdem Varol
software@cbica.upenn.edu

===============
1. INTRODUCTION
===============
This software performs clustering of heterogenous disease patterns within patient group. The clustering is based on seperating the patient imaging features from the control imaging features using a convex polytope classifier. Covariate correction can be performed optionally.


===============
2. TESTING & INSTALLATION
===============

This software has been primarily implemented in MATLAB for Linux operating systems.

----------------
 Requirements
----------------
- Matlab optimization toolbox
- Matlab version >2014


----------------
 Installation
----------------

Hydra can be run directly in a matlab environment without compilation.

OPTIONAL:

If the user wants to run hydra as a standalone executable, then it must be compiled as following (using the additionally obtained matlab compiler "mcc"):

Run the following command in a MATLAB environment:

   mcc -m hydra.m

-----------------
 Test
-----------------
We provided a test sample in the test folder.

To test in matlab enviroment, use the command:

hydra('-i','test.csv','-o','.','-z','test_covar.csv','-k',3,'-f',3)

To test in command line using the compiled executable, use the command:

hydra -i test.csv -o . -z test_covar.csv -k 3 -f 3

This runs a HYDRA experiment which may take a few minutes. The test case contains a subset of a functional MRI study dataset by T. Satterwaithe comprising 100 subjects and their functional ROI's. The output is the clustering labels of the input subjects (only patients are clustered) at varying clustering levels. Also, the clustering stability at varying levels is output to show the rationale for choosing the clustering level.

-----------------
 Test Verification
-----------------

Pre-computed HYDRA results have been included in directory "Pre_computed_test_results". The user may verify that their test results match the pre-computed results to confirm proper set-up. If the clustering occurred properly, ARI for clustering level k=3 should be greater than that of clustering level k=2.

==========
3. USAGE
==========

I. Running "HYDRA":

Here is a brief introduction to running HYDRA. For a complete list of parameters, see --help option.

To run this software, you will need an input csv file, with the following mandatory fields in the following column order:
(Column 1) ID: ID for subject
(Column 2---(last minus 1)) features: features to be used for clustering
(Column (last)) groups: label whether the subject is control (-1) or patient (1)

NOTE: Controls must be strictly -1 and patients must be 1 label. 
NOTE: Label headers names are not strict.

An example input csv file looks as following:
    
ID,        feature_1,    feauture_2,    feature_3,    group
subject_1,    5,        1,        79.3,        -1
subject_2,    10,        1,        71.4,        1
subject_3,    3,        1,        82.7,        -1

Optionally, you can provide a covariate file that will be used to remove covariate effects from imaging features before HYDRA analysis. The covariate file has the following format:
(Column 1) ID: ID for subject
(Column 2---(last)) covariates: covariates of subjects

An example covariate csv file looks as following:

ID,        age,        sex
subject_1,    29,        1
subject_2,    35,        1
subject_3,    51,        0        

If you install the package successfully, there will be two ways of running HYDRA:

1. Running HYDRA in a matlab environment, a simple example:

        hydra('-i','test.csv','-o','.','-z','test_covar.csv','-k',3,'-f',3)

2. Running matlab compiled HYDRA executables in the command line, a simple example:

    hydra -i test.csv -o . -z test_covar.csv -k 3 -f 3


The software returns:


1. HYDRA_results.mat in the specified output directory.
    
This mat file stores the following variables

CIDX - clustering indices for subjects (rows) at varying levels (columns)
ARI - adjusted rand index of clustering at varying levels, clustering level at the highest ARI should be selected
ID - subject ID of rows

===========
4. REFERENCE
===========

If you find this software useful, please cite:

Varol, Erdem, Aristeidis Sotiras, Christos Davatzikos, and Alzheimer's Disease Neuroimaging Initiative. "HYDRA: Revealing heterogeneity of imaging and genetic patterns through a multiple max-margin discriminative analysis framework." NeuroImage 145 (2017): 346-364.

===========
5. LICENSING
===========

  See https://www.med.upenn.edu/sbia/software-agreement.html or COPYING.txt file.

