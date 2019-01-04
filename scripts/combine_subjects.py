# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:47 2019

@author: David
"""


root = '/home/david/Desktop/KAROLINSKA/bysess_mix_2TR/Conditions/'


xls = pd.ExcelFile(Matrix_results_name)
# Now you can list all sheets in the file
xls.sheet_names


for SUBJECT_USE_ANALYSIS in ['d001', 'n001', 'r001', 'b001', 'l001', 's001']:
    for algorithm in ["visual", "ips"]:  
        Method_analysis = 'bysess'
        distance='mix'
        CONDITION = '1_0.2' #'1_0.2', '1_7', '2_0.2', '2_7'
        
        ## Load Results
        Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + algorithm + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
        Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=0)