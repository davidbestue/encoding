# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:47 2019

@author: David
"""




for SUBJECT_USE_ANALYSIS in ['d001', 'n001', 'r001', 'b001', 'l001', 's001']:
    for algorithm in ["visual", "ips"]:  
        Method_analysis = 'bysess'
        CONDITION = '1_0.2' #'1_0.2', '1_7', '2_0.2', '2_7'
        
        ## Load Results
        Matrix_weights_name = SUBJECT_USE_ANALYSIS + '_' + algorithm + '_' + Method_analysis + '_matrix.xlsx'
        Matrix_weights = pd.read_excel(Matrix_weights_name, sheet_name=session_enc)