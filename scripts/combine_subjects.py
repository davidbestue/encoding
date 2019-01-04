# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:56:47 2019

@author: David
"""


root = '/home/david/Desktop/KAROLINSKA/bysess_mix_2TR/Conditions/'

dfs_visual = {}
dfs_ips = {}


for SUBJECT_USE_ANALYSIS in ['d001', 'n001', 'r001', 'b001', 'l001', 's001']:
    for algorithm in ["visual", "ips"]:  
        Method_analysis = 'bysess'
        distance='mix'
        CONDITION = '1_0.2' #'1_0.2', '1_7', '2_0.2', '2_7'
        
        ## Load Results
        Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + algorithm + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
        xls = pd.ExcelFile(Matrix_results_name)
        sheets = xls.sheet_names
        ##
        if algorithm == 'visual':
            for sh in sheets:
                Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)            
                dfs_visual[ SUBJECT_USE_ANALYSIS + '_' + sh] = Matrix_results
        
        if algorithm == 'ips':
            for sh in sheets:
                Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)            
                dfs_ips[ SUBJECT_USE_ANALYSIS + '_' + sh] = Matrix_results




#####
#####
panel_v=pd.Panel(dfs_visual)
df_visual=panel_v.mean(axis=0)

panel_i=pd.Panel(dfs_ips)
df_ips=panel_i.mean(axis=0)
            

