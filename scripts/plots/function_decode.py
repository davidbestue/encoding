# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:25:25 2019

@author: David Bestue
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit.models import LinearModel, LorentzianModel, GaussianModel


root = '/mnt/c/Users/David/Desktop/together_mix_2TR_response_zs5/Conditions/'


#Parameters
presentation_period= 0.35 
presentation_period_cue=  0.50
inter_trial_period= 0.1 
pre_cue_period= 0.5 
pre_stim_period= 0.5 
limit_time=5 

ref_angle=45
dec_thing = 'response'


def ub_wind_path(PATH, system):
    if system=='wind':
        A = PATH                                                                                                    
        B = A.replace('/', os.path.sep)                                                                                               
        C= B.replace('\\mnt\\c\\', 'C:\\') 
    
    if system=='unix':
        C=PATH
    
    ###
    return C



for i_c, CONDITION in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): #
    dfs_visual = []
    dfs_ips = []
    plt.subplot(2,2,i_c+1)
    for SUBJECT_USE_ANALYSIS in ['n001']:  #'n001', 'd001', 'r001', 'b001', 'l001', 's001'
        for brain_region in ["visual", "ips"]:  
            Method_analysis = 'together'
            distance='mix'
            ## Load Results
            Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + brain_region + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
            Matrix_results_name= ub_wind_path(Matrix_results_name, system='wind') 
            xls = pd.ExcelFile(Matrix_results_name)
            sheets = xls.sheet_names
            ##
            if brain_region == 'visual':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                    df_rolled = Matrix_results.iloc[:180, :] ### just the quadrant
                    df_rolled=pd.DataFrame(df_rolled)
                    df_rolled['session'] = sh[-1]
                    dfs_visual.append(df_rolled)
            
            if brain_region == 'ips':
                for sh in sheets:
                    Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                    df_rolled = Matrix_results.iloc[:180, :] 
                    df_rolled=pd.DataFrame(df_rolled)
                    df_rolled['session'] = sh[-1]
                    dfs_ips.append(df_rolled) 





data=df_rolled['2.33'] 
X=data.index.values
Y=np.roll(df_rolled['2.33'], +80)
#Y=data.values

mod = LorentzianModel()
pars = mod.guess(Y, x=X)
out = mod.fit(Y, pars, x=X)
init = mod.eval(pars, x=X)
print(out.fit_report(min_correl=0.25))
Y_lorenz = out.init_fit
plt.plot(X, Y_lorenz, 'k--', label='Lorentzian')

mod2 = GaussianModel()
pars2 = mod2.guess(Y, x=X)
out2 = mod2.fit(Y, pars2, x=X)
init2 = mod2.eval(pars2, x=X)
print(out2.fit_report(min_correl=0.25))
Y_gauss = out2.init_fit
plt.plot(X, Y_gauss, 'y--', label='Gaussian')


plt.plot(X, Y, 'b-', label='raw')
plt.legend()

plt.show(block=False)


dec_angle_lorenz = np.where(max(Y_lorenz) == Y_lorenz)[0][0]/2 
dec_angle_gauss = np.where(max(Y_gauss) == Y_gauss)[0][0]/2 
dec_angle_raw = np.where(max(Y) == Y)[0][0]/2

print(dec_angle_lorenz, dec_angle_gauss, dec_angle_raw)





