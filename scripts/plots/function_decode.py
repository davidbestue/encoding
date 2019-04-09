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

results=[]

for i_c, CONDITION in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): #
    #plt.subplot(2,2,i_c+1)
    for SUBJECT_USE_ANALYSIS in ['n001', 'd001', 'r001', 'b001', 'l001', 's001']:  #'n001', 'd001', 'r001', 'b001', 'l001', 's001'
        for brain_region in ["visual", "ips"]:  
            Method_analysis = 'together'
            distance='mix'
            ## Load Results
            Matrix_results_name = root +  CONDITION + '/' + SUBJECT_USE_ANALYSIS + '_' + brain_region + '_'  + CONDITION + '_'  + distance + '_' + Method_analysis + '.xlsx'
            Matrix_results_name= ub_wind_path(Matrix_results_name, system='wind') 
            xls = pd.ExcelFile(Matrix_results_name)
            sheets = xls.sheet_names
            ##
            for sh in sheets:
                Matrix_results = pd.read_excel(Matrix_results_name, sheet_name=sh)  
                df = Matrix_results.iloc[:180, :] ### just the quadrant
                df=pd.DataFrame(df)
                for TR in df.columns:
                    data=df[TR] 
                    X=data.index.values
                    Y=data.values
                    mod = LorentzianModel()
                    #mod =GaussianModel()
                    pars = mod.guess(Y, x=X)
                    out = mod.fit(Y, pars, x=X)
                    Y_lorenz = mod.eval(pars, x=X)
                    #print(out.fit_report(min_correl=0.25))
                    #plt.plot(X, Y_lorenz, 'k--', label='Lorentzian')
                    dec_angle_lorenz =  (90-np.where(Y_lorenz==max(Y_lorenz))[0][0])/2  #out.params['center'].value / 2
                    error =  abs( (90-np.where(Y_lorenz==max(Y_lorenz))[0][0])/2 ) #round(ref_angle - dec_angle_lorenz, 3) 
                    results.append( [error, TR, CONDITION, SUBJECT_USE_ANALYSIS, sh[-1], brain_region])
            
            
            


df = pd.DataFrame(np.array(results)) 
df.columns = ['error', 'TR', 'CONDITION', 'subject', 'session', 'ROI']
df['TR'] = df.TR.astype(float)
df['TR'] = df['TR'] * 2
df['error'] = df.error.astype(float)



pall_chose = "tab10"
linestyles_use='-'
marker_use='o'

plt.figure()
for i_c, CONDITION in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): #
    df_cond = df.loc[df['CONDITION']==CONDITION]
    plt.subplot(2,2,i_c+1)
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1.5}  
    sns.set_context("paper", rc = paper_rc) 
    sns.pointplot(x='TR', y='error', hue='ROI', linestyles = linestyles_use, ci=68, palette = pall_chose, 
                  markers=marker_use, data=df_cond, size=5, aspect=1.5) #     
        
    ### 
    if CONDITION == '1_0.2':
        delay1 = 0.2
        delay2 = 11.8
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
    elif CONDITION == '1_7':
        delay1 = 7
        delay2 = 5
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
    elif CONDITION == '2_0.2':
        delay1 = 0.2
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2    
    elif CONDITION == '2_7':
        delay1 = 7
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2
    
    
    x_bins = len(df_cond.TR.unique()) 
    max_val_x = float(df_cond.TR.max())
    range_hrf = [float(5)/x_bins, float(6)/x_bins] #
    start_hrf = 4
    sec_hdrf = 2
    
    d_p1 = (start_hrf + d_p) * x_bins/ max_val_x
    t_p1 = (start_hrf +t_p)* x_bins/ max_val_x
    r_t1=  (start_hrf + r_t)* x_bins/ max_val_x
    #
    d_p2 = (start_hrf + d_p + sec_hdrf) * x_bins/ max_val_x
    t_p2 = (start_hrf + t_p + sec_hdrf)* x_bins/ max_val_x
    r_t2=  (start_hrf + r_t + sec_hdrf + 4) * x_bins/ max_val_x 
    
    y_vl_min = df_cond.error.min() 
    y_vl_max = df_cond.error.max() 
    
    
    ##all subj visual   
    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3, label='target'  )
    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3, label='distractor'  )
    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3, label='response'  )
    plt.plot([0, x_bins], [0,0], 'k--')
    plt.ylabel('error')
    plt.xlabel('time (s)')
    TITLE_BR = CONDITION 
    plt.legend(frameon=False)
    plt.title(TITLE_BR)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.gca().legend(loc= 0, frameon=False)
    plt.ylim(0,23)






plt.tight_layout()
plt.suptitle( 'Abs error: visual & ips, ' +distance + '_' + Method_analysis, fontsize=12)
plt.show(block=False)





plt.figure()
sns.barplot(x='subject', y='error', hue='CONDITION', palette=['deepskyblue', 'saddlebrown', 'forestgreen','darkorange'], data=df)
plt.show(block=False) 


plt.figure()
sns.barplot(x='CONDITION', y='error', palette=['deepskyblue', 'saddlebrown', 'forestgreen','darkorange'], data=df)
plt.show(block=False) 


plt.figure()
sns.barplot(x='subject', y='error', data=df)
plt.show(block=False) 



##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################

data=df_rolled['2.33'] 
X=data.index.values
#Y=np.roll(df_rolled['2.33'], +80)
Y=data.values

mod = LorentzianModel()
pars = mod.guess(Y, x=X)
out = mod.fit(Y, pars, x=X)
Y_lorenz = mod.eval(pars, x=X)
print(out.fit_report(min_correl=0.25))
plt.plot(X, Y_lorenz, 'k--', label='Lorentzian')
dec_angle_lorenz = round(out.params['center'].value / 2, 3)


mod2 = GaussianModel()
pars2 = mod2.guess(Y, x=X)
out2 = mod2.fit(Y, pars2, x=X)
Y_gauss = mod2.eval(pars2, x=X)
print(out2.fit_report(min_correl=0.25))
plt.plot(X, Y_gauss, 'y--', label='Gaussian')
dec_angle_gauss = round(out2.params['center'].value / 2, 3)

plt.plot(X, Y, 'b-', label='raw')
plt.legend()

plt.show(block=False)




error = ref_angle - dec_angle_lorenz 






