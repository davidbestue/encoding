

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from scipy.stats import wilcoxon

sns.set_context("poster", font_scale=1.1)
sns.set_style("ticks")



def heatmap__1_02(data, title, max_=1.):
    data=data.iloc[0:12, 0:12]
    ax = sns.heatmap(data,vmin=0., vmax=max_, cmap= 'viridis',
                    cbar_kws={"shrink": .82, 'ticks' : [0,round(max_/3,2), round(2*max_/3, 2) ,max_], 
                              'label': 'decoding strength'}) ##sns.cm.rocket_r
    #ax = sns.heatmap(data, cmap= 'viridis') ##sns.cm.rocket_r
    ax.invert_yaxis()
    ax.figure.axes[-1].yaxis.label.set_size(15)
    plt.gca().set_ylabel('test (TRs)')
    plt.gca().set_xlabel('train delay o=1 d=7')
    plt.gca().set_title(title)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([0,4,8,12])
    plt.gca().set_yticklabels([0,4,8,12])
    #sns.lineplot(x=[4.0,4.0001], y=[0,4], color='r', linewidth=12)
    #### lines
    presentation_period= 0.35 #stim presnetation time
    presentation_period_cue=  0.50 #presentation of attentional cue time
    pre_stim_period= 0.5 #time between cue and stim
    resp_time = 4  #time the response is active
    start_hrf = 4 #start of the Hemodynamic response (4seconds)
    sec_hdrf = 3
    delay1 = 0.2
    delay2 = 11.8
    cue=0
    t_p = cue + presentation_period_cue + pre_stim_period + start_hrf
    d_p = t_p + presentation_period +delay1
    r_t = d_p + presentation_period + delay2 
    #
    t_p_st = t_p/2.335    
    d_p_st = d_p/2.335
    r_t_st = r_t/2.335
    #t_p_en = (t_p+sec_hdrf) /2.335
    ##stim
    sns.lineplot(x=[0, 1], y=[t_p_st,t_p_st], color='b', linewidth=4) # añadir 0.001 o no se ve
    ##Distractor
    sns.lineplot(x=[0, 1], y=[d_p_st,d_p_st], color='g', linewidth=4) 
    ##Response
    sns.lineplot(x=[0, 1], y=[r_t_st, r_t_st], color='y', linewidth=4) 
    plt.show(block=False)