# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:05:33 2019

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


presentation_period= 0.35 #stim presnetation time
presentation_period_cue=  0.50 #presentation of attentional cue time
pre_stim_period= 0.5 #time between cue and stim
resp_time = 4  #time the response is active



def tiemcourse_by_subject_permtest(df_x, condition, title_plot,  ylims=[-20,20], decoding_thing='target'):
    ##
    ###
    ####   In the input dataframe you need the following columns:
    ###  'new_mean', 'inf', 'sup', 'brain_reg', 'time', 'condition'
    ## 
    pal = ['darkblue',  'darkorange',  'darkgreen'] #sns.color_palette("tab10", n_colors=12, desat=1).as_hex()[0:3]
    ##
    fig = plt.figure(figsize=(6,15))
    for indx_P, Plot in enumerate(['visual', 'ips', 'pfc']): 
        #features of the plot for the different conditions. Fixed values
        if condition == '1_0.2':
            condition_title = '' #'o:1, d:0.2'
            y_label_cond = 'decoding ' + decoding_thing 
            x_label_cond = 'time (s)'
            delay1 = 0.2
            delay2 = 11.8
            cue=0
            t_p = cue + presentation_period_cue + pre_stim_period 
            d_p = t_p + presentation_period +delay1 
            r_t = d_p + presentation_period + delay2
            xlim = [1, 25]

        elif condition == '1_7':
            condition_title = '' #'o:1, d:7'
            y_label_cond = 'decoding '  + decoding_thing 
            x_label_cond = 'time (s)' 
            delay1 = 7
            delay2 = 5
            cue=0
            t_p = cue + presentation_period_cue + pre_stim_period 
            d_p = t_p + presentation_period +delay1 
            r_t = d_p + presentation_period + delay2
            xlim = [1, 25]

        elif condition == '2_0.2':
            condition_title = '' #'o:2, d:0.2'
            y_label_cond = 'decoding ' + decoding_thing 
            x_label_cond = 'time (s)'        
            delay1 = 0.2
            delay2 = 12
            cue=0
            d_p = cue + presentation_period_cue + pre_stim_period 
            t_p = d_p + presentation_period +delay1 
            r_t = t_p + presentation_period + delay2   
            xlim = [1, 25]

        elif condition == '2_7':
            condition_title = '' #'o:2, d:7'
            y_label_cond = 'decoding '  + decoding_thing 
            x_label_cond = 'time (s)' 
            delay1 = 7
            delay2 = 12
            cue=0
            d_p = cue + presentation_period_cue + pre_stim_period 
            t_p = d_p + presentation_period +delay1 
            r_t = t_p + presentation_period + delay2
            xlim = [1, 30]


        start_hrf = 4 #start of the Hemodynamic response (4seconds)
        sec_hdrf = 3 #time it can last

        d_p1 = (start_hrf + d_p) ##strat of didtractor (time)
        t_p1 = (start_hrf +t_p) ## strat of target (time)
        r_t1=  (start_hrf + r_t) ## start of response (time)
        #
        d_p2 = d_p1 + sec_hdrf # end of distractor (time)
        t_p2 = t_p1 + sec_hdrf # end of target (time)
        r_t2=  r_t1 + sec_hdrf + resp_time #end of response (time)

        y_vl_min = -10 #df_all_by_subj.Decoding.min() #values min and max
        y_vl_max = 10 #◙df_all_by_subj.Decoding.max()

        #fig = plt.figure()
        ax = fig.add_subplot(3,1, indx_P+1) 

        if Plot=='visual':
            df_plot = df_x.loc[(df_x['condition']==condition) & (df_x['region']==Plot) ]
            #
            sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkblue', data=df_plot)
            #
            for subj in ['d001', 'n001', 'b001', 'r001', 's001', 'l001']:
                df_plot_s = df_plot.loc[df_plot['subject']==subj]
                sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkblue', alpha=0.5, linewidth=1, data=df_plot_s)
            #
            #
        elif Plot == 'ips':
            df_plot = df_x.loc[(df_x['condition']==condition) & (df_x['region']==Plot) ]
            #
            sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkorange', data=df_plot)
            #
            for subj in ['d001', 'n001', 'b001', 'r001', 's001', 'l001']:
                df_plot_s = df_plot.loc[df_plot['subject']==subj]
                sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkorange', alpha=0.5,  linewidth=1, data=df_plot_s)
            #
            #
        elif Plot =='pfc':
            df_plot = df_x.loc[(df_x['condition']==condition) & (df_x['region']==Plot) ]
            #
            sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkgreen', data=df_plot)
            #
            for subj in ['d001', 'n001', 'b001', 'r001', 's001', 'l001']:
                df_plot_s = df_plot.loc[df_plot['subject']==subj]
                sns.lineplot( ax=ax, x="times", y="decoding",  ci=68, color='darkgreen', alpha=0.5,  linewidth=1, data=df_plot_s)
            #
            #

        ####  Significance
        for idx_time, time in enumerate(df_plot.times.unique()):
            sign_ = []
            for Subj in df_plot.subject.unique():
                dec_subject = df_plot.loc[(df_plot['times']==time) & (df_plot['subject']==Subj) , 'decoding'].values
                pval_subject = df_plot.loc[(df_plot['times']==time) & (df_plot['subject']==Subj), 'pval'].values
                if dec_subject>0:
                    if pval_subject<0.05:
                        sign_.append(1)
                        
            #
            sign_sum = sum(sign_)
            #print(sign_sum)
            plt.plot(time, 10 , marker = 'o', color=pal[indx_P],  markersize=sign_sum*1.5 )
        #
        #
        ####
        plt.plot([0, 35], [0,0], 'k--', linewidth=1)   ## plot chance level (0)
        plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='grey', alpha=0.3) #, label='target'  ) #plot aprox time of target
        plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='darkred', alpha=0.3) #, label='distractor'  ) #plot aprox time of distractor
        plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3) #, label='response'  )   #plot aprox time of response
        #
        TITLE_BR = condition_title 
        plt.title(TITLE_BR, fontsize=20) #condition title
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.xticks([5,10,15,20,25,30,35], fontsize=15) #just this tcks
        plt.ylim(ylims[0], ylims[1])
        if ylims[0]==-20:
            plt.yticks([-20, -10, 0 , 10, 20], fontsize=15)
        if ylims[0]==-30:
            plt.yticks([-30, -15, 0 , 15, 30], fontsize=15)
        #
        plt.xlim(xlim)
        plt.xlabel(x_label_cond, fontsize=20)
        plt.ylabel(y_label_cond, fontsize=20)
        if indx_P==2: #legend in just this condition (to avoid repetitions)       
            plt.gca().legend(loc=1, frameon=False, bbox_to_anchor=(1.1, 0.45), fontsize=15)
            #plt.xticks([5,10, 15, 20, 25,30,35], fontsize=15)
            plt.xticks([5,10, 15, 20, 25,30], fontsize=15)

        else:
            plt.gca().legend(loc= 1, frameon=False).remove()



    ##
    plt.suptitle( title_plot, fontsize=25) ## main title
    plt.tight_layout(w_pad=1, h_pad=1, rect=[0, 0.03, 1, 0.95]) #correct the space between graphs
    plt.show(block=False) #show
