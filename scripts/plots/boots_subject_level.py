




data = dfsn.loc[(dfsn['condition']=='2_7') & (dfsn['times']==35.025) & (dfsn['region']=='frontinf'), ['decoding', 'subject'] ]


def boots_by_subj(data, col_int, col_subj, n_iterations, alpha, stat):
    #### you give a 2 column df, one column qith the value and the other column with subject index:
    list_subjects = data[col_subj].unique()
    sample=[]
    for n in range(n_iterations):
        resampled=[]
        new_sample = list(np.random.randint(0, len(list_subjects), len(list_subjects)))
        for res_s in new_sample:
            resampled = resampled + list(data.loc[data[col_subj]==list_subjects[res_s], col_int].values) 
        #
        sample.append(stat(resampled))
    #
    stats_sorted = np.sort(sample)
    new_mean=np.mean(sample)
    return (new_mean, stats_sorted[int((alpha/2.0)*n_iterations)],
            stats_sorted[int((1-alpha/2.0)*n_iterations)])



df_plot = []
for brain_reg in ['visual', 'ips', 'frontinf']:
    for time in list(dfsn.times.unique()) :
        for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:
            data = dfsn.loc[(dfsn['condition']==condition) & (dfsn['times']==time) & (dfsn['region']==brain_reg), ['decoding', 'subject'] ]
            old_mean = data.decoding.mean()
            new_mean, inf_l, sup_l = boots_by_subj(data, 'decoding', 'subject', 1000, 0.05, np.mean)
            df_plot.append( [old_mean, new_mean, inf_l, sup_l, brain_reg, time, condition])





df_plot = pd.DataFrame(df_plot) 
df_plot.columns=[ 'old_mean', 'new_mean', 'inf', 'sup', 'brain_reg', 'time', 'condition' ] #decode compared to shuffle



fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2,2, 1) 
data_cond =  df_plot.loc[ (df_plot['condition']=='1_7')]
sns.lineplot( ax=ax, x="time", y="new_mean", hue='brain_reg', hue_order =  ['visual', 'ips', 'frontinf'], ci=None, palette=pal, data=data_cond) #, 'visual', 'ips',  'frontmid', 'frontsup', 'frontinf'
plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='visual', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='visual', 'sup']) , color=pal[0], alpha=0.3) #, label='target'  ) #plot aprox time of target
plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='ips', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='ips', 'sup']) , color=pal[1], alpha=0.3) #, label='target'  ) #plot aprox time of target
plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='frontinf', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='frontinf', 'sup']) , color=pal[2], alpha=0.3) #, label='target'  ) #plot aprox time of target


plt.show(block=False)






##########################
##########################
##########################





fig = plt.figure(figsize=(10,8))
for indx_c, condition in enumerate(['1_0.2', '1_7', '2_0.2', '2_7']): 
    #features of the plot for the different conditions. Fixed values
    if condition == '1_0.2':
        delay1 = 0.2
        delay2 = 11.8
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
        xlim = [0, 27]
        
    elif condition == '1_7':
        delay1 = 7
        delay2 = 5
        cue=0
        t_p = cue + presentation_period_cue + pre_stim_period 
        d_p = t_p + presentation_period +delay1 
        r_t = d_p + presentation_period + delay2
        xlim = [0, 27]
        
    elif condition == '2_0.2':
        delay1 = 0.2
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2   
        xlim = [0, 27]
        
    elif condition == '2_7':
        delay1 = 7
        delay2 = 12
        cue=0
        d_p = cue + presentation_period_cue + pre_stim_period 
        t_p = d_p + presentation_period +delay1 
        r_t = t_p + presentation_period + delay2
        xlim = [0, 35]
        
    
    start_hrf = 4 #start of the Hemodynamic response (4seconds)
    sec_hdrf = 3 #time it can last
       
    d_p1 = (start_hrf + d_p) ##strat of didtractor (time)
    t_p1 = (start_hrf +t_p) ## strat of target (time)
    r_t1=  (start_hrf + r_t) ## start of response (time)
    #
    d_p2 = d_p1 + sec_hdrf # end of distractor (time)
    t_p2 = t_p1 + sec_hdrf # end of target (time)
    r_t2=  r_t1 + sec_hdrf + resp_time #end of response (time)
    
    y_vl_min = -5 #df_all_by_subj.Decoding.min() #values min and max
    y_vl_max = 5 #◙df_all_by_subj.Decoding.max()
    
    #fig = plt.figure()
    ax = fig.add_subplot(2,2, indx_c+1) 
    #ax = sns.lineplot(x='times', y='decoding',  color = 'black', data=n) #figure to get the intervals of shuffle
    #ax.lines[0].set_linestyle("--")
    data_cond =  df_plot.loc[ (df_plot['condition']==condition)]
    sns.lineplot( ax=ax, x="time", y="new_mean", hue='brain_reg', hue_order =  ['visual', 'ips', 'frontinf'], ci=None, palette=pal, data=data_cond) #, 'visual', 'ips',  'frontmid', 'frontsup', 'frontinf'
    plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='visual', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='visual', 'sup']) , color=pal[0], alpha=0.3) #, label='target'  ) #plot aprox time of target
    plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='ips', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='ips', 'sup']) , color=pal[1], alpha=0.3) #, label='target'  ) #plot aprox time of target
    plt.fill_between(  list(df_plot.time.unique()) , list(data_cond.loc[data_cond['brain_reg']=='frontinf', 'inf']) , list(data_cond.loc[data_cond['brain_reg']=='frontinf', 'sup']) , color=pal[2], alpha=0.3) #, label='target'  ) #plot aprox time of target
    #sns.lineplot( ax=ax, x="times", y="decoding", hue='region', hue_order =  ['visual', 'ips', 'frontinf'],  ci=95, palette=pal, data=dfsn.loc[ (dfsn['condition']==condition)]) #, 'visual', 'ips',  'frontmid', 'frontsup', 'frontinf'
    
    plt.plot([0, 35], [0,0], 'k--')   ## plot chance level (0)
    plt.fill_between(  [ t_p1, t_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='b', alpha=0.3) #, label='target'  ) #plot aprox time of target
    plt.fill_between(  [ d_p1, d_p2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='g', alpha=0.3) #, label='distractor'  ) #plot aprox time of distractor
    plt.fill_between(  [ r_t1, r_t2 ], [y_vl_min, y_vl_min], [y_vl_max, y_vl_max], color='y', alpha=0.3) #, label='response'  )   #plot aprox time of response
    TITLE_BR = condition 
    plt.title(TITLE_BR) #condition title
    plt.gca().spines['right'].set_visible(False) #no right axis
    plt.gca().spines['top'].set_visible(False) #no  top axis
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.xticks([5,15,25]) #just this tcks
    plt.yticks([-4, 0 , 4])
    plt.xlim(xlim)
    if indx_c==3: #legend in just this condition (to avoid repetitions)       
        plt.gca().legend(loc= 2, frameon=False)
        plt.xticks([10, 20 ,30])
        
    else:
        plt.gca().legend(loc= 1, frameon=False).remove()
    


##
plt.suptitle( '', fontsize=18) ## main title
plt.tight_layout(w_pad=5, h_pad=5, rect=[0, 0.03, 1, 0.95]) #correct the space between graphs
plt.show(block=False) #show

















boots_by_subj(data, 'decoding', 'subject', 1000, 0.05, np.mean)

data.decoding.mean() 


def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])





bootstrap(data.decoding, 1000, np.mean, 0.05)

ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2,2, 1) 
sns.lineplot( ax=ax, x="times", y="decoding", hue='region', hue_order =  ['visual', 'ips', 'frontinf'], ci=None, palette=pal, data=dfsn.loc[ (dfsn['condition']=='1_7')]) #, 'visual', 'ips',  'frontmid', 'frontsup', 'frontinf'
plt.show(block=False)


