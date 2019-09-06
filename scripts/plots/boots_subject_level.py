




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


