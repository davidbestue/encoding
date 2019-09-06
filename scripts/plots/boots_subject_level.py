




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





boots_by_subj(data, 'decoding', 'subject', 1000, 0.05, np.mean)

data.decoding.mean() 




n_iterations=10

list_subjects = data[col_subj].unique()
sample=[]
for n in range(n_iterations:
    resampled=[]
    new_sample = list(np.random.randint(0, len(list_subjects), len(list_subjects)))
    for res_s in new_sample:
        resampled = resampled + list(data.loc[data[col_subj]==list_subjects[res_s], col_int].values) 
    #
    sample.append(stat(resample))
#
stats_sorted = np.sort(sample)
return (stats_sorted[int((alpha/2.0)*n_iterations)],
        stats_sorted[int((1-alpha/2.0)*n_iterations)])