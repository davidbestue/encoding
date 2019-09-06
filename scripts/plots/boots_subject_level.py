




data = dfsn.loc[(dfsn['condition']=='2_7') & (dfsn['times']==35.025) & (dfsn['region']=='frontinf'), ['decoding', 'subject'] ]


def boots_by_subj(data, col_int, col_subj, n_iterations, stat):
	#### you give a 2 column df, one column qith the value and the other column with subject index:
	list_subjects = data.col_subj.unique()
	sample=[]
	for n in n_itertions:
		resampled=[]
		new_sample = np.random.randint(0, len(list_subjects), len(list_subjects))
		for res_s in new_sample:
			resampled = resampled + list(data.loc[data[col_int]==list_subjects[res_s], col_int].values) 

		sample.append(stat(resample))

	


