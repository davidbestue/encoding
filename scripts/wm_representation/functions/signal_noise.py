# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:23:51 2019

@author: David
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### Load reconstruction and take the interesting part
xls = pd.ExcelFile('/home/david/Desktop/Reconstructions_n001_LM.xlsx')
sheets = xls.sheet_names
##
R={}
for sh in sheets:
    R[sh]  = pd.read_excel(xls, sheet_name=sh)

Decoding_df =[]

for dataframes in R.keys():
    df = R[dataframes]
    a = pd.DataFrame(df.iloc[360,:])
    a = a.reset_index()
    a.columns = ['times', 'decoding']
    a['times']=a['times'].astype(float)
    a['region'] = dataframes.split('_')[1]
    a['subject'] = dataframes.split('_')[0]
    a['condition'] = dataframes.split('_')[-2] + '_' + dataframes.split('_')[-1] 
    Decoding_df.append(a)



## Load the shuffle (it already has the interesting part)
Df = pd.concat(Decoding_df)
Df['label'] = 'signal'

Df_shuff = pd.read_excel('/home/david/Desktop/Reconstructions_n001_LM_shuff.xlsx')
Df_shuff['label'] = 'shuffle'


##combine them
df = pd.concat([Df, Df_shuff])


plt.figure()
ax = sns.lineplot(x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']=='1_7') & (df['subject'] =='n001')  & (df['region'] =='visual')]) 
plt.show(block=False)

plt.figure()
ax = sns.lineplot(x="times", y="decoding", hue='label', hue_order = ['signal', 'shuffle'],  data=df.loc[ (df['condition']=='1_7') & (df['subject'] =='n001')  & (df['region'] =='ips')]) 
plt.show(block=False)
