# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:31:06 2019

@author: David
"""



root ='/home/david/Desktop/IEM_data/'

Beh_WM_files_sess=[root +'r001/WMtask/s08/r01/wm_beh.txt', root +'r001/WMtask/s08/r02/wm_beh.txt', root +'r001/WMtask/s08/r03/wm_beh.txt', root +'r001/WMtask/s08/r04/wm_beh.txt', root +'r001/WMtask/s08/r05/wm_beh.txt', root +'r001/WMtask/s08/r06/wm_beh.txt',
                    root +'r001/WMtask/s09/r01/wm_beh.txt', root +'r001/WMtask/s09/r02/wm_beh.txt', root +'r001/WMtask/s09/r03/wm_beh.txt', root +'r001/WMtask/s09/r04/wm_beh.txt',
                    root +'r001/WMtask/s10/r01/wm_beh.txt']

headers_col = ['type', 'delay1', 'delay2', 'T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2', 'distance_T_dist', 'cue', 'order',
                                'orient', 'horiz_vertical', 'A_R', 'A_err', 'Abs_angle_error', 'Error_interference', 'A_DC', 'A_DC_dist', 'Q_DC', 
                                'A_DF', 'A_DF_dist', 'Q_DF', 'A_DVF', 'Q_DVF', 'A_DVF_dist', 'Q_DVF_dist', 'presentation_att_cue_time', 'presentation_target_time',
                                'presentation_dist_time', 'presentation_probe_time', 'R_T', 'trial_time', 'disp_time']  
                
                



frames=[]
for i in range(0, len(Beh_WM_files_sess)):
    #Open file
    Beh_WM_files_path = Beh_WM_files_sess[i]
    behaviour=genfromtxt(Beh_WM_files_path, skip_header=1)
    Beh = pd.DataFrame(behaviour) 
    frames.append(Beh)


df = pd.concat(frames)
df.columns = headers_col



new_t = []
new_NT1 = []
new_NT2 = []
new_dt = []
new_dt1 =[]
new_dt2 = []

def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    if op1>360:
        op1=op1-360
    #angs=[a1,a2]
    #op2=min(angs)+(360-max(angs))
    #options=[op1,op2]
    return op1


for i in range(0, len(df)):
    if df['T'].iloc[i] >45:
        dist_roll = - circ_dist(df['T'].iloc[i], 45)
    else:
        dist_roll = circ_dist(df['T'].iloc[i], 45)
    
    
    new_t.append( df['T'].iloc[i] + dist_roll)
    
    #♠NT1
    t1 = df['NT1'].iloc[i]  + dist_roll
    if t1 > 360:
        t1 = t2 -360
    if t1<0:
        t1= 360+t1
    
    new_NT1.append(t1)
    
    ##NT2
    t2 = df['NT2'].iloc[i]  + dist_roll
    if t2 > 360:
        t2 = t2 -360
    if t2<0:
        t2= 360+t2
    
    new_NT2.append(t2)
    
    ##distractor
    dt = df['Dist'].iloc[i]  + dist_roll
    if dt > 360:
        dt = dt -360
    if dt<0:
        dt= 360+dt
    
    new_dt.append(dt)
    
    ##distractor 1
    dt1 = df['Dist_NT1'].iloc[i]  + dist_roll
    if dt1 > 360:
        dt1 = dt1 -360
    if dt1<0:
        dt1= 360+dt1
    
    new_dt1.append(dt1)
    
    ##distractor 2
    dt2 = df['Dist_NT2'].iloc[i]  + dist_roll
    if dt2 > 360:
        dt2 = dt2 -360
    if dt2<0:
        dt2= 360+dt2
    
    new_dt2.append(dt2)


    

df['nT'] = new_t
df['nNT1'] = new_NT1
df['nNT2'] = new_NT2
df['nd'] = new_dt
df['nd1'] = new_dt1
df['nd2'] = new_dt2







targets_all = []

targets_all.append(df['nT'].values)
targets_all.append(df['nNT1'].values)
targets_all.append(df['nNT2'].values)
#targets_all.append(df['nd'].values)
#targets_all.append(df['nd1'].values)
#targets_all.append(df['nd2'].values)




import itertools
list_t =  list(itertools.chain.from_iterable(targets_all))

sns.distplot(list_t, bins=25, color='darkblue', kde=False)
plt.title('T - NT rotation')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.tight_layout()
plt.show(block=False)
plt.show()



