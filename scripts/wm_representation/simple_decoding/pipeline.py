# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:32:56 2019
@author: David Bestue
"""

import sys
functions_path = os.path.join( os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'functions')  # go one directory back and join with functions
sys.path.append(functions_path) ## add the diectory to the path, so now you can import from that (see what is inside with sys.path 
from data_to_use import *
from process_encoding import *
from training_fucntion import *
from process_wm import *



def circdist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)



def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    #return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return abs( np.rad2deg(ang1-ang2))



def test_wm(testing_activity, testing_behaviour):
    df=[]
    testing_angles = np.array(testing_behaviour['T'])
    for scan_s in range(nscans_wm):
        for trial_n in range(len(testing_angles)):
            test_interc = [1] + list(testing_activity[trial_n, scan_s, :])
            x,y = weights.predict(test_interc)[0]
            y_real =np.sin(np.radians(testing_angles[trial_n]) )
            x_real = np.cos(np.radians(testing_angles[trial_n]) )
            error = angle_between( (x,y), (x_real, y_real))
            time = scan_s * TR
            target = testing_behaviour['T'].iloc[trial_n]
            response= testing_behaviour['A_R'].iloc[trial_n]
            df.append( [ error, Subject, Brain_region, time, trial_n, condition, target, response ])
    #
    df=pd.DataFrame(df)
    df.columns=['error', 'Subject', 'Brain_region', 'time', 'trial', 'condition', 'target', 'response']
    return df



def train_each_vxl( training_dataset, training_targets ):
    #
    ### X Training
    ## X matrix (intercept and spikes)
    X = np.column_stack([np.ones(np.shape(training_dataset)[0]),  training_dataset])
    ## Y (sinus and cos of the target)
    sinus =np.sin([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    cosinus = np.cos([np.radians(np.array(training_targets)[i]) for i in range(0, len(training_targets))])
    Y = np.column_stack([cosinus, sinus])
    Y = Y.astype(float) #to make it work in the cluster
    X = X.astype(float)
    model = sm.OLS(Y, X)
    ##train the model
    weights = model.fit()  ## training_weights.params
    return weights



nscans_wm = 18


cond_res = []
for Subject in ['n001', 'd001', 'r001', 's001', 'b001', 'l001']:
    print(Subject)
    for Brain_region in ['visual', 'ips', 'frontinf']:
        ### Data to use
        enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, 'together', Brain_region)
        ##### Process training data
        training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=6, TR=2.335) #4
        ##### Train your weigths
        weights = train_each_vxl( training_dataset, training_targets )
        for condition in ['1_0.2', '1_7', '2_0.2', '2_7']:
            ##### Process testing data
            testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=condition, distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
            results = test_wm(testing_activity, testing_behaviour)
            cond_res.append(results)







df = pd.concat(cond_res, ignore_index=True)
#### df.to_excel('liniar_decoding_results.xlsx')




presentation_period= 0.35 #stim presnetation time
presentation_period_cue=  0.50 #presentation of attentional cue time
pre_stim_period= 0.5 #time between cue and stim
resp_time = 4  #time the response is active

pal = sns.color_palette("tab10", n_colors=12, desat=1).as_hex()[0:3]


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
    
    y_vl_min = 75 #df_all_by_subj.Decoding.min() #values min and max
    y_vl_max = 150 #◙df_all_by_subj.Decoding.max()
    
    #fig = plt.figure()
    ax = fig.add_subplot(2,2, indx_c+1) 
    #ax = sns.lineplot(x='times', y='decoding',  color = 'black', data=n) #figure to get the intervals of shuffle
    #ax.lines[0].set_linestyle("--")
    sns.lineplot( ax=ax, x="time", y="error", hue='Brain_region', hue_order =  ['visual', 'ips', 'frontinf'],  ci=95, palette=pal, data=df.loc[ (df['condition']==condition)]) #, 'visual', 'ips',  'frontmid', 'frontsup', 'frontinf'
    
    #plt.plot([0, 35], [0,0], 'k--')   ## plot chance level (0)
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
    plt.yticks([60,100,140])
    plt.xlim(xlim)
    plt.ylim(50,150)
    if indx_c==3: #legend in just this condition (to avoid repetitions)       
        plt.gca().legend(loc= 2, frameon=False)
        plt.xticks([10, 20 ,30])
        
    else:
        plt.gca().legend(loc= 1, frameon=False).remove()
    


##
plt.suptitle( '', fontsize=18) ## main title
plt.tight_layout(w_pad=5, h_pad=5, rect=[0, 0.03, 1, 0.95]) #correct the space between graphs
plt.show(block=False) #show






#### best and worst subject

plt.figure()
sns.factorplot(x='Subject', y='error', kind='bar', data=df)
plt.ylim(110, 130)
plt.show(block=False)





#################################
#################################
#################################
#################################


test_wm(testing_activity, testing_behaviour)








def decode_trial( activity, target):
    test_interc = [1] + list(activity)
    x,y = weights.predict(test_interc)[0]
    y_real =np.sin(np.radians(target) )
    x_real = np.cos(np.radians(target) )
    error = angle_between( (x,y), (x_real, y_real))
    return error






testing_angles = np.array(testing_behaviour['T'])
for scan_s in range(nscans_wm):
    for trial_n in range(len(testing_angles)):
        test_interc = [1] + list(testing_activity[trial_n, scan_s, :])
        x,y = weights.predict(test_interc)[0]
        y_real =np.sin(np.radians(testing_angles[trial_n]) )
        x_real = np.cos(np.radians(testing_angles[trial_n]) )
        error = angle_between( (x,y), (x_real, y_real))
        time = scan_s * TR
        target = testing_angles.iloc[trial_n]
        response= testing_behaviour['A_R'].iloc[trail_n]
        df.append( [ error, Subject, Brain_region, time, trial_n, condition, target, response ])
    


df=pd.DataFrame(df)
df.columns=['error', 'Subject', 'Brain_region', 'time', 'trial', 'condition', 'target', 'response']



        #pred_angle = np.degrees(np.arctan2(y, x))
        ##
        #if pred_angle<0:
        #    pred_angle = 360+pred_angle 
        ##
        #abs_error = circdist(testing_angles[trial_n], pred_angle)





angle_between( (x,y), (x_real, y_real))

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    #return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return abs( np.rad2deg(ang1-ang2))











X = np.column_stack([1, testing_activity[0, 1, :] ])
p = fit.predict(X)
x = p[:,0]
y = p[:,1]

weights.predict(X)


spikes_train, spikes_test, beh_train, beh_test = train_test_split(df.groupby('neuron').get_group(Neur)['spikes'],
                                                                  df.groupby('neuron').get_group(Neur)['beh'],
                                                                  test_size=size_test)  

######## Trainning #########
## X matrix (intercept and spikes)
X = np.column_stack([np.ones(np.shape(spikes_train)[0]),spikes_train])
## Y (sinus and cos)
sinus =np.sin([np.radians(np.array(beh_train)[i]) for i in range(0, len(beh_train))])
cosinus = np.cos([np.radians(np.array(beh_train)[i]) for i in range(0, len(beh_train))])
Y = np.column_stack([cosinus, sinus])
### one OLS for sin and cos: output: beta of intercetp and bea of spikes (two B intercepts and 2 B for spikes )
Y = Y.astype(float) #to make it work in the cluster
X = X.astype(float)
model = sm.OLS(Y, X)
##train the model
training_weights = model.fit()
return training_weights





######### Testing ###########

#####
##### Error --> take the resulting vector in sin/cos space
### from sin and cos get the angle (-pi, pi)
#pred_angle = [ np.degrees(np.arctan2(y[i], x[i]) + np.pi) for i in range(0, len(y))]
pred_angle = [ np.degrees(np.arctan2(y[i], x[i])) for i in range(0, len(y))]
for i in range(0, len(pred_angle)):
    if pred_angle[i]<0:
        pred_angle[i]=360+pred_angle[i]
##
#error=[ circdist(beh_test[i], pred_angle[i]) for i in range(0, len(pred_angle))]
error=[ circdist(beh_test.values[i], pred_angle[i]) for i in range(0, len(pred_angle))]

#low_value --> predicted positionns close to real
neur_err.append(np.mean(error))













from model_functions import *
from fake_data_generator import *
from Weights_matrixs import *
from Representation import *
from process_encoding import *
from process_wm import *
from data_to_use import *
from joblib import Parallel, delayed
import multiprocessing
import time
import random


### Data to use
enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( 'n001', 'together', 'visual')

##### Process training data
training_dataset, training_targets = process_encoding_files(enc_fmri_paths, masks, enc_beh_paths, sys_use='unix', hd=4, TR=2.335)

##### Train your weigths
WM, Inter = Weights_matrix_Lasso_i( training_dataset, training_targets )
WM_t = WM.transpose()

##### Process testing data
testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition='2_7', distance='mix', sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
testing_angles = np.array(testing_behaviour['T'])
##random.shuffle(testing_angles)

### Respresentation
start_repres = time.time()
numcores = multiprocessing.cpu_count()

# TR separartion
signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    ####
Reconstruction = pd.concat(Reconstructions, axis=1) 
Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]

#Plot heatmap
plt.figure()
plt.title('Heatmap decoding')
######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") #, vmin=-0.4, vmax = 0.6) # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
plt.ylabel('Angle')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show(block=False)



######
######
######

end_repres = time.time()
process_recons = end_repres - start_repres
print( 'Time process reconstruction: ' +str(process_recons))

