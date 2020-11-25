
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

numcores = multiprocessing.cpu_count() 

def shuffled_reconstruction(signal_paralel, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    ### shuffle the targets
    testing_angles_sh=[] #new targets shuffled
    for n_rep in range(iterations):
        #new_targets = random.sample(targets, len(targets)) #shuffle the labels of the target
        #testing_angles_sh.append(new_targets)
        testing_angles_sh.append( np.array([random.choice([0, 90, 180, 270]) for i in range(len(targets))])) ## instead of shuffle, take a region where there is no activity!
    
    ### make the reconstryctions and append them
    Reconstructions_sh=[]
    for n_rep in range(iterations):
        time_rec_shuff_start = time.time() #time it takes
        Reconstructions_i = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles_sh[n_rep], WM, WM_t, intercept=Inter, ref_angle=180, plot=False)  for signal in signal_paralel) 
        Reconstruction_i = pd.concat(Reconstructions_i, axis=1) #mean of all the trials
        Reconstruction_i.columns =  [str(i * TR) for i in range(nscans_wm)] #column names
        Reconstructions_sh.append(Reconstruction_i) #append the reconstruction (of the current iteration)
        time_rec_shuff_end = time.time() #time
        time_rec_shuff = time_rec_shuff_end - time_rec_shuff_start
        print('shuff_' + str(n_rep) + ': ' +str(time_rec_shuff) ) #print time of the reconstruction shuffled
    
    ### Get just the supposed target location
    df_shuffle=[]
    for i in range(len(Reconstructions_sh)):
        n = Reconstructions_sh[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_sh[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_shuffle.append(n) #save thhis
    
    ##
    df_shuffle = pd.concat(df_shuffle)    #same shape as the decosing of the signal
    return df_shuffle



def bootstrap_reconstruction(testing_activity, targets, iterations, WM, WM_t, Inter, region, condition, subject, ref_angle=180):
    Reconstructions_boots=[]
    for n_rep in range(iterations):
        time_rec_boot_start=time.time()
        indexes_boots = np.random.randint(0,len(targets), len(targets))  #bootstraped indexes for reconstruction
        ### make the reconstryctions and append them
        targets_boot = targets[indexes_boots]
        signal_boots = testing_activity[indexes_boots, :, :] 
        signal_boots_paralel =[ signal_boots[:, i, :] for i in range(nscans_wm)]
        
        Reconstructions_boot = Parallel(n_jobs = numcores)(delayed(Representation)(signal, targets_boot, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_boots_paralel)    #### reconstruction standard (paralel)
        Reconstructions_boot = pd.concat(Reconstructions_boot, axis=1) #mean of all the trials
        Reconstructions_boot.columns =  [str(i * TR) for i in range(nscans_wm)] #column names
        Reconstructions_boots.append(Reconstructions_boot) #append the reconstruction (of the current iteration)
        time_rec_boot_end = time.time() #time
        time_rec_boot = time_rec_boot_end - time_rec_boot_start
        print('boot_' + str(n_rep) + ': ' +str(time_rec_boot) ) #print time of the reconstruction shuffled
        
    ### Get just the supposed target location
    df_boots=[]
    for i in range(len(Reconstructions_boots)):
        n = Reconstructions_boots[i].iloc[ref_angle*2, :] #around the ref_angle (x2 beacuse now we have 720 instead of 360)
        n = n.reset_index()
        n.columns = ['times', 'decoding']
        n['decoding'] = [sum(Reconstructions_boots[i].iloc[:, ts] * f2(ref_angle)) for ts in range(len(n))] #population vector method (scalar product)
        n['times']=n['times'].astype(float)
        n['region'] = region
        n['subject'] = subject
        n['condition'] = condition
        df_boots.append(n) #save thhis
    
    ##
    df_boots = pd.concat(df_boots)    #same shape as the decosing of the signal
    return df_boots



def all_process_condition_shuff_boot( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)

    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    else:
        'Error specifying the decode item'

    #
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
        ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
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
    print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
    
    df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, df_boots, shuffled_rec




def all_process_condition_shuff( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
    ##### Process testing data
    testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)

    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    else:
        'Error specifying the decode item'

    #
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    ### Respresentation
    start_repres = time.time()    
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
    #Plot heatmap
    if heatmap==True:
        plt.figure()
        plt.title(Condition)
        ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
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
    print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
    
    #df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
    ####### Shuff
    #### Compute the shuffleing
    shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec






training_activity, training_behaviour = delay_TR_cond, training_thing

enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)


def all_process_condition_shuff_l1o(WM, WM_t, testing_activity, testing_behaviour, decode_item):
        if decode_item == 'Target':
            dec_I = 'T'
        elif decode_item == 'Response':
            dec_I = 'A_R'
        elif decode_item == 'Distractor':
            dec_I = 'Dist'
        else:
            'Error specifying the decode item'
        ####
        #### Get the Trs with shared information and the TRs without shared information
        list_wm_scans= range(nscans_wm)  
        trs_shared = range(tr_st, tr_end)
        nope=[list_wm_scans.remove(tr_s) for tr_s in trs_shared]
        list_wm_scans2 = list_wm_scans
        ####
        #### Run the ones without shared information the same way
        testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
        ### Respresentation
        signal_paralel =[ testing_activity[:, i, :] for i in list_wm_scans2 ]
        Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
        Reconstruction_indep = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
        Reconstruction_indep.columns =  [str(i * TR) for i in list_wm_scans2 ]    ##column names
        ####
        #### Run the ones with shared information: leave one out
        for shared_TR in trs_shared:
            loo = LeaveOneOut()
            testing_data= testing_activity[:, shared_TR, :]
            
            for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles_sh[train_index], testing_angles_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_err = model_PV(X_train, X_test, y_train, y_test)

            WM, Inter = Weights_matrix_LM( delay_TR_cond, training_thing )
        WM_t = WM.transpose()








def all_process_condition_shuff_l1o( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
    if decode_item == 'Target':
        dec_I = 'T'
    elif decode_item == 'Response':
        dec_I = 'A_R'
    elif decode_item == 'Distractor':
        dec_I = 'Dist'
    else:
        'Error specifying the decode item'

    #
    testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
    # TR separartion
    signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
    Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
    Reconstruction = pd.concat(Reconstructions, axis=1) #mean of the reconstructions (all trials)
    Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names

    ####### Shuff
    #### Compute the shuffleing
    #shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
    
    return Reconstruction, shuffled_rec



def shuff_Pop_vect_leave_one_out2(testing_data, testing_angles, iterations):
    ## A esta función entrarán los datos de un TR y haré el shuffleing. 
    ## Es como Pop_vect_leave_one_out pero en vez de dar un solo error para un scan, 
    ## de tantas iterations shuffled (contiene un loop for y un shuffle )
    ## Alternativa: En vez de hacer n_iterations, hacer el shuffleing una vez y hacer una media de todos los errores
    ## Por eso es 2, en esta es shuffleing normal
    ## Pro alternativa: menos tiempo de computacion
    ## Contra: mas variabilidad (barras de error menos robustas)
    loo = LeaveOneOut()
    errors_shuffle=[]
    #########
    ########
    for i in range(iterations):
        # aquí estoy haciendo un shuffle normal (mezclar A_t)
        testing_angles_sh = np.array(random.sample(testing_angles, len(testing_angles)) )
        # una alternativa para que sea igual, sería asignar random 0, 90, 180 y 270
        #testing_angles_sh = np.array([random.choice([0, 90, 180, 270]) for i in range(len(testing_angles))])
        errors_=[]
        for train_index, test_index in loo.split(testing_data):
            X_train, X_test = testing_data[train_index], testing_data[test_index]
            y_train, y_test = testing_angles_sh[train_index], testing_angles_sh[test_index]
            ##
            ## correr el modelo en cada uno de los sets y guardar el error en cada uno de los trials
            ## la std no la hare con estos errores, sinó con el shuffle. No necesito guardar el error en cada repetición.
            model_trained_err = model_PV(X_train, X_test, y_train, y_test)
            errors_.append(model_trained_err) ## error de todos los train-test
        ##
        error_shuff_abs = np.mean([abs(errors_[i]) for i in range(0, len(errors_))]) 
        errors_shuffle.append(error_shuff_abs)
        #
    return errors_shuffle





#############################
#############################
#############################
#############################
#############################
############################# Remove after sending files to rita
#############################
#############################
#############################
#############################
#############################



# def all_process_condition_shuff_boot_rita( Subject, Brain_Region, WM, WM_t, Inter, Condition, iterations, distance, decode_item, method='together', heatmap=False):
#     enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
#     ##### Process testing data
#     testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
#     #
#     if decode_item == 'Target':
#         dec_I = 'T'
#     elif decode_item == 'Response':
#         dec_I = 'A_R'
#     elif decode_item == 'Distractor':
#         dec_I = 'Dist'
#     else:
#         'Error specifying the decode item'
#     #
#     #
#     testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
#     ### Respresentation
#     start_repres = time.time()    
#     # TR separartion
#     signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
#     Reconstructions_x  = Parallel(n_jobs = numcores)(delayed(Representation_rita)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
#     Reconstruction = pd.concat(Reconstructions_x[0], axis=1) #mean of the reconstructions (all trials)
#     Reconstruction.columns =  [str(i * TR) for i in range(nscans_wm)]    ##column names
#     #Plot heatmap
#     if heatmap==True:
#         plt.figure()
#         plt.title(Condition)
#         ax = sns.heatmap(Reconstruction, yticklabels=list(Reconstruction.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
#         ax.plot([0.25, np.shape(Reconstruction)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
#         plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
#         plt.ylabel('Angle')
#         plt.xlabel('time (s)')
#         plt.tight_layout()
#         plt.show(block=False)
#     ##
#     ######
#     ######
#     ######
#     end_repres = time.time()
#     process_recons = end_repres - start_repres
#     print( 'Time process reconstruction: ' +str(process_recons)) #print time of the process
#     #df_boots = bootstrap_reconstruction(testing_activity, testing_angles, iterations, WM, WM_t, Inter, Brain_Region, Condition, Subject, ref_angle=180)    
#     ####### Shuff
#     #### Compute the shuffleing
#     #shuffled_rec = shuffled_reconstruction(signal_paralel, testing_angles, iterations, WM, WM_t, Inter=Inter, region=Brain_Region, condition=Condition, subject=Subject, ref_angle=180)
#     return Reconstruction, Reconstructions_x, testing_behaviour #, df_boots, shuffled_rec




# testing_data=signal_paralel[0]



# def Representation_rita(testing_data, testing_angles, Weights, Weights_t, ref_angle=180, plot=False, intercept=False):
#     ## Make the data parallelizable
#     n_trials_test = len(testing_data) #number trials
#     data_prall = []
#     for i in range(n_trials_test):
#         data_prall.append(testing_data[i, :])
#         #data_prall.append(    np.array( stats.zscore(    testing_data[i, :] ))   ) ###what enters the formula is zscored!
        
        
    
#     ###
#     numcores = multiprocessing.cpu_count()
#     Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
#     #Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep_decode_trial_by_trial)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
#     Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)
    
#     df = pd.DataFrame()
#     n = list(Channel_all_trials_rolled.mean(axis=0)) #mean of all the trials rolled
#     df['TR'] = n #Name of the column
#     if plot==True:
#         #Plot heatmap
#         plt.figure()
#         plt.title('Heatmap decoding')
#         ######midpoint = df.values.mean() # (df.values.max() - df.values.min()) / 2
#         ax = sns.heatmap(df, yticklabels=list(df.index), cmap="coolwarm") # cmap= viridis "jet",  "coolwarm" RdBu_r, gnuplot, YlOrRd, CMRmap  , center = midpoint
#         ax.plot([0.25, np.shape(df)[1]-0.25], [posch1_to_posch2(18),posch1_to_posch2(18)], 'k--')
#         plt.yticks([posch1_to_posch2(4), posch1_to_posch2(13), posch1_to_posch2(22), posch1_to_posch2(31)] ,['45','135','225', '315'])
#         plt.ylabel('Angle')
#         plt.xlabel('time (s)')
#         plt.show(block=False)
    
#     return df, Channel_all_trials_rolled





# Reconstruction, Reconstructions_x, testing_behaviour = all_process_condition_shuff_boot_rita( Subject=Subject, Brain_Region=Brain_region, WM=WM, WM_t=WM_t, distance=Distance_to_use, decode_item= decoding_thing, iterations=50, Inter=Inter, Condition=Condition, method='together',  heatmap=False) #100



# testing_data, testing_angles, Weights, Weights_t, ref_angle=180, plot=False, intercept=False


# Subject=Subject
# Brain_Region=Brain_region
# WM=WM
# WM_t=WM_t
# distance=Distance_to_use
# decode_item= decoding_thing
# iterations=50
# Inter=Inter
# Condition=Condition
# method='together'
# heatmap=False


# enc_fmri_paths, enc_beh_paths, wm_fmri_paths, wm_beh_paths, masks = data_to_use( Subject, method, Brain_Region)
# ##### Process testing data
# testing_activity, testing_behaviour = preprocess_wm_files(wm_fmri_paths, masks, wm_beh_paths, condition=Condition, distance=distance, sys_use='unix', nscans_wm=nscans_wm, TR=2.335)
# #
# if decode_item == 'Target':
#     dec_I = 'T'
# elif decode_item == 'Response':
#     dec_I = 'A_R'
# elif decode_item == 'Distractor':
#     dec_I = 'Dist'
# else:
#     'Error specifying the decode item'
# #
# #

# testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
# ### Respresentation
# start_repres = time.time()    
# # TR separartion
# signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]

# Reconstructions_x  = Parallel(n_jobs = numcores)(delayed(Representation_rita)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)




# testing_data = testing_activity
# Weights =WM
# Weights_t = WM_t
# ref_angle=180
# plot=False
# intercept=False



# signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]



# n_trials_test = len(testing_data) #number trials
# data_prall = []
# for i in range(n_trials_test):
#     data_prall.append(testing_data[i, :])
#     #data_prall.append(    np.array( stats.zscore(    testing_data[i, :] ))   ) ###what enters the formula is zscored!
    
    

# ###
# numcores = multiprocessing.cpu_count()
# Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
# #Channel_all_trials_rolled = Parallel(n_jobs = numcores)(delayed(trial_rep_decode_trial_by_trial)(Signal, angle_trial, Weights, Weights_t, ref=ref_angle, intercept_ = intercept)  for Signal, angle_trial in zip( data_prall, testing_angles))    ####
# Channel_all_trials_rolled = np.array(Channel_all_trials_rolled)





# testing_angles = np.array(testing_behaviour[dec_I])    # A_R # T # Dist
# ### Respresentation
# start_repres = time.time()    
# # TR separartion
# signal_paralel =[ testing_activity[:, i, :] for i in range(nscans_wm)]
# Reconstructions = Parallel(n_jobs = numcores)(delayed(Representation)(signal, testing_angles, WM, WM_t, ref_angle=180, plot=False, intercept=Inter)  for signal in signal_paralel)    #### reconstruction standard (paralel)
