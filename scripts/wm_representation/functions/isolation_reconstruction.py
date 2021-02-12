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
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut


numcores = multiprocessing.cpu_count() 



df = pd.read_excel('C:\\Users\\David\Desktop\\KI_Desktop\\data_reconstructions\\IEM\\example_beh.xlsx') 

targets_distractors = df[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]





def isolated_one


 targets_distractors = df[['T', 'NT1', 'NT2', 'Dist', 'Dist_NT1', 'Dist_NT2']]




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



