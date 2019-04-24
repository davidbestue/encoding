# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:14:46 2019

@author: David Bestue
"""

#### function to get Weight matrix

## trainiing data (trials, vx)
## training_angles (trials)



def Weights_matrix( training_data, training_angles ):
    #####
    n_voxels = shape(training_data)[0]
    
    ### Expected activity from the model
    M_model=[] #matrix of the activity from the model
    for i in training_angles:
        channel_values=f(i)  #f #f_quadrant (function that generates the expectd reponse in each channel)
        M_model.append(channel_values)
        
    M_model=pd.DataFrame(array(M_model)) # (trials, channel_activity)
    channel_names = ['ch_' +str(i+1) for i in range(0, len(pos_channels))] #names of the channels 
    M_model.columns=channel_names
    
    
    ####   2. Train the model and get matrix of weights
    Matrix_weights=zeros(( n_voxels, len(pos_channels) )) # (voxels, channels) how each channels is represented in each voxel
    
    for voxel_x in range(0, n_voxels): #train each voxel
        # set Y and X for the GLM
        Y = training_data[:, voxel_x] ## Y is the real activity
        X = M_model ## X is the hipothetycal activity 
        
        #### Lasso with penalization of 0.0001 (higher gives all zeros), fiting intercept (around 10 ) and forcing the weights to be positive
        lin = Lasso(alpha=0.001, precompute=True,  fit_intercept=True,  positive=True, selection='random')   
        lin.fit(X,Y) # fits the best combination of weights to explain the activity
        betas = lin.coef_ #ignore the intercept and just get the weights of each channel
        Matrix_weights[voxel_x, :]=betas #save the 36 weights for each voxel
    
    
    #####
    
    #Save the matrix of weights 
    Matrix_save=pd.DataFrame(Matrix_weights) #convert the array to dataframe
    Matrix_save.to_excel(writer_matrix,'sheet{}'.format(session_enc))
    Matrix_weights_transpose=Matrix_weights.transpose() #create the transpose for the IEM
    os.chdir(encoding_path)