## Files:

Scripts to run the IEM in a simplified way. Some files are **functions** and other files run the whole **process**.  


### Process:  

#### fake_data_reconstruction.py  
+ Runs the whole analysis using fake data (calling functions).   

#### pipeline_example.py  
+ Runs the whole analysis for real data.  
+ No loop. 
+ Pipeline to know how to call the functions in the right order.  
+ One example.   

#### all_process.py  
+ Runs the whole analysis for real data.    
+ Creates a function with all the process.  
+ Fucntion of functions in the right order.  
+ Loop for the whole analysis.  
+ Returns a heatmap per subject and brain region with the 4 conditions.   

### Functions:  

#### fake_data_generator.py  
+ Function to generate fake data for 1 stim.  
+ Function to generate fake data for 3 stim.  

#### data_to_use.py  
+ Function that returns the paths for the data to analyze.  
+ Input: subject, brain_region, method of analysis.  
+ Output: images for encoding and wm, behaviour for enc and wm and masks.  

#### model_functions.py  
+ Functions related to the circular model.   
+ Model of the channel distribution and for the smoothing in the reconstruction.  

#### process_encoding.py  
+ Function to process the encoding images and behaviour.  
+ Input: paths of encoding fMRI, paths for encoding behaviour, hemodynamic delay and TR.  
+ Output: training images and training targets.  

#### Weights_matrix.py  
+ Function to train the model and estimate the weights to use for the testing.    
+ Input: training data (images and targets).   
+ Output: Matrix of weights.    

#### process_wm.py
+ Function to process the working memory task images and behaviour.  
+ Input: paths of wm fMRI, paths for wm behaviour, number of scans, condition and TR. 
+ Output: testing images and testing targets.  

#### Representation.py  
+ Function to reconstruct testing data using IEM.   
+ Input: tetsing images, tetsing behaviour, matrix of weights, matrix_transpose, reference angle.  
+ Output: Reconstrruction (720 ) and heatmap (optional).  


