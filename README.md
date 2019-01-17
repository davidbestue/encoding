# Encoding model

Explanation of the procudure & code files


## Background

#### Weights matrix obtention

+ 1 Use the model to get the **hypo_ch_res**(trials, ch) for each encoding trial
      <br/>
       ***Direct problem*** :  angle --> Model --> channel_activity
      <br/>
      For each trial of the encoding task --> **hypo_ch_res**(trials, ch)

+ 2 Raw data encoding --> Preprocessed SPM --> Apply ROI mask --> High-pass filter & z-score per voxel

+ 3 Get the TRs of interest (2 consecutive TRs) and average them--> **Enc_TRs**: (trials, vx)

+ 4 Estimation the weights of each Voxel --> Loop of Liniar model with Lasso Regularization for each Vx
      <br/>
      for each Vx in **Enc_TRs**:
      <br/>
         **Enc_TRs**(trials,1) = **hypo_ch_res**(trials, ch) x **weights**(ch, 1)
      <br/>
      Append the **weights**(ch, 1) of each voxel --> **Weight_matrix**(vx, ch) (**WM** )


##### Squema weights matrix obtention
![](https://github.com/davidbestue/encoding/blob/master/imgs/IMG_3753.JPG)

<br/>

#### WM representation

+ 1 Raw data WM --> Preprocessed SPM --> Apply ROI mask --> High-pass filter & z-score per voxel

+ 2 Get the TRs of every trial--> **WM_TRs**: (trials, TR, vx)

+ 3 Get the subset matrix of interest--> **WM_TRs**: (trials, TR, vx) smaller

+ 4 ***INVERTED ENCODING MODEL***: Transform voxel activity into Channel activity 
      <br/>
      for all the TRs:
      <br/>
      **ch_activity**(ch,1) =  inv( **WM**t(ch, vx) x **WM**(vx, ch) ) x **WM**t(ch, vx)  x **WM_TR1** (vx, 1)
      <br/>
      
 + 5 Solve the ***inverse problem*** : angle <-- Model <-- channel_activity
      <br/>
      Using a smoother model (instead of a model of 36 ch, a model of 720 ch2)
      <br/>
      ***Kind of population vector***:  **ch_activity**(ch,1) --> sum (ch(x) x "Model(ch2(x))) --> **Angle repr**(ch2, 1)
      <br/>
      **Angle representation** --> Roll to preferred location
      <br/>
      **Angle_reo_all**(trial, TR, ch2)
      <br/>
      Average trials in each TR : **Angle representation matrix**(angles, TR)
      <br/>
      
 + 5 Visual Respresentation (heatmap for each TR)
 
      
##### Squema WM representation
![](https://github.com/davidbestue/encoding/blob/master/imgs/IMG_3754.JPG)



## Files:


Inside "scripts" you have the two main scripts. They are the files described below put together 

#### 0. functions_encoding_loop.py

+ Takes the paths for the files and the mask depending on the subject and method of analysis
+ specification depending on close, far or mix distances
+ All the functions that you need

<br/>


#### 1. loop_sess_analysis_1TR.py
+ Trains the encoding model (always making the average of 2TR) 
+ Tests in individual TRs of WM 


#### 2. loop_sess_analysis_2TR.py
+ Trains the encoding model (always making the average of 2TR) 
+ Tests in the average of 2 consecutive TRs of WM 


#### 3. loop_sess_analysis_1TR_se.py
+ Skips the training 
+ Tests in individual TRs of WM 


#### 4. loop_sess_analysis_2TR_se.py
+ Skips the training 
+ Tests in the average of 2 consecutive TRs of WM 


#### 5. combine_subjects.py
+ Average the individual results into a population result
+ by condition


### STEPS OF THE LOOP

for each subject
For each ROI
For each condition of WM task (4)

Depending on:
+ Method of analysis 
      + together --> train in all the encoding sessions at the same time
      + bysess --> train and test by session
     

#### Train in the encoding task

+ 1.1 STEP 1 :  Crete the model of the channels (36 channels)

+ 1.2 STEP 2 : From the encoding (beh & images) --> Extract the portion of data (images) & Generate the hypothetical channel coefficients (beh) 

+ 1.3 STEP 3 : Estimate the channel weights in each voxel 

<br/>

##### Simulated situation
![](https://github.com/davidbestue/encoding/blob/master/imgs/simulated_situation.png)

##### Histogram max channel response
![](https://github.com/davidbestue/encoding/blob/master/imgs/mx_ch_vx.png)

##### Mean weigth for each channel in the region
![](https://github.com/davidbestue/encoding/blob/master/imgs/weigth_per_channel.png)

----
#### Test in WM task

+ 2.1 STEP 4 : Extract the encoded channel response from the WM task trails

+ 2.2 STEP 5 : Visualization of Heatmap and preferred 

<br/>

##### Heatmap
![](https://github.com/davidbestue/encoding/blob/master/imgs/heatmap.png)

##### Preferred
![](https://github.com/davidbestue/encoding/blob/master/imgs/roi_dec.png)


----

### Population

After running the most convineint loop, you can average the results of each individual by running the "combine_subjects.py".
It will return the decoding value by condition and ROI.
It generates the heatmap average population, the preferred and the whole ROI (360 degrees)


##### Heatmap population
![](https://github.com/davidbestue/encoding/blob/master/imgs/heatmap_pop.png)

##### Preferred population
![](https://github.com/davidbestue/encoding/blob/master/imgs/dec_br_rg.png)

##### Whole ROI population
![](https://github.com/davidbestue/encoding/blob/master/imgs/dec_br_360.png)


