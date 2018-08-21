# Encoding model

Explanation of the procudure & code files


### Background

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









### Files:

#### 0. functions_encoding.py

+ Decide the subject & the brian region
+ All the functions that you need

<br/>


#### 1. M_W.py

+ 1.1 STEP 1 :  Crete the model of the channels 

+ 1.2 STEP 2 : From the encoding (beh & images) --> Extract the portion of data (images) & Generate the hypothetical channel coefficients (beh) 

+ 1.3 STEP 3 : Estimate the channel weights in each voxel 


<br/>

#### Simulated situation
![](https://github.com/davidbestue/encoding/blob/master/imgs/simulated_situation.png)

#### Histogram max channel response
![](https://github.com/davidbestue/encoding/blob/master/imgs/mx_ch_vx.png)

#### Mean weigth for each channel in the region
![](https://github.com/davidbestue/encoding/blob/master/imgs/weigth_per_channel.png)

----

#### 2. WML3.py

+ 2.1 STEP 4 : Extract the encoded channel response from the WM task trails

+ 2.2 STEP 5 : Visualization



<br/>

#### Heatmap
![](https://github.com/davidbestue/encoding/blob/master/imgs/heatmap.png)

#### ROI decoding
![](https://github.com/davidbestue/encoding/blob/master/imgs/roi_dec.png)


----

#### 3. brain_regions_decoding.py

+  3.1 Compare the decoding in time of the diff. regions of the brain.


<br/>

#### Decoding brain regions
![](https://github.com/davidbestue/encoding/blob/master/imgs/dec_br_rg.png)




