# Encoding model

### Files:

#### 1. M_W.py

+ 1.1 STEP 1 --> Crete the model of the channels (line 23)

+ 1.2 STEP 2 --> From the encoding (beh & images) --> Extract the portion of data (images) & Generate the hypothetical channel coefficients (beh)  (line 52 & line 132)

+ 1.3 STEP 3 --> Estimate the channel weights in each voxel (line 215)


<br/>

##### Simulated situation
![](https://github.com/davidbestue/encoding/blob/master/imgs/simulated_situation.png)

##### Histogram max channel response
![](https://github.com/davidbestue/encoding/blob/master/imgs/mx_ch_vx.png)

----

#### 2. WML3.py
+ 2.1 STEP 4 --> Extract the encoded channel response from the WM task trails
+ 2.2 STEP 5 --> Visualization

<br/>
<br/>

##### Heatmap
![](https://github.com/davidbestue/encoding/blob/master/imgs/heatmap.png)

##### ROI decoding
![](https://github.com/davidbestue/encoding/blob/master/imgs/roi_dec.png)


----

#### 3. brain_regions_decoding.py
+  3.1 Compare the decoding in time of the diff. regions of the brain.

<br/>
<br/>

##### Decoding brain regions
![](https://github.com/davidbestue/encoding/blob/master/imgs/dec_br_rg.png)




