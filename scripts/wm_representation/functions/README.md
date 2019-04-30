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
