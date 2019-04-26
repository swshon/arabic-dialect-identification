# arabic-dialect-identification
Fine-grained, country-level Arabic dialect identification (17 Arabic countries)

This repository is to provide YouTube id for Arabic Dialect Identification (ADI) track of the fifth edition of the Multi-Genre Broadcast Challenge(MGB-5). Training example of end-to-end dialect identification system is also available.

# ADI Task 
The task of ADI is dialect identification of speech from YouTube to one of the 17 dialects (ADI17). 
The previous studies on Arabic dialect identification using audio signal is limited to 5 dialect classes by lack of speech corpus. 
To present a fine-grained analysis on the Arabic dialect speech, we collected Arabic dialect from YouTube.

# ADI17 dataset 
(Araibic dialect identification speech dataset of 17 Arabic countries) 

For Train set, about 3,000 hours of Arabic dialect speech data from 17 countries on the Arabic world was collected from YouTube. Since we collected the speech by considering the YouTube channels in a specific country, certain that the dataset might have some labeling errors. For this reason, we have two sub-tracks for the ADI task, supervised learning track and unsupervised track. Thus, the label of the train set can be either used or not and it completely depends on the choice of participants.

For the Dev and Test set, about 280 hours speech data was collected from YouTube. After automatic speaker linking and dialect labeling by human annotators, we selected 57 hours of speech dataset to use as Dev and Test set for performance evaluation. The test dataset was considered to have three sub-categories by the segment duration to represent short (under 5 sec), medium(between 5 sec and 20 sec), long duration (over 20 sec) of the dialectal speech.

# How to download Youtube dataset
All the YouTube unique id is available in the data folder.
First, clone this repository and then run the script as below

    To be updated...

# How to use data
Each data folder consisted of "segments", "utt2lang", "wav.scp" files. These file format is exactly same as Kaldi data preparation.

    segments: segment-level id, YouTube id, start, end time stamp(of segment)
    utt2lang: YouTube id, dialect label (one of 17 dialects)
    wav.scp: YouTube id, wav file location (you should change the /your_own_folder/ to the directory you downloaded

You can also find more examples at here (http://kaldi-asr.org/doc/data_prep.html)

# Submission of result file
You should submit dialect identification result in segment-level in the test set. (test set will be available along the timeline of MGB-5 challenge)

Example result on Dev set is available on the "result_dev.csv" file. The format are [segment id],[score of 1st dialect],...,[score of 17th dialect]. The order of dialects should follows the "data/language_id_initial" file. For example, 

    2XwBQJ7eHKs_055962-056391,4.243093,2.983541,2.239949,-0.058526,1.683865,0.002467,-2.371127,0.408665,0.663196,2.114708,0.014819,-0.584736,-1.905150,1.190056,-3.855960,-2.962667,-2.493101
    2XwBQJ7eHKs_056439-057583,13.503886,3.652979,9.747564,-0.765126,-1.163318,0.418676,-1.208075,-4.580471,-0.301157,4.584138,-3.774289,-5.396653,-8.809785,-0.212593,0.556646,4.696869,-8.873792
    2XwBQJ7eHKs_057651-057966,8.615689,-0.127747,6.443430,0.628338,-2.964191,0.540946,-2.832511,-0.427691,1.458990,2.927631,-3.526300,1.207520,-2.909723,-4.378843,0.685941,-1.413409,-6.319705

# Performance evaluation
Primary performance measure is accuracy (%) and alternative measure will be average EER for each dialects.
You can check the performance on the dev set as below (need result_dev.csv file)
    
    python scripts/measure_performance_dev.py
and the result shows

    Accuracy = 83.07%
    Average EER = 8.83%

# Original wav file download and embeddings
We will share the original wav file for the MGB-5 challenge particiants. Please contact us for more information

In a few days, we will also share the pre-trained network and extracted dialect embeddings for baseline.

# Contact
Please email to organizer if you have question.<br />
Suwon Shon (swshon@mit.edu)<br />
Ahmed Ali (amali@hbku.edu.qa)

https://arabicspeech.org/mgb5 





