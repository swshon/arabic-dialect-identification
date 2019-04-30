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

Or, if you participate MGB-5 Challenge, you can request original wav files to organizer. Please find contact information below.

# How to use data
Each data folder consisted of "segments", "utt2lang", "wav.scp" files. These file format is exactly same as Kaldi data preparation.

    segments: segment-level id, YouTube id, start, end time stamp(of segment)
    utt2lang: YouTube id, dialect label (one of 17 dialects)
    wav.scp: YouTube id, wav file location (you should change the /your_own_folder/ to the directory you downloaded

You can also find more examples at here (http://kaldi-asr.org/doc/data_prep.html)

# Submission of result file
You should submit dialect identification result in segment-level in the test set. (test set will be available along the timeline of MGB-5 challenge)

Example result on Dev set is available on the "result_dev.csv" file. The format are [segment id],[score of 1st dialect],...,[score of 17th dialect]. The order of dialects should follows the "data/language_id_initial" file. For example, 

    2XwBQJ7eHKs_055962-056391,3.096405,1.777897,1.452979,-2.077643,-0.419325,-0.345230,-1.016110,-1.385472,3.511110,1.943294,-1.195900,-2.251954,-1.363119,2.556440,-1.121042,0.347785,-2.648575
    2XwBQJ7eHKs_056439-057583,12.810358,2.679880,4.258311,-1.582249,-0.770333,-1.986860,-3.571980,-0.695768,0.929641,3.286271,-0.159408,-4.281460,-3.185079,-1.564716,-0.437594,2.433181,-5.155079
    2XwBQJ7eHKs_057651-057966,7.816638,0.154628,8.818256,-0.711897,-0.899901,-1.057132,-1.626845,-3.524030,-1.554264,-2.776326,-0.864858,2.107841,-2.718826,-1.714218,2.685726,3.349209,-4.997468
    
The scores should represent likelihood score P(O|L_i), a conditional probability for observed data (O) given each language model (L_i). You can also find the example code to generate this "result_dev.csv" file in the "scripts/baseline_dev.py" file at the [line 82](https://github.com/swshon/arabic-dialect-identification/blob/3f7c61982a3f85fe4f6be06dd19b88bbb2b44cea/scripts/baseline_dev.py#L82). 

# Performance evaluation
Primary performance measure is accuracy (%) and alternative measure will be average EER for each dialects.
You can check the performance on the dev set as below (need result_dev.csv file). 
    
    python scripts/measure_performance_dev.py
and the result on Dev set shows

    Accuracy = 83.03%
    Average EER = 8.57%

# Baseline example code with pre-trained model
A baseline using end-to-end dialect identification system is provided and you can run the example code as below

    python scripts/baseline_dev.py
    
The example code extract MFCC feature from wav file and feed it to end-to-end dialect identification system. Finally, the extracted output layer activations saved in CSV format (result_dev.csv).

You can find more information about this baseline system in the paper below

Suwon Shon, Ahmed Ali, James Glass,<br />
Convolutional Neural Networks and Language Embeddings for End-to-End Dialect Recognition,<br />
Proc. Odyssey 2018 The Speaker and Language Recognition Workshop, 98-104 <br />
https://arxiv.org/abs/1803.04567<br />


# Requirements (for example training code and baseline code)
    Python 2.7
    tensorflow (python library, tested on 1.12)
    librosa (python library, tested on 0.6.0)

# Contact
Please email to organizer if you have question.<br />
Suwon Shon (swshon@mit.edu)<br />
Ahmed Ali (amali@hbku.edu.qa)

https://arabicspeech.org/mgb5 





