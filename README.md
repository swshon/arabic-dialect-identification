# arabic-dialect-identification
Fine-grained, country-level Arabic dialect identification (17 Arabic countries)

This repository is to provide YouTube id for Arabic Dialect Identification (ADI) track of the fifth edition of the Multi-Genre Broadcast Challenge(MGB-5)

# Task 
The task of ADI is dialect identification of speech from YouTube to one of the 17 dialects (ADI17). 
The previous studies on Arabic dialect identification using audio signal is limited to 5 dialect classes by lack of speech corpus. 
To present a fine-grained analysis on the Arabic dialect speech, we collected Arabic dialect from YouTube.



# ADI17 dataset 
(Araibic dialect identification speech dataset of 17 Arabic countries) 

For Train set, about 3,000 hours of Arabic dialect speech data from 17 countries on the Arabic world was collected from YouTube. Since we collected the speech by considering the YouTube channels in a specific country, certain that the dataset might have some labeling errors. For this reason, we have two sub-tracks for the ADI task, supervised learning track and unsupervised track. Thus, the label of the train set can be either used or not and it completely depends on the choice of participants.
For the Dev and Test set, about 280 hours speech data was collected from YouTube. After automatic speaker linking and dialect labeling by human annotators, we selected 57 hours of speech dataset to use as Dev and Test set for performance evaluation. The test dataset was considered to have three sub-categories by the segment duration to represent short (under 5 sec), medium(between 5 sec and 20 sec), long duration (over 20 sec) of the dialectal speech.

# Performance measure
Primary performance measure is accuracy (%) and alternative measure will be average EER for each dialects.
You can check the performance on the dev set as below (need result_dev file)
    
    python scripts/measure_performance_dev.py
