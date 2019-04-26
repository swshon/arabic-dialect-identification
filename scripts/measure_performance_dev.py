import numpy as np
from sklearn.metrics import roc_curve

def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])

def calculate_EER(trials, scores):
# calculating EER
# input: trials = boolean(or int) vector, 1: postive pair 0: negative pair
#        scores = float vector

    # Calculating EER
    fpr,tpr,threshold = roc_curve(trials,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)    
    return EER

# load language dictionary file to map language str to index (ex: 'EGY'->2)
lines = open('data/language_id_initial').readlines()
lang2idx = dict()
for line in lines:
    cols = line.rstrip().split()
    lang2idx[cols[0]] = int(cols[1])
    
print lang2idx

# load dev set label in segment level
lines = open('data/dev/utt2lang').readlines()
yid2lang=dict()
for line in lines:
    cols = line.rstrip().split()
    yid2lang[cols[0]] = lang2idx[cols[1]]
lines = open('data/dev/segments').readlines()
dev_label = []
for line in lines:
    dev_label.append( yid2lang[line.rstrip().split()[1]])

    
# Read result file (should be in the same order like )
lines = open('result_dev.csv').readlines()
scores = []
segid = []
for line in lines:
    cols = line.rstrip().split(',')
    segid.append(cols[0])
    scores.append(np.float_(cols[1:]))
scores = np.array(scores)

# calculate accuracy
total_lang = 17
label_mat = np.eye(total_lang)[dev_label] 
acc = accuracy(scores, label_mat)
print "Accuracy = %0.2f%%"% acc


# calculate average eer
eers=[]
for key in lang2idx:
    lang_label_vec = label_mat[:,lang2idx[key]]
    scores_vec = scores[:,lang2idx[key]]
    eers.append(calculate_EER(lang_label_vec,scores_vec))
    print "EER (%s) = %0.2f%%"% (key,eers[-1]*100)

print "Average EER = %0.2f%%"% (np.mean(eers)*100)
