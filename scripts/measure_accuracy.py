import numpy as np
from sklearn.metrics import roc_curve
import sys

def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])


def accuracy_detail(predictions, labels,languages):
    
    confusionmat = np.zeros((len(languages),len(languages)))
    hypo_lang = np.argmax(predictions,axis = 1)
    temp = ((labels) - hypo_lang)
    acc =1- np.size(np.nonzero(temp)) / float(np.size(labels))
    print 'Final accurary on test dataset : %0.2f' %(acc*100)

    for i,lang in enumerate(languages):
        hypo_bylang = hypo_lang[ labels == i]
        hist_bylang = np.histogram(hypo_bylang,bins=range(len(languages)+1))
        confusionmat[:,i] = hist_bylang[0]

    # plt.rcParams["figure.figsize"]= [10,8]
    # plot_confusion_matrix(np.array(confusionmat.transpose()), classes=languages,normalize=True,
    #                       title='Confusion matrix')
    # plt.savefig('cm.png',dpi=300)

    true_positive =  np.diag(confusionmat)
    true_positive_plus_false_negative = np.sum(confusionmat,0)
    recall = np.mean(true_positive / true_positive_plus_false_negative)
    
    true_positive_plus_false_positive = np.sum(confusionmat,1)
    precision = np.mean(true_positive / true_positive_plus_false_positive)

    return acc*100, recall*100, precision*100


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


import argparse
parser = argparse.ArgumentParser(description="Measure accuracy, precision, recall ", add_help=True)
parser.add_argument("--label", type=str, default="data/test_segments/utt2lang_sorted",help="label filename")
parser.add_argument("--score", type=str, default="result_test_sorted.csv", help="score filename")
parser.add_argument("--detail", action='store_true', help="(option) use if you want to measure per language")
parser.add_argument("--duration", action='store_true', help="(option) use if you want to measure by duration category")
args = parser.parse_known_args()[0]


LABEL_FILENAME = args.label
SCORE_FILENAME = args.score
print "\n\nLabel = %s"% (LABEL_FILENAME)
print "Score = %s"% (SCORE_FILENAME)

# load language dictionary file to map language str to index (ex: 'EGY'->2)
lines = open('data/language_id_initial').readlines()
lang2idx = dict()
for line in lines:
    cols = line.rstrip().split()
    lang2idx[cols[0]] = int(cols[1])
    
print "language set = ",lang2idx

# load dev set label in segment level
lines = open(LABEL_FILENAME).readlines()
lines.sort()
true_label = []
for line in lines:
    cols = line.rstrip().split()
    true_label.append(lang2idx[cols[1]])
true_label = np.array(true_label)

# Read result file (should be in the same order like )
lines = open(SCORE_FILENAME).readlines()
lines.sort()
scores = []
segid = []
for line in lines:
    cols = line.rstrip().split(',')
    segid.append(cols[0])
    scores.append(np.float_(cols[1:]))
scores = np.array(scores)

# calculate accuracy
label_mat = np.eye(len(lang2idx.keys()))[true_label] 
acc,rec,pre = accuracy_detail(scores, true_label,lang2idx.keys())
print "\nPerformance evaluation (Overall)"
print "Accuracy = %0.2f%%, Recall = %0.2f%%, Precision = %0.2f%%"% (acc,rec,pre)


# calculate average eer
eers=[]
for key in lang2idx:
    lang_label_vec = label_mat[:,lang2idx[key]]
    scores_vec = scores[:,lang2idx[key]]
    eers.append(calculate_EER(lang_label_vec,scores_vec))
    if args.detail:
        print "EER (%s) = %0.2f%%"% (key,eers[-1]*100)

print "Average EER = %0.2f%%"% (np.mean(eers)*100)




languages = []
lines = open('data/language_id_initial').readlines()
for line in lines:
    languages.append(line.split()[0])

lines = open(SCORE_FILENAME).readlines()
lines.sort()
dur = []
for line in lines:
    line = line.split(',')[0]
    head,tail = line.split('_')[-1].split('.wav')[0].split('-')
    dur.append(np.float(tail)/100.0 -np.float(head)/100.0)
    
dur = np.array(dur)


if args.duration:
    duration = 5
    print "\nPerformance evaluation (duration<%d)"%duration
    # calculate accuracy
    acc,rec,pre = accuracy_detail(scores[dur<duration,:], true_label[dur<duration],lang2idx.keys())

    print "Accuracy = %0.2f%%, Recall = %0.2f%%, Precision = %0.2f%%"% (acc,rec,pre)

    # calculate average eer
    eers=[]
    duration = 5
    dur_labels = np.array(true_label)[dur<duration]
    dur_scores = scores[dur<duration,:]
    label_mat = np.eye(len(languages))[dur_labels] 

    for key in np.unique(true_label):
        lang_label_vec = label_mat[:,key]
        scores_vec = dur_scores[:,key]
        eers.append(calculate_EER(lang_label_vec,scores_vec))
        if args.detail:
            print "EER (%s) = %0.2f%%"% (languages[key],eers[-1]*100)

    print "Average EER = %0.2f%%"% (np.mean(eers)*100)



    duration1 = 5
    duration2 = 20

    print "\nPerformance evaluation (%d<=duration<%d)"%(duration1,duration2)
    # calculate accuracy
    acc,rec,pre = accuracy_detail(scores[(dur<duration2) & (dur>=duration1),:], true_label[(dur<duration2) & (dur>=duration1)],lang2idx.keys())
    print "Accuracy = %0.2f%%, Recall = %0.2f%%, Precision = %0.2f%%"% (acc,rec,pre)

    # calculate average eer
    eers=[]
    duration = 5
    dur_labels = np.array(true_label)[(dur<duration2) & (dur>=duration1)]
    dur_scores = scores[(dur<duration2) & (dur>=duration1),:]
    label_mat = np.eye(len(languages))[dur_labels] 

    for key in np.unique(true_label):
        lang_label_vec = label_mat[:,key]
        scores_vec = dur_scores[:,key]
        eers.append(calculate_EER(lang_label_vec,scores_vec))
        if args.detail:
            print "EER (%s) = %0.2f%%"% (languages[key],eers[-1]*100)

    print "Average EER = %0.2f%%"% (np.mean(eers)*100)



    duration = 20
    print "\nPerformance evaluation (duration>=%d)"%duration
    # calculate accuracy
    acc,rec,pre = accuracy_detail(scores[dur>=duration,:], true_label[dur>=duration],lang2idx.keys())
    print "Accuracy = %0.2f%%, Recall = %0.2f%%, Precision = %0.2f%%"% (acc,rec,pre)

    # calculate average eer
    eers=[]
    duration = 5
    dur_labels = np.array(true_label)[dur>=duration]
    dur_scores = scores[dur>=duration,:]
    label_mat = np.eye(len(languages))[dur_labels] 

    for key in np.unique(true_label):
        lang_label_vec = label_mat[:,key]
        scores_vec = dur_scores[:,key]
        eers.append(calculate_EER(lang_label_vec,scores_vec))
        if args.detail:
            print "EER (%s) = %0.2f%%"% (languages[key],eers[-1]*100)

    print "Average EER = %0.2f%%"% (np.mean(eers)*100)