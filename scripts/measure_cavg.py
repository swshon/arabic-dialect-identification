import numpy as np
from scipy.special import softmax

import argparse
parser = argparse.ArgumentParser(description="Measure Cost_average defined in NIST LRE2017 ", add_help=True)
parser.add_argument("--label", type=str, default="data/test_segments/utt2lang_sorted",help="label filename")
parser.add_argument("--score", type=str, default="result_test_sorted.csv", help="score filename")
parser.add_argument("--softmax", action='store_true', help="(option) use if scores need to be normalized using softmax")
parser.add_argument("--duration", action='store_true', help="(option) use if you want to measure by duration category")
args = parser.parse_known_args()[0]

LABEL_FILENAME = args.label
SCORE_FILENAME = args.score
print "\n\nLabel = %s"% (LABEL_FILENAME)
print "Score = %s"% (SCORE_FILENAME)


def measure_cavg(llr_scores,labels):
    C_mat = [[0]*np.shape(llr_scores)[1] for n in range(np.shape(llr_scores)[1])]
    P_m_mat = [[0]*np.shape(llr_scores)[1] for n in range(np.shape(llr_scores)[1])]
    P_fa_mat = [[0]*np.shape(llr_scores)[1] for n in range(np.shape(llr_scores)[1])]

    beta_1 = 1.0
    beta_2 = 9.0
    beta = beta_1

    for L_p in range(np.shape(llr_scores)[1]):
        positive_llr = llr_scores[:,L_p][labels ==L_p]
        P_m = len(np.nonzero(positive_llr>np.log(beta))[0]) /np.float32(len(positive_llr))
    #     negative_llr = llr_scores[:,L_p][labels !=L_p]
        for L_n in range(np.shape(llr_scores)[1]):
            if not L_p == L_n:
                negative_llr_by_Ln = llr_scores[:,L_p][labels ==L_n]
                P_fa = len(np.nonzero(negative_llr_by_Ln<np.log(beta))[0])/np.float32(len(negative_llr_by_Ln))
                C = P_m + beta*P_fa
                C_mat[L_p][L_n] = C
                P_m_mat[L_p][L_n] = P_m
                P_fa_mat[L_p][L_n] = P_fa

    C_mat = np.array(C_mat)
    P_m_mat = np.array(P_m_mat)
    P_fa_mat = np.array(P_fa_mat)

    C_avg = np.mean(np.sum(C_mat,0)/16.0)
    print "C_avg*100 = %.2f" % (C_avg*100.0)


lines = open(SCORE_FILENAME).readlines()
lines.sort()
scores = []
for line in lines:
    row = line.rstrip().split(',')
    scores.append(row[1:])

scores = np.array(scores) #[utt,lang]
scores = np.float32(scores)

lines = open('./data/language_id_initial').readlines()
lang2num={}
for line in lines:
    row = line.rstrip().split()
    lang2num[row[0]]=row[1]
print lang2num

lines = open(LABEL_FILENAME).readlines()
lines.sort()
labels = []
for line in lines:
    row = line.rstrip().split()
    labels.append(np.int(lang2num[row[1]]))
labels = np.array(labels)

lines = open(SCORE_FILENAME).readlines()
lines.sort()
dur = []
for line in lines:
    line = line.split(',')[0]
    head,tail = line.split('_')[-1].split('.wav')[0].split('-')
    dur.append(np.float(tail)/100.0 -np.float(head)/100.0)
    
dur = np.array(dur)

if args.softmax:
    for i in range(np.shape(scores)[0]):
        scores[i,:]= softmax(scores[i,:])
        
epsilon = 1e-20
scores = np.log(scores+epsilon)

llr_scores = np.zeros(np.shape(scores))
for L_p in range(np.shape(scores)[1]):
    for i in range(np.shape(scores)[0]):
        temp = 0
        for L_n in range(np.shape(scores)[1]):
            if L_n!=L_p:
                temp += np.exp( scores[i,L_n] - scores[i,L_p] )
        temp = temp / (np.shape(scores)[1]-1.0)
        llr_scores[i,L_p] = np.log(temp)

llr_scores = np.array(llr_scores)

print "\nPerformance evaluation (Overall)"
measure_cavg(llr_scores,labels)


if args.duration:    
    duration1=5
    duration2=20
    print "\nPerformance evaluation (duration>=%d)"%duration1
    measure_cavg(llr_scores[dur<duration1,:],labels[dur<duration1])
    print "\nPerformance evaluation (%d<=duration<%d)"%(duration1,duration2)
    measure_cavg(llr_scores[(dur<duration2) & (dur>=duration1),:],labels[(dur<duration2) & (dur>=duration1)])
    print "\nPerformance evaluation (duration>=%d)"%duration2
    measure_cavg(llr_scores[dur>=duration2,:],labels[dur>=duration2])


    
