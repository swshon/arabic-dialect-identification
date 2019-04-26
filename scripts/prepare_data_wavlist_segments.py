from pandas import read_table, DataFrame, concat
from glob import glob
from tqdm import tqdm

import os,sys
import tensorflow as tf
import numpy as np
import librosa
import feature_tools as ft


def write_tfrecords(feat, utt_label, utt_shape, tfrecords_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    trIdx = range(np.shape(utt_label)[0])
    
    # iterate over each example
    # wrap with tqdm for a progress bar
    for count,idx in enumerate(trIdx):
        feats = feat[idx].reshape(feat[idx].size)
        label = utt_label[idx]
        shape = utt_shape[idx]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(     
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'labels': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                'shapes': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=shape)),
                'features': tf.train.Feature(
                    float_list=tf.train.FloatList(value=feats.astype("float32"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    writer.close()
    print tfrecords_name+": total "+str(len(feat))+" feature and "+str(np.shape(utt_label))+" label saved"
    


# Feature extraction configuration

FEAT_TYPE = 'mfcc' #mfcc or melspec
N_FFT = 512
HOP = 160
VAD = True
CMVN = 'm'
EXCLUDE_SHORT=0 # assign 0(zero) if you don't want

if len(sys.argv)>1:
    if len(sys.argv)< 7:
        print "not enough arguments"
        print "example : python step01_prepare_data.py melspec 512 160 True mv 500"

    FEAT_TYPE = sys.argv[1]
    N_FFT = int(sys.argv[2])
    HOP = int(sys.argv[3])
    VAD = sys.argv[4]
    CMVN = sys.argv[5]
    EXCLUDE_SHORT = int(sys.argv[6])
    DATA_FOLDER = sys.argv[7]
    TOTAL_SPLIT = sys.argv[8]
    CURRRENT_SPLIT = sys.argv[9]
    SAVE_FOLDER = sys.argv[10]
    
if VAD =='False':
    VAD = False
if CMVN == 'False':
    CMVN = False
    

lines = open('data/language_id_initial').readlines()

lang2idx = {}
for line in lines:
    lang = line.rstrip().split()[0]
    idx = line.rstrip().split()[1]
    lang2idx[lang]=int(idx)

lines=open(DATA_FOLDER+'/utt2lang').readlines()
utt2lang={}
for iter in range(len(lines)):
    utt=lines[iter].rstrip().split()[0]
    lang=lines[iter].rstrip().split()[1]
    idx = lang2idx[lang]
    utt2lang[utt]=idx
    
    
lines=open(DATA_FOLDER+'/wav.scp').readlines()
utt2wav={}
for iter in range(len(lines)):
    utt=lines[iter].rstrip().split()[0]
    wav=lines[iter].rstrip().split()[1]
    utt2wav[utt]=wav
lines=open(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT+'/segments').readlines()
wav_list = []
seg_windows = []
utt_label = []

for iter in range(len(lines)):
    wav_list.append( utt2wav[lines[iter].rstrip().split()[1]] )
    utt_label.append( utt2lang[lines[iter].rstrip().split()[1]] )
    seg_windows.append([ np.float(lines[iter].rstrip().split()[2]), np.float(lines[iter].rstrip().split()[3])])

feat, _, utt_shape, tffilename = ft.feat_extract(wav_list,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT,seg_windows)

TFRECORDS_NAME = SAVE_FOLDER+'/'+DATA_FOLDER.split('/')[-1]+'_'+ tffilename +'.'+CURRRENT_SPLIT+'.tfrecords'
write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)



