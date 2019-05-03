import os,sys
import tensorflow as tf
import numpy as np
import feature_tools as ft


def write_tfrecords(feat, utt_label, utt_shape, tfrecords_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    trIdx = range(np.shape(utt_label)[0])
    
    # iterate over each example
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

lines=open(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT+'/utt2lang').readlines()
utt_label = []
for line in lines:
    lang=line.rstrip().split()[1]
    utt_label.append(lang2idx[lang])
    
wav_list = []
lines=open(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT+'/wav.scp').readlines()
for line in lines:
    cols = line.rstrip().split()
    wav_list.append(cols[1])

feat, _, utt_shape, tffilename = ft.feat_extract(wav_list,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)

TFRECORDS_NAME = SAVE_FOLDER+'/'+DATA_FOLDER.split('/')[-1]+'_'+ tffilename +'.'+CURRRENT_SPLIT+'.tfrecords'
write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)






