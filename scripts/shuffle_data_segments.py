import numpy as np
import os,sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd
import shutil

BASE_FOLDER = sys.argv[1]
TARGET_FOLDER = sys.argv[2]

os.mkdir(TARGET_FOLDER)
segments = open(BASE_FOLDER+'/segments').readlines()
segments = np.array(segments)
np.random.shuffle(segments)
fid = open(TARGET_FOLDER+'/segments','w')
for line in segments:
    fid.write('%s'%line)
fid.close()


shutil.copyfile(BASE_FOLDER+'/wav.scp',TARGET_FOLDER+'/wav.scp')
shutil.copyfile(BASE_FOLDER+'/utt2lang',TARGET_FOLDER+'/utt2lang')



