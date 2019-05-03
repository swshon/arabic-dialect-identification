import numpy as np
import os,sys
sys.path.insert(0,'scripts/')
import shutil

BASE_FOLDER = sys.argv[1]
TARGET_FOLDER = sys.argv[2]

os.mkdir(TARGET_FOLDER)
wavlist = open(BASE_FOLDER+'/wav.scp').readlines()
wavlist = np.array(wavlist)
utt2lang = open(BASE_FOLDER+'/utt2lang').readlines()
utt2lang = np.array(utt2lang)


idx = range(len(wavlist))
np.random.shuffle(idx)

wavlist = wavlist[idx]
utt2lang = utt2lang[idx]

fid = open(TARGET_FOLDER+'/wav.scp','w')
for line in wavlist:
    fid.write('%s'%line)
fid.close()

fid = open(TARGET_FOLDER+'/utt2lang','w')
for line in utt2lang:
    fid.write('%s'%line)
fid.close()



