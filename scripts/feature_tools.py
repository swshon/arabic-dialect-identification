from tqdm import tqdm

import os,sys
import numpy as np
import librosa


def cmvn_slide(feat,winlen=300,cmvn=False): #feat : (length, dim) 2d matrix
    maxlen = np.shape(feat)[0]
    new_feat = np.empty_like(feat)
    cur = 1
    leftwin = 0
    rightwin = winlen/2
    
    # middle
    for cur in range(maxlen):
        cur_slide = feat[cur-leftwin:cur+rightwin,:] 
        #cur_slide = feat[cur-winlen/2:cur+winlen/2,:]
        mean =np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (feat[cur,:]-mean)/std # for cmvn        
        elif cmvn =='m':
            new_feat[cur,:] = (feat[cur,:]-mean) # for cmn
        if leftwin<winlen/2:
            leftwin+=1
        elif maxlen-cur < winlen/2:
            rightwin-=1    
    return new_feat


def feat_extract(filelist,feat_type,n_fft_length=512,hop=160,vad=True,cmvn=False,exclude_short=500,seg_windows=False):
#     filelist = np.loadtxt(filename,delimiter='\t',dtype='string',usecols=(0))
#     utt_label = np.loadtxt(filename,delimiter='\t',dtype='int',usecols=(1))
    
    feat = []
    utt_shape = []
    new_utt_label =[]
    for index,wavname in enumerate(tqdm(filelist)):
        
        #read audio input
        y, sr = librosa.core.load(wavname,sr=16000,mono=True,dtype='float')
        
        # take short segment if specified
        if seg_windows !=False:
            seg_window=seg_windows[index]
            start_sample = seg_window[0] * sr
            end_sample = seg_window[1] * sr
            y = y[ int(start_sample): int(end_sample) ]

        #extract feature
        if feat_type =='melspec':
            Y = librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955)
        elif feat_type =='mfcc':
            Y = librosa.feature.mfcc(y,sr,n_fft=n_fft_length,hop_length=hop,n_mfcc=40,fmin=133,fmax=6955)
        elif feat_type =='spec':
            Y = np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) )
        elif feat_type =='logspec':
            Y = np.log( np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) ) )
        elif feat_type =='plspec': #power-law compression
            Y = np.power( np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) ),0.3 )
        elif feat_type =='plspec_real': #power-law compression
            Y_original = librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400)
            Y = np.power( np.abs( np.real( Y_original )),0.3 ) * np.sign(np.real(Y_original))
        elif feat_type =='plspec_imag': #power-law compression
            Y_original = librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400)
            Y = np.power( np.abs( np.imag( Y_original )),0.3 ) * np.sign(np.imag(Y_original))
        elif feat_type =='logmel':
            Y = np.log( librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955) )

        Y = Y.transpose()

        # Simple VAD based on the energy
        if vad:
            E = librosa.feature.rmse(y, frame_length=n_fft_length,hop_length=hop,)
            threshold= np.mean(E)/2 * 0.90
            vad_segments = np.nonzero(E>threshold)
            if vad_segments[1].size!=0:
                Y = Y[vad_segments[1],:]

                
        #exclude short utterance under "exclude_short" value
        if exclude_short == 0 or (Y.shape[0] > exclude_short):
            if cmvn:
                Y = cmvn_slide(Y,300,cmvn)
            feat.append(Y)
            utt_shape.append(np.array(Y.shape))
#             new_utt_label.append(utt_label[index])
#             sys.stdout.write('%s\r' % index)
#             sys.stdout.flush()
            
#        if index ==100:
#            break

        
    tffilename = feat_type+'_fft'+str(n_fft_length)+'_hop'+str(hop)
    if vad:
        tffilename += '_vad'
    if cmvn=='m':
        tffilename += '_cmn'
    elif cmvn =='mv':
        tffilename += '_cmvn'
    if exclude_short >0:
        tffilename += '_exshort'+str(exclude_short)

    return feat, new_utt_label, utt_shape, tffilename #feat : (length, dim) 2d matrix
    
    
    
def do_shuffle(feat,utt_label,utt_shape):
    
    #### shuffling
    shuffleidx = np.arange(0,len(feat))
    np.random.shuffle(shuffleidx)

    feat=np.array(feat)
    utt_label=np.array(utt_label)
    utt_shape = np.array(utt_shape)

    feat = feat[shuffleidx]
    utt_label = utt_label[shuffleidx]
    utt_shape = utt_shape[shuffleidx]

    feat = feat.tolist()
    utt_label = utt_label.tolist()
    utt_shape = utt_shape.tolist()
    
    return feat, utt_label, utt_shape



