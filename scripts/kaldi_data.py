import os
import numpy as np
import subprocess

def read_data_list(data):
    fileid = open(data+'/wav.scp','r')
    temp = fileid.readlines()
    fileid.close()
    
    filelist = []
    utt_label = []
    for iter,line in enumerate(temp):
        if len(line.split(' ')) ==2:
            filelist.extend([line.rstrip().split(' ')[-1]])
            utt_label.extend([line.rstrip().split(' ')[0]])
        else:
            if len(line.split('.sph'))==2:
                filelist.extend([line.rstrip().split('.sph')[0].split(' ')[-1] + '.sph'])
                utt_label.extend([line.rstrip().split(' ')[0]])
            elif len(line.split('.flac'))==2:
                filelist.extend([line.rstrip().split('.flac')[0].split(' ')[-1] + '.flac'])
                utt_label.extend([line.rstrip().split(' ')[0]])
                    
        
    fileid = open(data+'/utt2spk','r')
    temp = fileid.readlines()
    fileid.close()
    spk_label = []
    
    for iter,line in enumerate(temp):
        spk_label.extend([line.rstrip().split()[-1]])
    return filelist, utt_label, spk_label

def label2num(label,original_label):
    fid = open(original_label)
    lines = fid.readlines()
    fid.close()
    spks=[]
    for iter, line in enumerate(lines):
        spks.append(line.rstrip().split()[-1])
    spks = np.unique(spks)    
    spk_dict=dict()
    for iter in range(len(spks)):
        spk_dict[spks[iter]]=iter
    spk_label_num = []
    for iter in range(len(label)):
        spk_label_num.append(spk_dict[label[iter]])
    spk_label_num = np.array(spk_label_num)
    return spk_label_num

def split_data(name,filelist,utt_label,spk_label,total_split,lang_label=-1):
    split_len = len(utt_label)/total_split
    overflow = len(utt_label)%total_split
    print split_len, overflow
#     os.mkdir(name+'/split'+str(total_split))
    subprocess.call(['mkdir','-p', name+'/split'+str(total_split)])
    start=0
    end_=0

    for split in range(1,total_split+1):
        os.mkdir(name+'/split'+str(total_split)+'/'+str(split))
        filename_wav = name+'/split'+str(total_split)+'/'+str(split)+'/wav.scp'
        filename_utt2spk = name+'/split'+str(total_split)+'/'+str(split)+'/utt2spk'
        filename_utt2lang = name+'/split'+str(total_split)+'/'+str(split)+'/utt2lang'
        start = end_
        end_ = start+split_len
        if overflow>0:
            overflow-=1
            end_+=1
        if split==total_split:
            end_ = len(utt_label)
        with open(filename_wav,'w') as file:
            for iter in range(start,end_):
                file.write('%s %s\n' % (utt_label[iter], filelist[iter]) )
        with open(filename_utt2spk,'w') as file:
            for iter in range(start,end_):
                file.write('%s %s\n' % (utt_label[iter], spk_label[iter]) )
        if lang_label!=-1:
            with open(filename_utt2lang,'w') as file:
                for iter in range(start,end_):
                    file.write('%s %s\n' % (utt_label[iter], lang_label[iter]) )

                    
def split_segments(name,segments,total_split):
    split_len = len(segments)/total_split
    overflow = len(segments)%total_split
    print split_len,overflow
    start=0
    end_=0

    for split in range(1,total_split+1):
        subprocess.call(['mkdir','-p', name+'/split'+str(total_split)+'/'+str(split)])
        filename_segments = name+'/split'+str(total_split)+'/'+str(split)+'/segments'
        start = end_
        end_ = start+split_len
        if overflow>0:
            overflow-=1
            end_+=1
        if split==total_split:
            end_ = len(segments)

        with open(filename_segments,'w') as file:
            for iter in range(start,end_):
                file.write('%s\n' % (segments[iter].rstrip()) )
            

    
                        
def write_data(name,utt_label,filelist,spk_label,lang_label=-1):
#     os.mkdir(name)
    subprocess.call(['mkdir','-p', name])

    filename = name+'/wav.scp'
    with open(filename,'w') as file:
        for iter in range(len(utt_label)):
            file.write('%s %s\n' % (utt_label[iter], filelist[iter]) )

    # Utterance label using number
    filename = name+'/utt2spk'
    with open(filename,'w') as file:
        for iter in range(len(utt_label)):
            file.write('%s %s\n' % (utt_label[iter], spk_label[iter]) )
    if lang_label!=-1:
        filename = name+'/utt2lang'
        with open(filename,'w') as file:
            for iter in range(len(utt_label)):
                file.write('%s %s\n' % (utt_label[iter], lang_label[iter]) )
        
        
        


# def data_list_subset(max_spk,filelist, list, label, label_num):
#     idx = np.nonzero(label_num==max_spk)[0][0]
#     return filelist[:idx], list[:idx], label[:idx], label_num[:idx]

# def shuffle_write_data(name,list,filelist,label,label_num):
#     os.mkdir(name+'_shuffle')

#     shuffleidx = np.arange(0,len(list))
#     np.random.shuffle(shuffleidx)
#     # dev_spk_id = np.array(dev_spk_id)[shuffleidx]
#     list = np.array(list)[shuffleidx]
#     filelist = np.array(filelist)[shuffleidx]
#     label = np.array(label)[shuffleidx]
#     label_num = np.array(label_num)[shuffleidx]

#     filename = name+'_shuffle/wav.scp'
#     with open(filename,'w') as file:
#         for iter in range(len(list)):
#             file.write('%s %s\n' % (list[iter], filelist[iter]) )

#     # Utterance label using number
#     filename = name+'_shuffle/utt2spk'
#     with open(filename,'w') as file:
#         for iter in range(len(list)):
#             file.write('%s %s\n' % (list[iter], label_num[iter]) )
#     return filelist,list,label,label_num


    