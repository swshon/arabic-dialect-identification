#! /usr/bin/env python

# Tested with Python 3.6.5
# youtube-dl --version 2019.04.30
# sox:      SoX v14.4.1
#ffmpeg version 2.8.15-0ubuntu0.16.04.1

##
# This script shows how to crawl the ADI17 audio files.
# You may need to run the script few times as sometime youtube blocks crawling
##

import subprocess
import os
import sys

youtubeIDList=sys.argv[1]

if not os.path.exists('wav'): os.mkdir('wav')

f = open(youtubeIDList,"r")


for line in f:
    subFolder='./wav/'+str(line.split()[1])
    youtubeID=str(line.split()[0])
    if not os.path.exists(subFolder): os.mkdir(subFolder)
    audioFile=subFolder+'/'+youtubeID+'.wav'
    if os.path.exists(audioFile):
        print ("Exist: ", audioFile)
        continue
    print ("Processing: ", audioFile, youtubeID)
    subprocess.run(['youtube-dl','-f','[ext=mp4]','--output','tmp.mp4',"--",str(youtubeID)],stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    subprocess.run(['ffmpeg','-i','tmp.mp4','tmp.wav'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    subprocess.run(['sox','tmp.wav','-r','16000','-c','1',str(audioFile)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if os.path.exists("tmp.mp4"): os.remove("tmp.mp4")
    if os.path.exists("tmp.wav"): os.remove("tmp.wav")
    if os.path.exists(audioFile):
        print ("Successed to download: ", line)
    else: 
        print ("Failed to download: ", line)


