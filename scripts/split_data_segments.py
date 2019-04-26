import sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd

BASE_FOLDER = sys.argv[1]
TOTAL_SPLIT = int(sys.argv[2])

segments = open(BASE_FOLDER+'/segments').readlines()
kd.split_segments(BASE_FOLDER,segments,TOTAL_SPLIT)



