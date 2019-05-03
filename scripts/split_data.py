import sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd

BASE_FOLDER = sys.argv[1]
TOTAL_SPLIT = int(sys.argv[2])

wavlist,utt_label,lang_label = kd.read_data_list(BASE_FOLDER,utt2lang=True)
kd.split_data(BASE_FOLDER,wavlist,utt_label,lang_label=lang_label,total_split=TOTAL_SPLIT)



