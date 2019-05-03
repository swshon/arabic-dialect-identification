#!/data/sls/u/swshon/tools/pytf/bin/python
import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])

def txtwrite(filename, dict):
    with open(filename, "w") as text_file:
        for key, vec in dict.iteritems():
            text_file.write('%s [' % key)
            for i, ele in enumerate(vec):
                text_file.write(' %f' % ele)
            text_file.write(' ]\n')

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
            
        
#### function for read tfrecords
def read_and_decode_emnet_mfcc(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer(filename, name = 'queue')
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'labels': tf.FixedLenFeature([], tf.int64),
            'shapes': tf.FixedLenFeature([2], tf.int64),
            'features': tf.VarLenFeature( tf.float32)
        })
    # now return the converted data
    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
    shapes = tf.cast(shapes, tf.int32)
    feats2d = tf.reshape(feats.values, shapes)
    feats1d = feats.values
    return labels, shapes, feats2d


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

### Variable Initialization    
NUMGPUS = 1
SAVE_INTERVAL = 16000
LOSS_INTERVAL = 100
TESTSET_INTERVAL = 2000
MAX_SAVEFILE_LIMIT = 500
EERTEST_INTERVAL = 2000

utt_label = []
duration = np.empty(0,dtype='int')
spklab = []
TFRECORDS_FOLDER = './data/tfrecords/'
SAVER_FOLDERNAME = 'saver'

if len(sys.argv)< 13:
    print "not enough arguments"
    
resume = False
is_batchnorm = False
NN_MODEL = sys.argv[1]
LEARNING_RATE = np.float(sys.argv[2])
INPUT_DIM = sys.argv[3]
IS_BATCHNORM = sys.argv[4]
BATCHSIZE = int(sys.argv[5])
FEAT_TYPE = sys.argv[6]
DATA_NAME = sys.argv[7]
TOTAL_SPLIT = np.int(sys.argv[8])
SOFTMAX_NUM = np.int(sys.argv[9])
RESUME_STARTPOINT = np.int(sys.argv[10])
MAX_ITER = np.int(sys.argv[11])
TEST_SET_NAME = sys.argv[12]
MAX_INPUT_LENGTH = np.int(sys.argv[13]) # in frame
if RESUME_STARTPOINT > 0:
    resume = True


SAVER_FOLDERNAME = 'saver/'+NN_MODEL+'_'+str(MAX_INPUT_LENGTH)+'frame_'+FEAT_TYPE
if IS_BATCHNORM=='True':
    SAVER_FOLDERNAME = SAVER_FOLDERNAME + '_BN'
    is_batchnorm = True
nn_model = __import__(NN_MODEL)

records_shuffle_list = []
for i in range(1,TOTAL_SPLIT+1):
    records_shuffle_list.append(TFRECORDS_FOLDER+DATA_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


labels,shapes,feats = read_and_decode_emnet_mfcc(records_shuffle_list)
labels_batch,feats_batch,shapes_batch = tf.train.batch(
    [labels, feats,shapes], batch_size=BATCHSIZE, dynamic_pad=True, allow_smaller_final_batch=True,
    capacity=50)
FEAT_TYPE = FEAT_TYPE.split('_exshort')[0]
records_test_list = []
for i in range(1,2):
    records_test_list.append(TFRECORDS_FOLDER+TEST_SET_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


#data for validation
vali_labels,vali_shapes,vali_feats = read_and_decode_emnet_mfcc(records_test_list)

# test trials
tst_segments = []
tst_trials=[]
tst_enrolls = []
tst_tests = []


### Initialize network related variables
with tf.device('/cpu:0'):

    softmax_num = SOFTMAX_NUM
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               50000, 0.98, staircase=True)

    opt = tf.train.GradientDescentOptimizer(learning_rate)

    emnet_losses = []
    emnet_grads = []
    
    feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM)],name="feat_batch")
    label_batch = tf.placeholder(tf.int32, [None],name="label_batch")
    shape_batch = tf.placeholder(tf.int32, [None,2],name="shape_batch")
    
    test_feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM)],name="test_feat_batch")
    test_label_batch = tf.placeholder(tf.int32, [None],name="test_label_batch")
    test_shape_batch = tf.placeholder(tf.int32, [None,2],name="test_shape_batch")
    

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUMGPUS):
            with tf.device('/gpu:%d' % i):
                emnet = nn_model.nn(feat_batch,label_batch,label_batch,shape_batch, softmax_num,True,INPUT_DIM,is_batchnorm)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(emnet.loss)
                emnet_losses.append(emnet.loss)
                emnet_grads.append(grads)
        
        with tf.device('/gpu:0'):
            emnet_validation = nn_model.nn(test_feat_batch,test_label_batch,test_label_batch,test_shape_batch, softmax_num,False,INPUT_DIM,is_batchnorm);
            tf.get_variable_scope().reuse_variables()
    
    loss = tf.reduce_mean(emnet_losses)        
    grads = average_gradients(emnet_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=MAX_SAVEFILE_LIMIT)
    
    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)
    
    #load entire test set
    test_list = np.loadtxt('data/'+TEST_SET_NAME+'/wav.scp',dtype='string',usecols=[1])
    test_feats =[]
    test_labels = []
    test_shapes = []
    for iter in range(len(test_list)):
        a,b,c = sess.run([vali_feats,vali_labels,vali_shapes])
        test_feats.extend([a])
        test_labels.extend([b])
        test_shapes.extend([c])
    
    ### Training neural network 
    if resume:
        saver.restore(sess,SAVER_FOLDERNAME+'/model'+str(RESUME_STARTPOINT)+'.ckpt-'+str(RESUME_STARTPOINT))

    for step in range(RESUME_STARTPOINT,MAX_ITER):
        temp_feat,temp_labels,temp_shapes = sess.run([feats_batch,labels_batch,shapes_batch])
        
        
        #feed $MAX_INPUT_LENGTH frames for training
        if np.shape(temp_feat)[1]<MAX_INPUT_LENGTH:
            #use as it is if the frame length < $MAX_INPUT_LENGTH
            new_feat = temp_feat
        else:
            #randomly segment utterances to have MAX_INPUT_LENGTH
            new_feat = np.zeros([len(temp_feat),MAX_INPUT_LENGTH,np.int(INPUT_DIM)])
            for iter in range(len(temp_feat)):
                feat_start = 0
                feat_end = MAX_INPUT_LENGTH
                if c[iter,0]>MAX_INPUT_LENGTH:
                    feat_start = np.random.randint(0,c[iter,0]-MAX_INPUT_LENGTH,1)[0]
                    feat_end = feat_start +MAX_INPUT_LENGTH
                    temp_shapes[iter,0] = MAX_INPUT_LENGTH
                new_feat[iter,:,:] = temp_feat[iter,feat_start:feat_end,:]

        _, loss_v,mean_loss = sess.run([apply_gradient_op, emnet.loss,loss],feed_dict={
            feat_batch: new_feat,
            label_batch: temp_labels,
            shape_batch: temp_shapes
        })
        
        
        #quit if diverge
        if np.isnan(loss_v):
            print ('Model diverged with loss = NAN')
            quit()
            
        #measure accuracy on validataion set
        if step % EERTEST_INTERVAL ==0 and step>=RESUME_STARTPOINT:
            embeddings = []

            for iter in range(len(test_list)):
                eb = emnet_validation.o1.eval({test_feat_batch:[test_feats[iter]], test_label_batch:[test_labels[iter]], test_shape_batch:[test_shapes[iter]]})
                embeddings.extend([eb])
            embeddings = np.squeeze(embeddings)
            
            spklab_num_mat = np.eye(softmax_num)[test_labels] 
            acc = accuracy(embeddings, spklab_num_mat)
            print ('Step %d: loss %.6f, lr : %.5f, Accuracy : %f' % (step,mean_loss, sess.run(learning_rate),acc))

            
        #print loss    
        if step % LOSS_INTERVAL ==0:
            print ('Step %d: loss %.3f, lr : %.5f' % (step, mean_loss, sess.run(learning_rate)))

        #save model parameters    
        if step % SAVE_INTERVAL == 0 and step >=RESUME_STARTPOINT:
            saver.save(sess, SAVER_FOLDERNAME+'/model'+str(step)+'.ckpt',global_step=step)
