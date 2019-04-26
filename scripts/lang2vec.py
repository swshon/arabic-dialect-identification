import tensorflow as tf 
import numpy as np
class nn:

    # Create model
    def __init__(self, x1, y_, y_string, shapes_batch, softmax_num,is_training,input_dim, is_batchnorm):

        with tf.variable_scope("softmax") as scope:
            self.ea, self.eb, self.o1,self.res1,self.conv,self.ac1,self.ac2 = self.net(x1, shapes_batch, softmax_num,is_training,input_dim,is_batchnorm)
            
            # Create loss
            self.loss    = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=self.o1),name="loss_mean")
        self.label=y_
        self.shape = shapes_batch
        self.true_length = x1
        self.label_string=y_string
        
        

    def net(self,x, shapes_batch,softmax_num,is_training, input_dim, is_batchnorm):   
        shape_list = shapes_batch[:,0]
        
        featdim = input_dim #channel
        weights = []
        kernel_size =5
        stride = 1
        depth = 1000
                
        shape_list = shape_list/stride
        conv1 = self.conv_layer(x,kernel_size,featdim,stride,depth,'conv1',shape_list)
        conv1_bn = self.batch_norm_wrapper_1dcnn(conv1, is_training,'bn1',shape_list,is_batchnorm)
        conv1r= tf.nn.relu(conv1_bn)
       

        featdim = depth #channel
        weights = []
        kernel_size =7
        stride = 2
        depth = 1000
                
        shape_list = shape_list/stride
        conv2 = self.conv_layer(conv1r,kernel_size,featdim,stride,depth,'conv2',shape_list)
        conv2_bn = self.batch_norm_wrapper_1dcnn(conv2, is_training,'bn2',shape_list,is_batchnorm)
        conv2r= tf.nn.relu(conv2_bn)
       
        featdim = depth #channel
        weights = []
        kernel_size =1
        stride = 1
        depth = 1000
                
        shape_list = shape_list/stride
        conv3 = self.conv_layer(conv2r,kernel_size,featdim,stride,depth,'conv3',shape_list)
        conv3_bn = self.batch_norm_wrapper_1dcnn(conv3, is_training,'bn3',shape_list,is_batchnorm)
        conv3r= tf.nn.relu(conv3_bn)
       
        featdim = depth #channel
        weights = []
        kernel_size =1
        stride = 1
        depth = 1500
                
        shape_list = shape_list/stride
        conv4 = self.conv_layer(conv3r,kernel_size,featdim,stride,depth,'conv4',shape_list)
        conv4_bn = self.batch_norm_wrapper_1dcnn(conv4, is_training,'bn4',shape_list,is_batchnorm)
        conv4r= tf.nn.relu(conv4_bn)
        
        print conv1
        

        shape_list = tf.cast(shape_list, tf.float32)
        shape_list = tf.reshape(shape_list,[-1,1,1])
        mean = tf.reduce_sum(conv4r,1,keep_dims=True)/shape_list
        var = tf.sqrt( tf.reduce_sum( tf.square(conv4r-mean),1,keep_dims=True)/shape_list )
        res= tf.concat([mean,var],2)
        res1=tf.squeeze(res,axis=1)
        
        

        fc1 = self.fc_layer(res1,1500,"fc1")
        fc1_bn = self.batch_norm_wrapper_fc(fc1, is_training,'bn5',is_batchnorm)
#        ac1 = tf.nn.relu(fc1_bn)
        fc2 = self.fc_layer(fc1_bn,600,"fc2")
        fc2_bn = self.batch_norm_wrapper_fc(fc2, is_training,'bn6',is_batchnorm)
        ac2 = tf.nn.relu(fc2_bn)
        
        fc3 = self.fc_layer(ac2,softmax_num,"fc3")
        return fc1, fc2, fc3,res1,conv1r,fc2,fc3
        
    def xavier_init(self,n_inputs, n_outputs, uniform=True):
      if uniform:
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
      else:
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

    def fc_layer(self, bottom, n_weight, name):
        print( bottom.get_shape())
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]

        initer = self.xavier_init(int(n_prev_weight),n_weight)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.random_uniform([n_weight],-0.001,0.001, dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc
    
    
    def conv_layer(self, bottom, kernel_size,num_channels, stride, depth, name, shape_list):   # n_prev_weight = int(bottom.get_shape()[1])
        n_prev_weight = tf.shape(bottom)[1]

        inputlayer=bottom
        initer = tf.truncated_normal_initializer(stddev=0.1)

        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[kernel_size, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.001, shape=[depth], dtype=tf.float32))
        
        conv =  ( tf.nn.bias_add( tf.nn.conv1d(inputlayer, W, stride, padding='SAME'), b))
        
        mask = tf.sequence_mask(shape_list,tf.shape(conv)[1]) # make mask with batch x frame size
        mask = tf.where(mask, tf.ones_like(mask,dtype=tf.float32), tf.zeros_like(mask,dtype=tf.float32))
        mask=tf.tile(mask, tf.stack([tf.shape(conv)[2],1])) #replicate make with depth size
        mask=tf.reshape(mask,[tf.shape(conv)[2], tf.shape(conv)[0], -1])
        mask = tf.transpose(mask,[1, 2, 0])
        print mask
        conv=tf.multiply(conv,mask)
        return conv
    





    def batch_norm_wrapper_1dcnn(self, inputs, is_training, name, shape_list, is_batchnorm,decay = 0.999 ):
	if is_batchnorm:
	    shape_list = tf.cast(shape_list, tf.float32)
    	    epsilon = 1e-3
	    scale = tf.get_variable(name+'scale',dtype=tf.float32,initializer=tf.ones([inputs.get_shape()[-1]]) )
	    beta = tf.get_variable(name+'beta',dtype=tf.float32,initializer= tf.zeros([inputs.get_shape()[-1]]) )
	    pop_mean = tf.get_variable(name+'pop_mean',dtype=tf.float32,initializer = tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	    pop_var = tf.get_variable(name+'pop_var',dtype=tf.float32,initializer = tf.ones([inputs.get_shape()[-1]]), trainable=False)
	    if is_training:
	        #batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
                batch_mean = tf.reduce_sum(inputs,[0,1])/tf.reduce_sum(shape_list) # for variable length input
                batch_var = tf.reduce_sum(tf.square(inputs-batch_mean), [0,1])/tf.reduce_sum(shape_list) # for variable length input
	        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
	        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
	        with tf.control_dependencies([train_mean, train_var]):
	            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
	    else:
	        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
	else:
	    return inputs
		
                
                
                
    def batch_norm_wrapper_fc(self, inputs, is_training, name, is_batchnorm, decay = 0.999 ):
	if is_batchnorm:
            epsilon = 1e-3
            scale = tf.get_variable(name+'scale',dtype=tf.float32,initializer=tf.ones([inputs.get_shape()[-1]]) )
            beta = tf.get_variable(name+'beta',dtype=tf.float32,initializer= tf.zeros([inputs.get_shape()[-1]]) )
            pop_mean = tf.get_variable(name+'pop_mean',dtype=tf.float32,initializer = tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            pop_var = tf.get_variable(name+'pop_var',dtype=tf.float32,initializer = tf.ones([inputs.get_shape()[-1]]), trainable=False)
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
	else:
	    return inputs


