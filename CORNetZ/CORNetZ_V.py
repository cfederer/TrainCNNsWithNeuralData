
import tensorflow as tf
import numpy as np


class CORNetZV(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer, v_batch_size=50): 
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        #
        # Parse input arguments into class variables
        self.X = x
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.NUM_CLASSES = num_classes
        self.BATCH_SIZE=128
        self.V_BATCH_SIZE = v_batch_size

        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE) as scope:
            
            self.W1 = tf.get_variable("W1", shape=[7, 7, 3, 64])
            self.b1 = tf.get_variable("b1", shape=[64])
            self.W2 = tf.get_variable("W2", shape=[3, 3, 64, 128])
            self.b2 = tf.get_variable("b2", shape=[128])
            self.W3 = tf.get_variable("W3", shape=[3, 3, 128, 256])
            self.b3 = tf.get_variable("b3", shape=[256])
            self.W4 = tf.get_variable("W4", shape=[3, 3, 256, 512])
            self.b4 = tf.get_variable("b4", shape=[512])
            self.W5 = tf.get_variable("W5", shape=[8192, 4096])
            self.b5 = tf.get_variable("b5", shape=[4096])
            self.W6 = tf.get_variable("W6", shape=[4096, 4096])
            self.b6 = tf.get_variable("b6", shape=[4096])
            self.W7 = tf.get_variable("W7", shape=[4096, self.NUM_CLASSES])
            self.b7 = tf.get_variable("b7", shape=[self.NUM_CLASSES])
            
        
    def forward(self):
        """Create the network graph."""

        ### V1 
        conv1 = conv(self.X, self.W1, self.b1, strides=[1, 2, 2, 1], padding='VALID', name='conv1')
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius =2 , alpha = 2e-05, beta = .75, name='norm1', bias=1.0)
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        ### V2
        conv2 = conv(pool1, self.W2, self.b2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        ##V4
        conv3 = conv(pool2, self.W3, self.b3, strides=[1, 1, 1, 1], padding='VALID', name='conv3')
        norm3 = tf.nn.local_response_normalization(conv3, depth_radius=2, alpha=2e-05, beta=.75, name='norm3')
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool3')

        ## IT
        conv4 = conv(pool3, self.W4, self.b4, strides=[1, 1, 1, 1], padding='VALID', name='conv4')
        norm4 = tf.nn.local_response_normalization(conv4, depth_radius=2, alpha=2e-05, beta=.75, name='norm4')
        pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool4')
        ##decoder
        flattened = tf.reshape(pool4, [128, 8192])

        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE) as scope:
            fc6 = fc(flattened, self.W5, self.b5, name='fc6')
        dropout6 = tf.nn.dropout(fc6, self.KEEP_PROB, seed=0)
        dropout6 = dropout6
        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE) as scope:
            fc7 = fc(dropout6, self.W6, self.b6, name='fc7')
        dropout7 = tf.nn.dropout(fc7, self.KEEP_PROB, seed=0)

        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE) as scope:
            fc8 = fc(dropout7, self.W7, self.b7, relu=False, name='fc8')

        return fc8 
        
    def forward_V1(self, img_batch, visual=False):
       ### V1
        conv1 = conv(img_batch, self.W1, self.b1, strides=[1, 2, 2, 1], padding='VALID', name='conv1')
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius =2 , alpha = 2e-05, beta = .75, name='norm1', bias=1.0)
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        if(visual):
            return(pool1)
        return tf.reshape(pool1, [self.V_BATCH_SIZE, -1])

    def forward_V4(self, img_batch, visual=False):
        ### V1 
        conv1 = conv(img_batch, self.W1, self.b1, strides=[1, 2, 2, 1], padding='VALID', name='conv1')
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius =2 , alpha = 2e-05, beta = .75, name='norm1', bias=1.0)
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        ### V2
        conv2 = conv(pool1, self.W2, self.b2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        ##V4
        conv3 = conv(pool2, self.W3, self.b3, strides=[1, 1, 1, 1], padding='VALID', name='conv3')
        norm3 = tf.nn.local_response_normalization(conv3, depth_radius=2, alpha=2e-05, beta=.75, name='norm3')
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool3')
        if(visual):
            return(pool3)
        return tf.reshape(pool3, [self.V_BATCH_SIZE, -1])

    def forward_IT(self, img_batch, visual=False):
        ### V1 
        conv1 = conv(img_batch, self.W1, self.b1, strides=[1, 2, 2, 1], padding='VALID', name='conv1')
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius =2 , alpha = 2e-05, beta = .75, name='norm1', bias=1.0)
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        ### V2
        conv2 = conv(pool1, self.W2, self.b2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        ##V4
        conv3 = conv(pool2, self.W3, self.b3, strides=[1, 1, 1, 1], padding='VALID', name='conv3')
        norm3 = tf.nn.local_response_normalization(conv3, depth_radius=2, alpha=2e-05, beta=.75, name='norm3')
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool3')

        ## IT
        conv4 = conv(pool3, self.W4, self.b4, strides=[1, 1, 1, 1], padding='VALID', name='conv4')
        norm4 = tf.nn.local_response_normalization(conv4, depth_radius=2, alpha=2e-05, beta=.75, name='norm4')
        pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool4')
        if(visual):
            return tf.reshape(pool4, [self.BATCH_SIZE, -1])
        return tf.reshape(pool4, [self.V_BATCH_SIZE, -1])

def conv(x, W, b, strides, padding, name):
    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)
    pre_act = tf.nn.bias_add(conv, b)
    relu = tf.nn.relu(pre_act)
    return relu

    
def fc(x,W, b, name, relu=True):
    """Create a fully connected layer."""
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, W, b, name=name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


