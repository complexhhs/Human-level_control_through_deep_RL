# Build-up for the DQN Atari-breakout 
# neural network model
# input : None,84,84,4
# 1st layer : kernel= 8*8, strides= 4*4, channel=32, activation_function = relu, valid padding
# 2nd layer : kernel= 4*4, strides= 2*2, channel=64, activation_function = relu, valid padding
# 3rd layer : kernel= 3*3, strides= 1*1, channel=64, activation_function = relu, valid padding
# 1st Fullyconnected : 512 output, activation_function = relu, so far, 3rd layer # parameter is 7*7*64
# output: output dimension is action_space.n for action options, activation_function = softmax
from parameters import *
import tensorflow as tf

def Build_up(action_size):
	input = tf.placeholder(shape=[None,84,84,4],dtype = tf.float32)
	hidden_1 = tf.nn.conv2d(input,filter=[8,8,4,32],strides=[1,4,4,1],padding='VALID')
	hidden_1 = tf.nn.relu(hidden_1,name='h1')
	hidden_2 = tf.nn.conv2d(hidden_1,filter=[4,4,32,64],strides=[1,2,2,1], padding='VALID')
	hidden_2 = tf.nn.relu(hidden_2,name='h2')
	hidden_3 = tf.nn.conv2d(hidddn_2,filter=[3,3,64,64],strides=[1,1,1,1], padding='VALID')
	hidden_3 = tf.layers.flatten(tf.nn.relu(hidden_3),name='h3')
	weight_1 = tf.Variable(tf.truncated_normal(shape=[7*7*64,512],stddev=0.1),name='weight_1')
	bias_1 = tf.Variable(tf.constant(shape=[512],0.1),name='bias_1')
	fc_1 = tf.matmul(weight_1,hidden_3)+bias_1
	fc_1 = tf.nn.relu(fc_1,name='fc1')
	weight_2 = tf.Variable(tf.truncated_normal(shape=[512,action_size],stddev=0.1),name='weight_2')
	bias_2 = tf.Variable(tf.constant(shape=[action_size],0.1),name='bias_2')
	output = tf.nn.softmax(tf.matmul(weight_2,fc_1)+bias_2,name='output')
	models = {'h1':hidden_1,'h2':hidden_2,'h3':hidden_3,'weight_1':weight_1,'bias_1':bias_1,
		'fc_1':fc_1,'weight_2':weight_2,'bias_2':bias_2,'output':output}
	return models
	
	
def optimizing(prediction, target):	
	object = tf.reduce_mean(tf.square(prediction-target),name='obj_func')
	opt = tf.train.RMSProp(learning_rate = learning_rate ,momentum = momentum).minimize(object)
	return opt

	
