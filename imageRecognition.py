#
# Project 2A, part 3.
# Various CNNs for NTU NN course
#
PROJECT_DIR = 'figures/partA_ex3'

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

import os
if not os.path.isdir(PROJECT_DIR):
    print('creating the exercise folder')
    os.makedirs(PROJECT_DIR)
logger = open(PROJECT_DIR + "/log.txt","w+")

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 50
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

def cnn(images, keep_prob):
  images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

  # Parameters
  no_filters_1 = 50
  filter_1 = [9,9]
  no_filters_2 = 60
  filter_2 = [5,5]
  pooling_window = [2,2]
  stride = 2
  hidden = 300
    
  with tf.variable_scope('CNN'):
    conv1 = tf.layers.conv2d(
        images,
        filters=no_filters_1,
        kernel_size=filter_1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=pooling_window,
        strides=stride,
        padding='VALID')
    pool1_drop = tf.nn.dropout(pool1, keep_prob)
    conv2 = tf.layers.conv2d(
        pool1_drop,
        filters=no_filters_2,
        kernel_size=filter_2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=pooling_window,
        strides=stride,
        padding='VALID')
    

    dim = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value 
    pool2_flat = tf.reshape(pool2, [-1, dim])
    pool2_drop = tf.nn.dropout(pool2_flat, keep_prob)
    
  fc = tf.layers.dense(pool2_drop, hidden, activation=tf.nn.relu)
  fc_drop = tf.nn.dropout(fc, keep_prob)
  logits = tf.layers.dense(fc_drop, NUM_CLASSES)

  return conv1, pool1, conv2, pool2, logits

def main():
    trainX, trainY = load_data('data_batch_1')
    testX, testY = load_data('test_batch_trim')

    # Scale both images by the maximum intensity level allowed for an image with 8 bit colors
    trainX /= 255
    testX /= 255

    tf.reset_default_graph()
    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])   
    keep_prob = tf.placeholder(tf.float32)
    
    conv_1, pool_1, conv_2, pool_2, logits = cnn(x, keep_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

       
    training_steps = [
        tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss),
        tf.train.RMSPropOptimizer(learning_rate).minimize(loss),
        tf.train.AdamOptimizer(learning_rate).minimize(loss),
        tf.train.GradientDescentOptimizer(learning_rate, name='Dropouts').minimize(loss)
    ]
    
    keep_probs = [1.0, 1.0, 1.0, 0.5]

    N = len(trainX)
    idx = np.arange(N)
    
    train_cost = []
    test_acc = []
    
    for i in range(len(training_steps)):
        print('Calculating for ' + training_steps[i].name.split('_')[0] + '...')
        logger.write('Calculating for ' + training_steps[i].name.split('_')[0] + '...' + '\n')
        train_cost_ = []
        test_acc_ = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                np.random.shuffle(idx)
                trainX, trainY = trainX[idx], trainY[idx]

                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    training_steps[i].run(feed_dict={x: trainX[start:end], y_: trainY[start:end], 
                                                    keep_prob: keep_probs[i]})

                loss_ = loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: keep_probs[i]})
                train_cost_.append(loss_)
                acc_ = accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0})
                test_acc_.append(acc_)

                if e % 10 == 0:
                    print('epoch {:0>4d}: entropy {:.3f}, test accuracy {:.3f}'.format(e, loss_, acc_))
                    logger.write('epoch {:0>4d}: entropy {:.3f}, test accuracy {:.3f}'.format(e, loss_, acc_) + '\n')

        train_cost.append(train_cost_)
        test_acc.append(test_acc_)
    
    plt.figure()
    for i in range(len(train_cost)):
        plt.plot(np.arange(epochs), train_cost[i], label=(training_steps[i].name.split('_')[0]))
    plt.xlabel('epochs')
    plt.ylabel('training cost')
    plt.legend(loc='lower left')
    plt.savefig( PROJECT_DIR + '/3_train_cost.png')

    plt.figure()
    for i in range(len(test_acc)):
        plt.plot(np.arange(epochs), test_acc[i], label=(training_steps[i].name.split('_')[0]))
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend(loc='lower right')
    plt.savefig(PROJECT_DIR + '/3_test_accuracy.png')
    
    plt.show()
    
    logger.close()
  
if __name__ == '__main__':
    main()
