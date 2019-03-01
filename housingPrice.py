#
# Various networks to predict house price (for NN module)
# Project 1B, part 4
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import preprocessing

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')  

NUM_FEATURES = 8

learning_rate = 1.e-9
beta = 1.e-3
epochs = 100
batch_size = 32
seed = 10
prob = 0.9
np.random.seed(seed)

first_hidden_layer = 20
second_hidden_layer = 20
third_hidden_layer = 20

# Read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]
testX, testY = X_data[:m], Y_data[:m]

# Normalize input data
scaler = preprocessing.StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

n = trainX.shape[0]

def ffnThreeLayers(x, y_):
    # Build the computational graph
    
    # Hidden layer
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, first_hidden_layer],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights1')
    b1 = tf.Variable(tf.zeros([first_hidden_layer]), name='biases1')
    hidden = tf.nn.relu(tf.matmul(x, w1) + b1)

    # Output layer
    w2 = tf.Variable(
        tf.truncated_normal([first_hidden_layer, 1],
            stddev=1.0 / math.sqrt(float(first_hidden_layer))), name='weights2')
    b2 = tf.Variable(tf.zeros([1]), name='biases2')
    y = tf.matmul(hidden, w2) + b2
    
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    loss = tf.reduce_mean(tf.square(y_ - y) + beta*regularization)
    mse = tf.reduce_mean(tf.square(y_ - y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    return train_op, mse

def ffnFourLayers(x, y_):
    # Build the computational graph
    
    # Hidden layer 1
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, first_hidden_layer],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights1')
    b1 = tf.Variable(tf.zeros([first_hidden_layer]), name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # Hidden layer 2
    w2 = tf.Variable(
        tf.truncated_normal([first_hidden_layer, second_hidden_layer],
            stddev=1.0 / math.sqrt(float(first_hidden_layer))), name='weights2')
    b2 = tf.Variable(tf.zeros([second_hidden_layer]), name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

    # Output layer
    w3 = tf.Variable(
        tf.truncated_normal([second_hidden_layer, 1],
            stddev=1.0 / math.sqrt(float(second_hidden_layer))), name='weights3')
    b3 = tf.Variable(tf.zeros([1]), name='biases3')
    y = tf.matmul(hidden2, w3) + b3
    
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
    loss = tf.reduce_mean(tf.square(y_ - y) + beta*regularization)
    mse = tf.reduce_mean(tf.square(y_ - y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    return train_op, mse

def ffnFiveLayers(x, y_):
    # Build the computational graph
    
    # Hidden layer 1
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, first_hidden_layer],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights1')
    b1 = tf.Variable(tf.zeros([first_hidden_layer]), name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    # Hidden layer 2
    w2 = tf.Variable(
        tf.truncated_normal([first_hidden_layer, second_hidden_layer],
            stddev=1.0 / math.sqrt(float(first_hidden_layer))), name='weights2')
    b2 = tf.Variable(tf.zeros([second_hidden_layer]), name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

    # Hidden layer 3
    w3 = tf.Variable(
        tf.truncated_normal([second_hidden_layer, third_hidden_layer],
            stddev=1.0 / math.sqrt(float(second_hidden_layer))), name='weights3')
    b3 = tf.Variable(tf.zeros([third_hidden_layer]), name='biases3')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w3) + b3)

    # Output layer
    w4 = tf.Variable(
        tf.truncated_normal([third_hidden_layer, 1],
            stddev=1.0 / math.sqrt(float(third_hidden_layer))), name='weights4')
    b4 = tf.Variable(tf.zeros([1]), name='biases4')
    y = tf.matmul(hidden3, w4) + b4
    
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
    loss = tf.reduce_mean(tf.square(y_ - y) + beta*regularization)
    mse = tf.reduce_mean(tf.square(y_ - y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, mse
  
def ffnFourLayersWithDropouts(x, y_, keep_prob):
    # Build the computational graph

    # Hidden layer 1
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, first_hidden_layer],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights1')
    b1 = tf.Variable(tf.zeros([first_hidden_layer]), name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    # Hidden layer 2
    w2 = tf.Variable(
        tf.truncated_normal([first_hidden_layer, second_hidden_layer],
            stddev=1.0 / math.sqrt(float(first_hidden_layer))), name='weights2')
    b2 = tf.Variable(tf.zeros([second_hidden_layer]), name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1_dropout, w2) + b2)
    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    # Output layer
    w3 = tf.Variable(
        tf.truncated_normal([second_hidden_layer, 1],
            stddev=1.0 / math.sqrt(float(second_hidden_layer))), name='weights3')
    b3 = tf.Variable(tf.zeros([1]), name='biases3')
    y = tf.matmul(hidden2_dropout, w3) + b3
    
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)
    loss = tf.reduce_mean(tf.square(y_ - y) + beta*regularization)
    mse = tf.reduce_mean(tf.square(y_ - y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    return train_op, mse

def ffnFiveLayersWithDropouts(x, y_, keep_prob):
    # Build the computational graph

    # Hidden layer 1
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, first_hidden_layer],
            stddev=1.0 / math.sqrt(float(NUM_FEATURES))), name='weights1')
    b1 = tf.Variable(tf.zeros([first_hidden_layer]), name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    # Hidden layer 2
    w2 = tf.Variable(
        tf.truncated_normal([first_hidden_layer, second_hidden_layer],
            stddev=1.0 / math.sqrt(float(first_hidden_layer))), name='weights2')
    b2 = tf.Variable(tf.zeros([second_hidden_layer]), name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1_dropout, w2) + b2)
    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    # Hidden layer 3
    w3 = tf.Variable(
        tf.truncated_normal([second_hidden_layer, third_hidden_layer],
            stddev=1.0 / math.sqrt(float(second_hidden_layer))), name='weights3')
    b3 = tf.Variable(tf.zeros([third_hidden_layer]), name='biases3')
    hidden3 = tf.nn.relu(tf.matmul(hidden2_dropout, w3) + b3)
    hidden3_dropout = tf.nn.dropout(hidden3, keep_prob)

    # Output layer
    w4 = tf.Variable(
        tf.truncated_normal([third_hidden_layer, 1],
            stddev=1.0 / math.sqrt(float(third_hidden_layer))), name='weights4')
    b4 = tf.Variable(tf.zeros([1]), name='biases4')
    y = tf.matmul(hidden3_dropout, w4) + b4
    
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
    loss = tf.reduce_mean(tf.square(y_ - y) + beta*regularization)
    mse = tf.reduce_mean(tf.square(y_ - y))
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, mse

def train():
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, 1])

    ffns = [ffnThreeLayers, ffnFourLayers, ffnFiveLayers]
    
    test_err = []
    for ffn in ffns:
        train_op, mse = ffn(x, y_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            idx = np.arange(n)
            
            test_err_ = []
            for i in range(epochs):
                np.random.shuffle(idx)
                trainX_shuffled, trainY_shuffled = trainX[idx], trainY[idx]

                for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                    train_op.run(feed_dict={x: trainX_shuffled[start:end], y_: trainY_shuffled[start:end]})

                test_err_.append(mse.eval(feed_dict={x: testX, y_: testY}))

                if i % 100 == 0 or i == epochs - 1:
                    print('iter %d: Test error: %g'%(i, test_err_[i]))

        test_err.append(test_err_)

    return test_err
      
def trainWithDropouts():
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    ffnsWithDropouts = [ffnFourLayersWithDropouts, ffnFiveLayersWithDropouts]
    
    test_err = []
    for ffnWithDropouts in ffnsWithDropouts:
        train_op, mse = ffnWithDropouts(x, y_, keep_prob)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            idx = np.arange(n)
            
            test_err_ = []
            for i in range(epochs):
                np.random.shuffle(idx)
                trainX_shuffled, trainY_shuffled = trainX[idx], trainY[idx]

                for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                    train_op.run(feed_dict={x: trainX_shuffled[start:end], y_: trainY_shuffled[start:end], keep_prob: prob})

                test_err_.append(mse.eval(feed_dict={x: testX, y_: testY, keep_prob: prob}))

                if i % 100 == 0 or i == epochs - 1:
                    print('iter %d: Test error: %g'%(i, test_err_[i]))

        test_err.append(test_err_)

    return test_err
    

def main():
    errorsWithoutDropouts = train()
    errorsWithDropouts = trainWithDropouts()
      
    # Plot test error for different layers
    plt.figure(1)
    plt.plot(np.arange(epochs), errorsWithoutDropouts[0], label=('Number of layers: 3'))
    plt.plot(np.arange(epochs), errorsWithoutDropouts[1], label=('Number of layers: 4'))
    plt.plot(np.arange(epochs), errorsWithoutDropouts[2], label=('Number of layers: 5'))
    plt.title('3-, 4- and 5-layer FFN networks')
    plt.legend(loc='upper right')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test error')
    plt.savefig('./figures/B_4_test_err_vs_epochs_no_dropouts.png')

    # Plot test error for 4-layer dropouts
    plt.figure(2)
    plt.plot(np.arange(epochs), errorsWithoutDropouts[1], label=('Without dropouts'))
    plt.plot(np.arange(epochs), errorsWithDropouts[0], label=('With dropouts'))
    plt.title('4-layer FFN networks with and without dropouts')
    plt.legend(loc='upper right')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test error')
    plt.savefig('./figures/B_4_test_err_vs_epochs_4layer_dropouts.png')

    # Plot test error for 5-layers dropouts
    plt.figure(3)
    plt.plot(np.arange(epochs), errorsWithoutDropouts[2], label=('Without dropouts'))
    plt.plot(np.arange(epochs), errorsWithDropouts[1], label=('With dropouts'))
    plt.title('5-layer FFN networks with and without dropouts') 
    plt.legend(loc='upper right')
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test error')
    plt.savefig('./figures/B_4_test_err_vs_epochs_5layer_dropouts.png')
    
    plt.show()

if __name__ == '__main__':
    main()
