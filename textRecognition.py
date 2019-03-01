# Various experiments for NTU Neural Network course
import numpy as np
import pandas
import tensorflow as tf
import pylab as plt
import csv

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

no_epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def getLayer2(layer_type, input_list):
  with tf.variable_scope('RNN' + layer_type):
    if layer_type == 'GRU' or layer_type == 'grad_clipping':
      cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      _, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
      return encoding
    
    elif layer_type == 'LSTM':
      cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
      outputs, states = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
      # two states in states, return the last output instead
      return outputs[-1]
    
    elif layer_type == 'vanilla':
      cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
      _, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
      return encoding
    
    elif layer_type == 'two_layers':
      cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
      _, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
      # return state of second layer
      return encoding[1]
    else:
      raise Exception('Invalid layer type given!')

def getLayer(layer_type, input_list):
  with tf.variable_scope('RNN' + layer_type):
    if layer_type == 'GRU' or layer_type == 'grad_clipping':
      cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      outputs, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
    
    elif layer_type == 'LSTM':
      cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
      outputs, states = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
    
    elif layer_type == 'vanilla':
      cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
      outputs, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
    
    elif layer_type == 'two_layers':
      cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
      outputs, encoding = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
      
    else:
      raise Exception('Invalid layer type given!')
    
  return outputs[-1]

def char_rnn_model(X_input, layer_type='GRU'):
  byte_vectors = tf.one_hot(X_input, 256)
  byte_list = tf.unstack(byte_vectors, axis=1)
  encoding = getLayer(layer_type, byte_list)
  
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits

def word_rnn_model(x, layer_type='GRU'):
  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)
  encoding = getLayer(layer_type, word_list)
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, word_list

def data_read_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)

  return x_train, y_train, x_test, y_test, no_words

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def main():
  global n_words, use_dropouts
  
  x_train_chars, y_train_chars, x_test_chars, y_test_chars = read_data_chars()
  x_train_words, y_train_words, x_test_words, y_test_words, n_words = data_read_words()

  layer_types = ['GRU', 'vanilla', 'LSTM', 'two_layers', 'grad_clipping']
  all_word_acc = []
  all_word_loss = []
  all_char_acc = []
  all_char_loss = []
  
  # Train the word RNN model
  x_train, y_train, x_test, y_test = x_train_words, y_train_words, x_test_words, y_test_words
  for layer_type in layer_types:
    print('Word', layer_type)
    tf.reset_default_graph()
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    logits, word_list = word_rnn_model(x, layer_type)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    
    if layer_type == 'grad_clipping':
      minimizer = tf.train.AdamOptimizer(lr)
      grads_and_vars = minimizer.compute_gradients(entropy)

      grad_clipping = tf.constant(2.0, name="grad_clipping")
      clipped_grads_and_vars = []
      for grad, var in grads_and_vars:
          clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
          clipped_grads_and_vars.append((clipped_grad, var))
          
      # Gradient updates
      train_op = minimizer.apply_gradients(clipped_grads_and_vars)
      
    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # training
      loss = []
      acc = []
      for e in range(no_epochs):
        np.random.shuffle(idx)
        x_train, y_train = x_train[idx], y_train[idx]
      
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          sess.run([word_list, train_op], {x: x_train[start:end], y_: y_train[start:end]})
        
        loss.append(entropy.eval(feed_dict={x:x_train, y_:y_train}))
        acc.append(accuracy.eval(feed_dict={x:x_test, y_:y_test}))
        if e%10 == 0:
          print('epoch {:0>4d}: entropy {:.3f}, test accuracy {:.3f}'.format(e, loss[e], acc[e]))

    all_word_acc.append(acc)
    all_word_loss.append(loss)
    
  # Train the char RNN model
  x_train, y_train, x_test, y_test = x_train_chars, y_train_chars, x_test_chars, y_test_chars
  for layer_type in layer_types:
    print('Character', layer_type)
    tf.reset_default_graph()
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    logits = char_rnn_model(x, layer_type)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL), 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    
    if layer_type == 'grad_clipping':
      minimizer = tf.train.AdamOptimizer(lr)
      grads_and_vars = minimizer.compute_gradients(entropy)

      grad_clipping = tf.constant(2.0, name="grad_clipping")
      clipped_grads_and_vars = []
      for grad, var in grads_and_vars:
          clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
          clipped_grads_and_vars.append((clipped_grad, var))
          
      # Gradient updates
      train_op = minimizer.apply_gradients(clipped_grads_and_vars)
      
    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # training
      loss = []
      acc = []
      for e in range(no_epochs):
        np.random.shuffle(idx)
        x_train, y_train = x_train[idx], y_train[idx]
      
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
      
        loss.append(entropy.eval(feed_dict={x: x_train, y_: y_train}))
        acc.append(accuracy.eval(feed_dict={x:x_test, y_:y_test}))
        if e%10 == 0:
          print('epoch {:0>4d}: entropy {:.3f}, test accuracy {:.3f}'.format(e, loss[e], acc[e]))

    all_char_acc.append(acc)
    all_char_loss.append(loss)


  plt.figure()
  plt.title('Training cost of character RNNs')
  for i, layer_type in enumerate(layer_types):
    plt.plot(np.arange(no_epochs), all_char_loss[i], label=layer_type)
  plt.xlabel('epochs')
  plt.ylabel('training cost')
  plt.legend(loc='lower left')
  plt.savefig('./figures/6_train_cost_char.png')

  plt.figure()
  plt.title('Test accuracy of character RNNs')
  for i, layer_type in enumerate(layer_types):
    plt.plot(np.arange(no_epochs), all_char_acc[i], label=layer_type)
  plt.xlabel('epochs')
  plt.ylabel('test accuracy')
  plt.legend(loc='lower right')
  plt.savefig('./figures/6_test_accuracy_char.png')
  
  plt.figure()
  plt.title('Training cost of word RNNs')
  for i, layer_type in enumerate(layer_types):
    plt.plot(np.arange(no_epochs), all_word_loss[i], label=layer_type)
  plt.xlabel('epochs')
  plt.ylabel('training cost')
  plt.legend(loc='lower left')
  plt.savefig('./figures/6_train_cost_word.png')

  plt.figure()
  plt.title('Test accuracy of word RNNs')
  for i, layer_type in enumerate(layer_types):
    plt.plot(np.arange(no_epochs), all_word_acc[i], label=layer_type)
  plt.xlabel('epochs')
  plt.ylabel('test accuracy')
  plt.legend(loc='lower right')
  plt.savefig('./figures/6_test_accuracy_word.png')

  plt.show()

if __name__ == '__main__':
  main()

