# Basic NN
# Competition https://www.kaggle.com/c/sf-crime
import csv
import gzip
import numpy as np
import pandas as pd
from datetime import datetime
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

import keras.callbacks as Callback
import time

class TimeHistory(Callback.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# My own location where i store data for this on google drive
# gdrive/My Drive/ is the root folder for google drive
PROJECT_DIR = 'gdrive/My Drive/CE4032/'
#PROJECT_DIR =  'gdrive/My Drive/University/3.Singapore/CZ4032/'

EPOCHS = 15

BATCHES = 128
NETWORK = [128, 64]
OPTIMIZER = 'adam'
ACTIVATION_FUNCTION = 'relu'
DROPOUT = 0.5

RUN_FOLDS = True

# CONFIGURATIONS = [[batches, [hidden_layer1, hidden_layer2,...], 'OPTIMIZER'],[...],[....]]

CONFIGURATIONS = [
                   [128, [100, 60], 'nadam', 'basic'],
                   [
                    'Choose best optimizer',
                    [128,  [60], 'sgd', 'sgd'],
                    [128,  [60], 'rms', 'rms'],
                    [128,  [60], 'nadam', 'nadam'],
                   ],    
                   [
                    'Choose optimal network',
                    [128, [100, 60, 40], 'adam', '[100, 60, 40]'],
                    [128,  [40, 100, 60], 'adam', '[40, 100, 60]'],
                    [128,  [190, 140, 80, 40], 'adam', '[190, 140, 80, 40]']
                   ],
                   [
                    'Choose_optimal_batch_size',
                    [64, [64], 'adam', '64'],
                    [32, [64], 'adam', '32'],
                    [18, [64], 'adam', '18']
                   ]
                 ]

# testing
LIMIT_INPUT = False
# All categories for 200000
LIMIT = 100000
IS_PLOTTING = True
IS_SUBMITTING = False
NAME_OF_SUBMISSION = 'some-name'

SAVE_PROCESSED_DATA = False

def setConfig(config):
  global BATCHES
  global NETWORK
  global OPTIMIZER

  BATCHES = config[0]
  NETWORK = config[1]
  OPTIMIZER = config[2]
  
def getConfig():
  return 'Config: BATCHES:{}, NETWORK: {}, OPTIMIZER: {}'.format(BATCHES, NETWORK, OPTIMIZER)

def printNicely(string, number=20):
  print('\n' + '---' * number)
  print(string)
  print('---' * number)


def parseTime(x):
  DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
  return DD.hour,DD.month

def purifyAddress(dataFrame):
  frequentCrimeAddresses = dataFrame['Address'].value_counts()
  frequentCrimeAddresses = frequentCrimeAddresses[frequentCrimeAddresses >=100].index

  dataFrame['Address_clean'] = dataFrame['Address']
  dataFrame.loc[~dataFrame['Address'].isin(frequentCrimeAddresses), "Address_clean"] = 'not-important'

  dataFrame['Address']= dataFrame['Address_clean']
  del dataFrame['Address_clean']

def processData(dataFrame, purify=False):

  # Optional
  # purifyAddress(dataFrame)
  dataFrame['is_block'] = dataFrame['Address'].apply(lambda x : 1 if 'block' in x.lower() else 0)
  # dataFrame["is_intersection"]=dataFrame["Address"].apply(lambda x: 1 if "/" in x else 0)

  dataFrame.drop(['Address'],inplace=True,axis=1)
  if 'Descript' in dataFrame:
    dataFrame.drop(['Descript', 'Resolution'],inplace=True,axis=1)
  else:
    dataFrame.drop(['Id'],inplace=True,axis=1)
  
  if purify:
    #To check null - there is no missing data 
    dataFrame.isnull().sum()
    print('input data shape after NaN cleaning: {}'.format(np.shape(dataFrame)))
    print(dataFrame[dataFrame.duplicated(keep=False)])
    dataFrame.drop_duplicates(inplace = True) 
    print('input data shape after cleaning: {}'.format(np.shape(dataFrame)))

  # Handling for outliers for lattitude and langitude
  dataFrame['Y'] = dataFrame['Y'].apply(lambda x : x if 37.82 > x else 37.82 )
  dataFrame['X'] = dataFrame['X'].apply(lambda x : x if -122.3 > x else -122.3 )

  # Split datetime
  dataFrame['hour'], dataFrame['month'] = zip(*dataFrame['Dates'].apply(parseTime))
  del dataFrame['Dates']

  dataFrame = pd.get_dummies(dataFrame, columns = ['DayOfWeek','PdDistrict'], drop_first=True)

  # Optional
  # dataFrame["is_awake"]=dataFrame["hour"].apply(lambda x: 1 if hour >=8 and hour ==0 else 0) 
  # dataFrame["summer"], dataFrame["fall"], dataFrame["winter"], dataFrame["spring"]=zip(*dataFrame["month"].apply(get_season))

  print('input data shape after input processing: {}'.format(np.shape(dataFrame)))
  return dataFrame

def splitData(dataFrame):
  # split into input and output
  X = dataFrame.drop(['Category'],axis=1).values
  Y = dataFrame['Category'].values

  # scale input
  X = StandardScaler().fit_transform(X)

  if SAVE_PROCESSED_DATA:
    print('Saving training data...')
    X.to_csv(PROJECT_DIR + 'train_proces_X.csv', index=False)
    Y.to_csv(PROJECT_DIR + 'train_proces_Y.csv', index=False)
    
  Y = pd.get_dummies(dataFrame['Category'], columns = ['Category']).values # Categories to vectors for category crosscorelation
    
  return X, Y

def shuffle(data, seed=1337):
  np.random.seed(seed)
  shuffle = np.arange(len(data))
  np.random.shuffle(shuffle)
  data = data[shuffle]
  return data

def build_model(input_dim, output_dim):
  model = Sequential()
  model.add(Dense( NETWORK[0], input_shape=(input_dim,), activation='relu'))
  model.add(Dropout(DROPOUT))

  for i in range(len(NETWORK)):
    dim = output_dim
    if i+1 < len(NETWORK): # if not last layer, shape to the next layer
      dim = NETWORK[i+1]
    model.add(Dense(dim, input_shape=(NETWORK[i],), activation='relu'))
    model.add(BatchNormalization(input_shape=(NETWORK[i],)))
    model.add(Dropout(DROPOUT))
  
  model.add(Dense(output_dim, input_shape=(NETWORK[-1],)))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
  return model

def kfold(X, Y, input_dim, output_dim):
  nb_folds = 4
  kfolds = KFold(len(Y), nb_folds)
  av_loss = 0.
  av_time = 0
  fold = 0
  
  for train, valid in kfolds:
    print('\nFold', fold+1)
    fold += 1
    X_train = X[train]
    X_valid = X[valid]
    Y_train = Y[train]
    Y_valid = Y[valid]

    print("Building model...")
    model = build_model(input_dim, output_dim)

    print("Training model...")

    time_callback = TimeHistory()
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCHES, validation_data=(X_valid, Y_valid), verbose=1, callbacks=[time_callback])
    valid_preds = model.predict_proba(X_valid)
    loss = metrics.log_loss(Y_valid, valid_preds)
    print("Loss:", loss)
    av_loss += loss
    av_time += sum(time_callback.times)/EPOCHS

  print('Average loss:', av_loss / nb_folds)
  print('Average epoch time:', av_time / nb_folds)
  return (av_loss/nb_folds, av_time / nb_folds)

def plotTestingResults(configurations, av_losses, avg_times, run_type):
  final = []
  for i in range(len(configurations)):
    print(configurations[i], av_losses[i], avg_times[i])
    final.append((configurations[i], av_losses[i], avg_times[i]))
  
  final.append((default_conf, default_conf_loss, default_conf_time))
  final_sorted =  sorted(final, key=lambda tup: tup[1])
  
  # plt.figure()
  fig, ax1 = plt.subplots()
  ind = np.arange(len(final_sorted))

  ax1.set_xticks(ind)
  ax1.set_xticklabels(list(map(lambda x: x[0][3], final_sorted)))

  color = 'tab:red'
  ax1.set_ylabel('av_losses', color=color)
  ax1.plot(ind, list(map(lambda x: x[1], final_sorted)), color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.set_ylabel('avg_times (sec)', color=color)  # we already handled the x-label with ax1
  ax2.plot(ind, list(map(lambda x: x[2], final_sorted)), color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped  
  plt.savefig('figures/{}.png'.format(run_type))
  plt.show()

  printNicely("Best training model...\n{}".format(final_sorted[0][0][3]), 10)
  print('Results:')
  for i in range(len(final_sorted)):
    print('{} Average loss: {}'.format(final_sorted[i][0][3], final_sorted[i][1]))

def produceSubmission(model):
  labels = 'ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'.split(',')

  print('Loading testing data...')
  raw_test = pd.read_csv(PROJECT_DIR + 'test.csv')
  if LIMIT_INPUT:
    raw_test = raw_test[:LIMIT]
  print('Creating testing data...')
  X_test = processData(raw_test, purify=False)
  del raw_test
  
  print('Features:{}'.format(list(X_test)))
  print(getConfig())

  X_test = X_test.values #convert to numpy from pandas
  print('test data shape: {}'.format(np.shape(X_test)))

  # scale input
  X_test = StandardScaler().fit_transform(X_test)

  if SAVE_PROCESSED_DATA:
    print('Saving testing data...')
    X_test.to_csv(PROJECT_DIR + 'test_proces_X.csv', index=False)

  print('Predicting over testing data...')
  preds = model.predict_proba(X_test, verbose=1)
  
  print('Prediction shape')
  print(np.shape(preds))

  print('Writing to file...')
  import sys
  rows_no = X_test.shape[0]
  
  with gzip.open(PROJECT_DIR + NAME_OF_SUBMISSION + '.csv.gz', 'wt') as outf:
    fo = csv.writer(outf, lineterminator='\n')
    fo.writerow(['Id'] + labels)

    for i, pred in enumerate(preds):
      fo.writerow([i] + list(pred))  
      
      if i % int(rows_no*0.02) == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{:.1%}'.format(i/rows_no))
        sys.stdout.flush()


def main():

  print('Loading training data...')
  raw_data = pd.read_csv(PROJECT_DIR + 'train.csv')
  if LIMIT_INPUT: 
    raw_data = pd.DataFrame(shuffle(raw_data.values), index=raw_data.index, columns=raw_data.columns)
    raw_data = raw_data[:LIMIT]

  print('raw data shape: {}'.format(np.shape(raw_data)))

  print('Processing training data...')
  data = processData(raw_data)
  del raw_data
  X, Y = splitData(data)
  X, Y = shuffle(X), shuffle(Y)
  print('Processed data shape: X: {}, Y: {}'.format(np.shape(X), np.shape(Y)))

  input_dim = X.shape[1]
  output_dim = Y.shape[1]
  
  # This runs various configurations and picks the best one
  if RUN_FOLDS:
    print('Running folds...')
    startTime = time.time()
    setConfig(CONFIGURATIONS[0])
    global default_conf
    global default_conf_loss
    global default_conf_time

    print('Running for default configuration...')
    default_conf = CONFIGURATIONS[0]
    default_conf_loss, default_conf_time = kfold(X, Y, input_dim, output_dim)
    for configurations in CONFIGURATIONS[1:]:
#     for configurations in CONFIGURATIONS:
      run_type = configurations[0]
      printNicely(run_type, 20)
      configurations = configurations[1:] # first elem is a string telling the type
      configs_average_loss = []
      configs_average_time = []

      for config in configurations:
        setConfig(config) # Updates global variables, declared at the top
        printNicely(getConfig(), 10)

        avg_loss, avg_time = kfold(X, Y, input_dim, output_dim)
        configs_average_loss.append(avg_loss)
        configs_average_time.append(avg_time)

      plotTestingResults(configurations, configs_average_loss, configs_average_time, run_type)

    endTime = time.time()

    print("It took {} to run". format(endTime - startTime))

#   model = build_model(input_dim, output_dim)
#   history = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCHES, verbose=1)
  
#   if IS_PLOTTING:
#     # Plot training & validation accuracy values
#     plt.figure()
#     plt.plot(history.history['acc'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.savefig('figures/final_model_acc.png')
#     plt.show()

#     # Plot training & validation loss values
#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.savefig('figures/final_model_loss.png')
#     plt.show()
    
#   if IS_SUBMITTING:
#     produceSubmission(model)

if __name__=='__main__':
    main()
