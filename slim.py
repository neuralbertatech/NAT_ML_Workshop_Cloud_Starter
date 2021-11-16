# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:57:16 2021

@author: eredm
"""

#%% Imports
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Get data
eeg_data = pd.read_csv('eeg1.csv', skiprows=[0], header=None)


#%% Define fft 

def sliding_window_fft(chunk, win_size=50, step_size = 10, freq_low = 4, freq_high=38):
  '''
  Input:Chunk of eeg data (250 sample by 16 channel array (2 seconds of data))

  slides over chunk of eeg data with a 50 sample window size (shape (50,16)) (0.4 seconds)
  by steps of 7. 
  use this window to calculate the fft across time on all channels and select
  only frequencies between 4hz and 38hz
  add sliding window ffts to X and return X as training data

  output: list of 46 (7 frequency by 16 channel array)
  '''
  chunk_size = len(chunk)
  X = []

  #iterate over indices where i is the start of the window and i+win_size is the end
  for i in range(0, chunk_size - win_size + step_size, step_size):
    #get window

    win = chunk[i:i+win_size]
    #get fft of window for input data
    fft_data = np.absolute(np.fft.rfft(win,axis=0))
    fft_freqs = np.fft.rfftfreq(win.shape[0], 1.0 / 125) 

    relevant_inds = np.where((fft_freqs >= freq_low) & (fft_freqs <= freq_high))[0]
    relevant_data = fft_data[relevant_inds]

    X.append(relevant_data)
  return X



#%%

def get_labeled_dataset(data):
  '''
  Input: pandas dataframe of labeled eeg data

  find each time t in the data where a stimulus is presented that has a valence
  score lower than neg_thresh or higher than pos_thresh (to ignore any middle valued valences
  since we're doing binary classification)

  The image was presented for 2.5 seconds (625 samples). Since it takes some 
  time for the brain to respond to the stimuli, we will take the eeg data from 
  time t-0.2 seconds to t+1.8 seconds (250 samples)

  Each chunk is processed by sliding_window_fft to turn a 500 sample by 16 channel 
  time domain array, to 46 shape (7,16) frequency domain arrays. These are all labeled
  with 1 or 0 depending on the valence score for the 500 sample chunk. 

  output: ~1000+ labeled training/testing samples

  '''
  tmin = -25
  tmax = 225
  X = []
  y = []
  
  data.columns = ['chan_1','chan_2','chan_3','chan_4','chan_5','chan_6','chan_7','chan_8','chan_9','chan_10','chan_11','chan_12','chan_13','chan_14','chan_15','chan_16','accel_1','accel_2','accel_3','timestamp','trig','junk']

  #get indices where valence is less than or eq neg_thresh to label as negative (0)
  std_inds = data[(data['trig'] == 2)].index
  #get indices where valence is greater than or eq pos_thresh to label as positive (1)
  odd_inds = data[(data['trig'] != 1002)].index

  for i in std_inds:
    #get chunk of eeg from i+tmin to i+tmax, i is when the std stim was presented
    chunk = data.iloc[i+tmin:i+tmax].drop(['accel_1','accel_2','accel_3','timestamp','trig','junk'],axis=1).values
    #do a sliding window over the chunk and do an fft many more labeled samples
    fft_chunks = sliding_window_fft(chunk)
    labels = [np.array([1])] * len(fft_chunks)

    X.extend(fft_chunks)
    y.extend(labels)

  for i in odd_inds:
    #get chunk of eeg from i+tmin to i+tmax, i is when the oddball stim was presented
    chunk = data.iloc[i+tmin:i+tmax].drop(['accel_1','accel_2','accel_3','timestamp','trig','junk'],axis=1).values
    #do a sliding window over the chunk and do an fft many more labeled samples
    fft_chunks = sliding_window_fft(chunk)
    labels = [np.array([0])]*len(fft_chunks)

    X.extend(fft_chunks)
    y.extend(labels)

  return np.array(X), np.array(y)

#%%
x, y = get_labeled_dataset(eeg_data)

# convert y to one hot encoding
one_hot_y = np.zeros((y.size, 2))
one_hot_y[np.arange(y.size), y.reshape(-1)] = 1

eeg_x_train, eeg_x_test, eeg_y_train, eeg_y_test = train_test_split(x, one_hot_y, test_size=.2)

#%%
def create_model(input_shape = (7,16) ,dropout=.25):

    eeg_model = Sequential()
    
    eeg_model.add(Flatten(input_shape=input_shape)) #flatten 7 freqs by 16 channels array to (112,) shape array
    
    eeg_model.add(Dense(224))
    eeg_model.add(Activation('relu'))
    eeg_model.add(Dropout(dropout))
    
    eeg_model.add(Dense(448))
    eeg_model.add(Activation('relu'))
    eeg_model.add(Dropout(dropout))
    
    eeg_model.add(Dense(224))
    eeg_model.add(Activation('relu'))
    eeg_model.add(Dropout(dropout))
    
    eeg_model.add(Dense(112))
    eeg_model.add(Activation('relu'))
    eeg_model.add(Dropout(dropout))
    
    eeg_model.add(Dense(56))
    eeg_model.add(Activation('relu'))
    eeg_model.add(Dropout(dropout))
    
    eeg_model.add(Dense(1))
    eeg_model.add(Activation('sigmoid'))
    
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                  epsilon=10e-8, decay=0.0, amsgrad=False)
    
    eeg_model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    return model


#%%

def train_and_test(X,y,batch_size = 64,epochs = 200,test_size=0.2):

  model = create_model(dropout=0.5)

  X_train,X_val, y_train,y_val = train_test_split(X, y, test_size=test_size)
  history = model.fit(X_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True)

  return history,model

#%%

X,y = get_labeled_dataset(eeg_data)

#%%

history,model = train_and_test(X,y,batch_size=128)