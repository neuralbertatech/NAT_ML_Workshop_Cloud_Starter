import sys
from io import StringIO
from scipy import signal
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# conditional import
if sys.platform == 'win32':
    pass
else:
    from google.cloud import storage

##############################
if sys.platform == 'win32':
    def grab_data():
        data = pd.read_csv('eeg1.csv', skiprows=[0], header=None)
    
        return data
else:    
    def grab_data():
      storage_cli = storage.Client()
      bucket = storage_cli.get_bucket('ml-workshop-123')
    
      blob_oddball_path = bucket.get_blob('eeg1.csv')
      oddball_data = blob_oddball_path.download_as_string()
    
      raw_data_string = oddball_data.decode("utf-8") 
      raw_data = np.genfromtxt(StringIO(raw_data_string), delimiter=",",skip_header=1,dtype=None)
      data = pd.DataFrame(data=raw_data)
    
      return data

##############################
def get_labeled_dataset(data):

    print(data.head())

    data.columns = ['chan_1','chan_2','chan_3','chan_4','chan_5','chan_6','chan_7','chan_8','chan_9','chan_10','chan_11','chan_12','chan_13','chan_14','chan_15','chan_16','trig','timestamp']
    
    print(data.head())
    
    targ_trigs = data[(data['trig'] == 2) | (data['trig'] == 1002)].index
    
    tmin = -25
    tmax = 225
    X=[]
    stfts = []
    y = []
    
    for i in range(len(targ_trigs)):
    
        label = data.iloc[targ_trigs[i]]['trig']
        chunk = data.iloc[targ_trigs[i]+tmin:targ_trigs[i]+tmax].drop(['trig','timestamp'],axis=1).values
        f, t, Zxx = signal.stft(chunk.T,fs=256,nperseg=64,nfft=256)
        
        amplitude = 2*np.abs(Zxx)
        power = amplitude**2
        
        start = np.max(np.where(t<0.5))
        baseline = np.expand_dims(np.mean(power[:,0:start],1),axis=1)
        decible = 10*np.log10(np.divide(power,baseline))
        
        X.append(chunk[128:])
        y.append(label)
        stfts.append(decible[:,4:35,start:])
        
    y = np.array(y)
    stfts = np.array(stfts)
    t=t[start:]-t[start]
    f=f[4:35]
    X = np.array(X)
    
    return X, y

##############################
def create_model(input_shape = (122,16) ,dropout=.25): #6,31,4

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
    
    return eeg_model

##############################
def train_and_test(X,y,batch_size = 64,epochs = 200,test_size=0.2):
  model = create_model(dropout=0.5)
  X_train,X_val, y_train,y_val = train_test_split(X, y, test_size=test_size)
  history = model.fit(X_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True)

  return history, model

##############################
if __name__ == '__main__':    
    data = grab_data()
    X,y = get_labeled_dataset(data)    
    history, model = train_and_test(X,y,batch_size=128)
    model.save('my_model')
