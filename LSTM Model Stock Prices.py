#!/usr/bin/env python
# coding: utf-8

# ## Apple Stock Price Prediction Using LSTM

# ### Import Section

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from math import sqrt
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM, Dense


# In[2]:


df = pd.read_excel(r'B:\Project_Data\AAPL.xlsx')
df.head()


# ## We check whether there is missing value in our data

# In[4]:


df.isnull().sum()


# # Data Preprocessing

# ## Make the date column as index of our data

# In[6]:


df=df.set_index('Date')


# In[7]:


df.head()


# ## Plotting Graphs

# ### Close Prices over years

# In[8]:


plt.plot(df.index, df["Close"])

plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.xticks(rotation=45)

plt.show()


# ### Trading Volumes Over Years

# In[9]:


plt.plot(df.index, df["Volume"])

plt.title('Trading Volumes Over Years')
plt.xlabel('Date')
plt.ylabel('Volume')

plt.xticks(rotation=45)

plt.show()


# ### Index Movement of Every Stock Feature

# In[10]:


my_values = df.values
groups = [0, 1, 2, 3, 4, 5]

ind = 1

plt.figure(figsize=(20,25))
for group in groups:
    plt.subplot(len(groups), 1, ind)
    plt.plot(my_values[:, group])
    plt.title(df.columns[group], y=0.5, loc='right')
    ind = ind + 1
plt.show()


# ## Converting Series Data to Supervised Format.
# Here we will shift our data features 1 hour before the current index. Using these features, we are going to forecast the stock price for the next day (1 day after the current price). This function is very important and useful tho!

# In[11]:


# convert series to supervised
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[12]:


values = df.values
values = values.astype('float32')


# In[13]:


values


# ## Normalization of the Data

# In[14]:


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
df_new = series_to_supervised(scaled, 1, 1)
print(df_new.head())


# #### For the current price target (forecast price), we are only going to use closed price.

# In[15]:


df_new.drop(df_new.columns[[8,9,10,11]], axis=1, inplace=True)
print(df_new.head())


# In[16]:


df_new.shape


# ### Data Splitting and Reshaping
# We are going to use the past 3 years of data as training set. We also reshape our input data (X) to 3 dimensions format because it is needed for our LSTM Model with shaping format namely [samples, timesteps, features].

# #### Splitting into train and test sets

# In[17]:


values = df_new.values
n_train_days= 365 * 3 #3 years
train = values[:n_train_days, :]
test = values[n_train_days: , :]


# #### Splitting into Inputs & Outputs

# In[18]:


train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]


# #### Reshaping Input to be 3D (samples, timesteps, features)

# In[19]:


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# ## Forecasting Modelling Using LSTM Model

# ## LSTM Network Design and Training
# We will define the LSTM with 50 neurons in the first and second hidden layer, 1 dropout layer to make sure model doesn't overfit, and 1 neuron in the output layer for predicting stock price. The input shape will be 1 time step with 11 features.
# We will use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.

# In[20]:


# design network
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ### BIGGER LOSS IN TRAINING SAMPLE

# ## Model Evaluation

# In[21]:


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# In[22]:


# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat[:, :-1])
inv_yhat = inv_yhat[:,0]


# In[23]:


# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y[:, :-1])
inv_y = inv_y[:, 0]


# In[24]:


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# The RMSE is quite small. Our forecasting model has a good performance while forcasting test data.

# ## Actual and Forecasted Price Comparision

# In[25]:


result_comp = pd.DataFrame(list(zip(inv_y, inv_yhat)),
               columns =['Close_Price_Real', 'Close_Price_Forecast'])


# In[26]:


result_comp.head()


# In[27]:


plt.figure(figsize=(21,13))
plt.plot(result_comp['Close_Price_Real'],label='actual',color='red')
plt.plot(result_comp['Close_Price_Forecast'],label='forecast',color='blue')
plt.title('Comparasion Between Actual and Forecasted Closing Price of Apple Stock Using LSTM Model',fontsize=20)
plt.legend()


# The actual and forecasted or predicted price are quite simillar. We have proofed that LSTM model is able to be developed for stock price forecasting and it has capability of forecasting 1 step ahead data based on the past data (1 day before).
