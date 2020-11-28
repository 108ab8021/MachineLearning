#!/usr/bin/env python
# coding: utf-8

# In[88]:


import os
import seaborn as sns
import numpy as np
import pandas as pd
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *


# In[89]:


train_data = pd.read_csv('train-v3.csv')
test_data = pd.read_csv('test-v3.csv')
valid_data = pd.read_csv('valid-v3.csv')


# In[90]:


#處理訓練檔案,用X_train預測Y_train
X_train = train_data.drop(["id","price"],axis=1).values #去除id及答案(price)
Y_train = train_data["price"].values #保留答案(price)

#處理驗證檔案(一般狀況是從訓練檔案分割出來),用X_valid預測Y_valid
X_valid = valid_data.drop(["id","price"],axis=1).values #去除id及答案(price)
Y_valid = valid_data["price"].values #保留答案(price)

#處理測試檔案,將不需要的欄位drop掉,只保留要預測的欄位
X_test = test_data.drop(["id"],axis=1).values


# In[91]:


#資料正規化
scaler = StandardScaler().fit(X_train)
X_train_normal = scale(X_train)
X_valid_normal = scaler.transform(X_valid)
X_test_normal = scaler.transform(X_test)


# In[114]:


model = Sequential()
model.add(layers.Dense(40,kernel_initializer = 'normal',
             activation = 'relu',input_dim = X_train.shape[1]))

    
model.add(layers.Dense(80, kernel_initializer = 'normal', 
                        activation = 'relu'))

    
model.add(layers.Dense(128, kernel_initializer = 'normal', 
                        activation = 'relu'))

    
model.add(layers.Dense(80, kernel_initializer = 'normal', 
                           activation = 'relu'))

    
model.add(layers.Dense(40, kernel_initializer = 'normal', 
                           activation = 'relu'))

    
model.add(layers.Dense(64, kernel_initializer = 'normal', activation = 'relu'))

    
model.add(layers.Dense(32, kernel_initializer = 'normal', activation = 'relu'))

model.add(layers.Dense(80, kernel_initializer = 'normal', 
                        activation = 'relu'))

    
model.add(layers.Dense(128, kernel_initializer = 'normal', 
                        activation = 'relu'))

    
model.add(layers.Dense(80, kernel_initializer = 'normal', 
                           activation = 'relu'))

    
model.add(layers.Dense(40, kernel_initializer = 'normal', 
                           activation = 'relu'))

    
model.add(layers.Dense(64, kernel_initializer = 'normal', activation = 'relu'))

    
model.add(layers.Dense(32, kernel_initializer = 'normal', activation = 'relu'))


    
model.add(layers.Dense(1, kernel_initializer = 'normal',
                          activation = 'relu'))
    
#adam = optimizers.Adam(lr=0.001)
    #lr學習率
model.compile(optimizer = adam, loss = 'mae')


# In[115]:


epochs = 234
batch_size = 32


file_name= str(epochs)+"_"+str(batch_size)
TB = TensorBoard(log_dir = "logs/"+file_name, histogram_freq = 0)
model.fit(X_train_normal, Y_train,
                    validation_data = (X_valid_normal, Y_valid),
                    epochs =epochs,batch_size=batch_size,callbacks = [TB],verbose=1)
model.save(file_name+".h5")


# In[116]:


pred = model.predict(X_test_normal)


# In[117]:


with open('predict.csv', 'w') as f: #開啟一個檔案，house_predict.csv是名字；w是寫入
    f.write('id,price\n') #寫入最上方的列，並用\n往下一列
    for i in range(len(pred)): #len(pred)看整個test有多長，然後用for去跑全部
        f.write(str(i+1) + ',' + str(float(pred[i])) + '\n')


# In[ ]:





# In[49]:





# In[ ]:





# In[ ]:




