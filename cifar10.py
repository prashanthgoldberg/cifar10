#!/usr/bin/env python
# coding: utf-8

# In[24]:


from keras.layers import Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,Flatten


# In[7]:


from keras.datasets import cifar10


# In[8]:


(trainx,trainy),(testx,testy)=cifar10.load_data()


# In[9]:


import numpy as np


# In[20]:


from keras.utils import to_categorical as t
x1,x2=t(trainy,dtype="float32"),t(testy,dtype="float32")
trainy,testy=x1,x2


# In[23]:


x1,x2=trainx.astype("float32"),testx.astype("float32")
trainx,testx=x1/255.0,x2/255.0


# In[11]:


from keras.models import Sequential


# In[31]:


a=Sequential()
a.add(Conv2D(32,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(32,32,3)))
a.add(Conv2D(32,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(MaxPooling2D(2,2))
a.add(BatchNormalization())
a.add(Dropout(0.2))
a.add(Conv2D(64,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(Conv2D(64,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(MaxPooling2D(2,2))
a.add(BatchNormalization())
a.add(Dropout(0.3))
a.add(Conv2D(128,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(Conv2D(128,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same"))
a.add(MaxPooling2D(2,2))
a.add(BatchNormalization())
a.add(Dropout(0.4))
a.add(Flatten())
a.add(Dense(128,activation="relu",kernel_initializer="he_uniform"))
a.add(Dropout(0.5))
a.add(Dense(10,activation="softmax"))


# In[32]:


from keras.optimizers import SGD
opt=SGD(lr=0.001,momentum=0.9)
a.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[35]:


a.fit(trainx,trainy, epochs=200, batch_size=64, verbose=0)


# In[37]:


str1=a.predict(testx)


# In[51]:


max(str1[2])


# In[50]:


max(testy[2])


# In[55]:


max(str1[3])


# In[53]:


testy[3]


# In[56]:


a.save("cifar10.h5")


# In[ ]:




