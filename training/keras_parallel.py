from keras.utils.vis_utils import model_to_dot
import json
import numpy as np

import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import os
import matplotlib.pyplot as plt




# # Setup

# In[3]:





# In[2]:


DATASET_NAME = "dog_breed"
#DATASET_NAME = "leaf"
#DATASET_NAME = "food"
#DATASET_NAME = "bird"
#DATASET_NAME = "fungus"

NUM_GPUS = 4
NUM_EPOCHS = 60
PATH = f"data/{DATASET_NAME}"
train_folder = f"data/{DATASET_NAME}/train"
valid_folder = f"data/{DATASET_NAME}/valid"


# In[5]:





# In[6]:


IMAGE_SIZE=224 
BATCH_SIZE=64 # number of images seen by model at once


# In[7]:





# In[8]:


train_generator = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input
                                     ,rotation_range=45
                                     ,width_shift_range=0.2
                                     ,height_shift_range=0.2
                                     ,shear_range=0.2
                                     ,zoom_range=0.25
                                     ,horizontal_flip=True
                                     ,fill_mode='nearest'
                                  )

train_batches = train_generator.flow_from_directory(
    train_folder, target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=BATCH_SIZE,shuffle=True, seed=13,class_mode='categorical')



valid_generator =  ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)


valid_batches = valid_generator.flow_from_directory(
    valid_folder, target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=BATCH_SIZE)


# In[9]:


NUM_CLASSES = train_batches.num_classes


# In[10]:


base_model = keras.applications.mobilenet.MobileNet()


# In[12]:


base_model.summary()


# In[13]:




# In[14]:


base_model.layers[-6].output


# In[15]:


x = base_model.layers[-6].output
predictions = Dense(NUM_CLASSES, activation='softmax')(x)


# In[16]:


model = Model(inputs=base_model.input, outputs=predictions)


# In[17]:


model.summary()


# In[18]:


for layer in base_model.layers:
    layer.trainable = False


# In[19]:




# In[20]:


model.summary()


# In[21]:


optimizer = RMSprop(lr=0.001, rho=0.9)

#optimizer = Adam(lr=0.001)

# model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[23]:


try:
    model = multi_gpu_model(model, gpus=NUM_GPUS)
except:
    pass
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[24]:


steps_per_epoch = train_batches.n//BATCH_SIZE
validation_steps =valid_batches.n // BATCH_SIZE

# steps_per_epoch = 10
# validation_steps = 3


# In[26]:


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)

tb = keras.callbacks.TensorBoard(log_dir=f'{PATH}/logs', write_graph=True, write_images=True)


# In[ ]:


model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch 
                    , validation_data=valid_batches
                    ,  validation_steps=validation_steps, verbose=2,workers=4 
                    , use_multiprocessing=True
                    , epochs=NUM_EPOCHS
                   # , callbacks = [tb]
                    , callbacks = [reduce_lr, tb]
                   )


# In[5]:


MODEL_SAVE_DIR = f'model_benchmark/{DATASET_NAME}'
shutil.rmtree(MODEL_SAVE_DIR)
os.makedirs(MODEL_SAVE_DIR,exist_ok=True)


# In[ ]:


model.save(f'{MODEL_SAVE_DIR}/model.h5')


# In[4]:


with open(f'{MODEL_SAVE_DIR}/labels.txt', 'w') as file_handler:
    for item in train_batches.class_indices.keys():
        file_handler.write("{}\n".format(item))


# In[3]:


with open(f'{MODEL_SAVE_DIR}/labels.json', 'w') as outfile:
    json.dump(train_batches.class_indices, outfile)

