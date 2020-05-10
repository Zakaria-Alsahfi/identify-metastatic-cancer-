#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:26:59 2020

@author: zakariaalsahfi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from glob import glob 
from skimage.io import imread #read images from files
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
#from numpy.random import seed
#seed(101)
#------------------------------------------------------------------------------
# Output files
MODEL_FILE = "model.h5"
ROC_PLOT_FILE = "roc1.png"
MODEL_PLOT_FILE = "model_plot1.png"
TRAINING_PLOT_FILE = "training1.png"
VALIDATION_PLOT_FILE = "validation1.png"
CONFUSION_MATRIX_FILE = "cmatrix1.png"
CLASSIFICATION_REPORT_FILE = 'Classification_Report.png'
#------------------------------------------------------------------------------
os.chdir("/Users/zakariaalsahfi/Documents/Maryvill/SP20/DSCI 419 DEEP LEARNING/DL-FinalProject/data")
base_tile_dir = 'train'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[1].split(".")[0])
labels = pd.read_csv("train_labels.csv")
df_data = df.merge(labels, on = "id")
df.head(3)
#------------------------------------------------------------------------------
df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# removing this image because it's black
df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df_data.head(3)
#------------------------------------------------------------------------------
SAMPLE_SIZE = 80000 # load 80k negative examples
# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
# filter out class 1
df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

#------------------------------------------------------------------------------
# concat the dataframes
df_data = shuffle(pd.concat([df_0, df_1], axis=0).reset_index(drop=True))
df_data['label'].value_counts()
#------------------------------------------------------------------------------

# train_test_split # stratify=y creates a balanced validation set.
y = df_data['label']
df_train, df_val = train_test_split(df_data, test_size=0.10, 
                                    random_state=101, stratify=y)
#------------------------------------------------------------------------------
# Create directories
train_path = 'base_dir/train'
valid_path = 'base_dir/valid'
test_path = '../data/test'
for fold in [train_path, valid_path]:
    for subf in ["0", "1"]:
        os.makedirs(os.path.join(fold, subf))
#------------------------------------------------------------------------------
# Set the id as the index in df_data
df_data.set_index('id', inplace=True)
df_data.head()
#------------------------------------------------------------------------------
for images_and_path in [(df_train, train_path), (df_val, valid_path)]:
    images = images_and_path[0]
    path = images_and_path[1]
    for image in images['id'].values:
        file_name = image + '.tif'
        label = str(df_data.loc[image,'label'])
        destination = os.path.join(path, label, file_name)
        if not os.path.exists(destination):
            source = os.path.join('../data/train', file_name)
            shutil.copyfile(source, destination)    
#------------------------------------------------------------------------------
IMAGE_SIZE = 96
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


datagen = ImageDataGenerator(rescale= 1 / 255.0, rotation_range=20,zoom_range=0.05,
                             width_shift_range=0.05, height_shift_range=0.05,
                             shear_range=0.05, horizontal_flip=True, 
                             vertical_flip=True, fill_mode="nearest")

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
#----------------------------------------------------------------------------
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

# Compile the model
model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()

#------------------------------------------------------------------------------
plot_model(model,
           to_file=MODEL_PLOT_FILE,
           show_shapes=True,
           show_layer_names=True)
#------------------------------------------------------------------------------
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)
history = model.fit_generator(train_gen, 
                              steps_per_epoch=train_steps, 
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              epochs=10,
                              callbacks=[reducel, earlystopper,
                                         ModelCheckpoint(MODEL_FILE,
                                                         monitor='val_acc',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         mode='max')])
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
epochs = [i for i in range(0, len(history.history['loss']))]

# Plot training & validation accuracy values
plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig(TRAINING_PLOT_FILE, bbox_inches='tight')
plt.close()

# Plot training & validation loss values
plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.savefig(VALIDATION_PLOT_FILE, bbox_inches='tight')
plt.close()
#------------------------------------------------------------------------------
# Here the best epoch will be used.
model.load_weights(MODEL_FILE)

val_loss, val_acc = \
model.evaluate_generator(test_gen, steps=len(df_val))
print('val_loss:', val_loss)
print('val_acc:', val_acc)

# make a prediction
y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
print(y_pred_keras.shape)

# Put the predictions into a dataframe.
df_preds = pd.DataFrame(y_pred_keras, columns=['no_tumor', 'has_tumor'])
df_preds.head()

# Get the true labels
y_true = test_gen.classes
# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor']
print('ROC AUC Score = ',roc_auc_score(y_true, y_pred))


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC = {:.2f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristics (ROC) Curve')
plt.legend(loc='best')
plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
plt.close()
#------------------------------------------------------------------------------
#Create a Confusion Matrix
# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = y_pred_keras.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred_binary)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                               cmap = 'Dark2')
plt.savefig(CONFUSION_MATRIX_FILE, bbox_inches='tight')
plt.close()
#------------------------------------------------------------------------------
# Create a Classification Report
# Generate a classification report
report = classification_report(y_true, y_pred_binary, target_names = ['no_tumor', 'has_tumor'])
report.to_csv(CLASSIFICATION_REPORT_FILE, sep='\t', index=False)
print(report)
#From the confusion matrix and classification report we see that our model 
# is equally good at detecting both classes.
#------------------------------------------------------------------------------
# Put the predictions into a dataframe
df_preds = pd.DataFrame(y_pred_keras, columns=['no_tumor', 'has_tumor'])
df_preds.head()
#------------------------------------------------------------------------------
# This outputs the file names in the sequence in which the generator processed the test images.
test_filenames = test_gen.filenames

# add the filenames to the dataframe
df_preds['file_names'] = test_filenames

# Create an id column
# A file name now has tif format: 
# This function will extract the id
def extract_id(x):
    
    # split into a list
    a = x.split('/')
    # split into a list
    b = a[1].split('.')
    extracted_id = b[0]
    
    return extracted_id

df_preds['id'] = df_preds['file_names'].apply(extract_id)

df_preds.head()
# Get the predicted labels.
# We were asked to predict a probability that the image has tumor tissue
y_pred = df_preds['has_tumor']

# get the id column
image_id = df_preds['id']
submission = pd.DataFrame({'id':image_id, 
                           'label':y_pred, 
                          }).set_index('id')

submission.to_csv('submission.csv', columns=['label'])
