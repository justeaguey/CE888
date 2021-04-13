import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import DenseNet121
from keras import callbacks

#parameters for our model
batch_size = 32
epochs = 25
image_size = 254

#data generator to feed through batches of training images, with a 20% validation split and image augmentation
training_datagenerator = ImageDataGenerator(rescale = 1./255,
    validation_split = 0.2,
    brightness_range = [0.6, 0.8])

#data generator to feed through batches of test images
test_datagenerator = ImageDataGenerator(rescale=1./255)

#iterators to feed our model with the images
train_iterator = training_datagenerator.flow_from_directory('C:\\Users\\juste\\Documents\\Masters\\Data Science\\Proposal\\Training\\Training\\', class_mode='binary', target_size=(image_size, image_size), batch_size = batch_size, shuffle=True, subset = 'training')

validation_iterator = training_datagenerator.flow_from_directory('C:\\Users\\juste\\Documents\\Masters\\Data Science\\Proposal\\Training\\Training\\', class_mode='binary', target_size=(image_size, image_size), batch_size = batch_size, shuffle=True, subset = 'validation')

test_iterator = test_datagenerator.flow_from_directory('C:\\Users\\juste\\Documents\\Masters\\Data Science\\Proposal\\Test\\Test\\', class_mode='binary', target_size=(image_size, image_size), batch_size = batch_size, shuffle=False)

#monitors the validation loss value and will stop training when there is an increase
earlystop = callbacks.EarlyStopping(monitor = "val_loss", 
                                        mode = "min", patience = 5, 
                                        restore_best_weights = True)

#load the Xception model
XceptionModel = tf.keras.applications.Xception(input_shape=(image_size,image_size,3), include_top = False, weights = "imagenet")

#allow parameters in the model to be trainable
XceptionModel.trainable = True

#add custom output layers to the model
transfer_model = tf.keras.Sequential([XceptionModel,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(1, activation="sigmoid")                                     
                                ])

#compile the model with an adam optimiser
optim = Adam(lr = 0.00005)
transfer_model.compile(optimizer = optim,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#load the DenseNet model
DenseNetModel = tf.keras.applications.DenseNet121(input_shape=(image_size,image_size,3), include_top = False)

#allow parameters in the model to be trainable
DenseNetModel.trainable = True

#add custom output layers to the model
transfer_model2 = tf.keras.Sequential([DenseNetModel,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(1, activation="sigmoid")                                     
                                ])

#compile the model with an adam optimiser
optim = Adam(lr = 0.00005)
transfer_model2.compile(optimizer = optim,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Enviroment ready, please use functions XceptionTraining() or DenseNetTraining() to train a model, and then use XceptionTesting() or DenseNetTesting() to evaluate the trained model')

def XceptionTraining():
    train_iterator.reset()
    validation_iterator.reset()
    #train the model
    transfer_outputs = transfer_model.fit(train_iterator, 
                    epochs = epochs, steps_per_epoch = 100,
                    validation_data = validation_iterator,
                    validation_steps = 20,
                    #callbacks = [earlystop], #stops the function before it starts to over fit
                    class_weight = {0: 0.73, 1: 1.27}) #encourages model to pay more attention to the 'No-Fire' class by weighting its loss higher
    
    #plot accuracy and loss for each epoch
    transfer_accuracy = transfer_outputs.history['accuracy']
    transfer_validation_accuarcy = transfer_outputs.history['val_accuracy']
    transfer_loss = transfer_outputs.history['loss']
    transfer_validation_loss = transfer_outputs.history['val_loss']

    epochs_len = len(transfer_outputs.history['loss']) #displays number of runs the model trained for before stopping

    plt.figure(figsize=(18, 18))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs_len + 1), transfer_accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs_len + 1), transfer_validation_accuarcy, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Training/Validation Accuracy for Xception')

    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs_len + 1), transfer_loss, label='Training Loss')
    plt.plot(range(1, epochs_len + 1), transfer_validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss for Xception')
    plt.show()


def DenseNetTraining():
    train_iterator.reset()
    validation_iterator.reset()
    #train the model
    transfer_outputs2 = transfer_model2.fit(train_iterator, 
                    epochs = epochs, steps_per_epoch = 100,
                    validation_data = validation_iterator,
                    validation_steps = 20,
                    #callbacks = [earlystop], #stops the function before it starts to over fit
                    class_weight = {0: 0.73, 1: 1.27}) #encourages model to pay more attention to the 'No-Fire' class by weighting its loss higher
    
    #plot accuracy and loss for each epoch
    transfer_accuracy = transfer_outputs2.history['accuracy']
    transfer_validation_accuarcy = transfer_outputs2.history['val_accuracy']
    transfer_loss = transfer_outputs2.history['loss']
    transfer_validation_loss = transfer_outputs2.history['val_loss']

    epochs_len = len(transfer_outputs2.history['loss']) #displays number of runs the model trained for before stopping

    plt.figure(figsize=(18, 18))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs_len + 1), transfer_accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs_len + 1), transfer_validation_accuarcy, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Training/Validation Accuracy for DenseNet')

    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs_len + 1), transfer_loss, label='Training Loss')
    plt.plot(range(1, epochs_len + 1), transfer_validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss for DenseNet')
    plt.show()

def XceptionTesting():
    #reset the test iterator before evaluation
    test_iterator.reset()
    
    #calculate loss and accuracy of the model on the test set
    transfer_test_loss, transfer_test_accuarcy = transfer_model.evaluate(test_iterator, steps = test_iterator.samples//batch_size)

    print('Loss: ' + str(transfer_test_loss))
    print('Accuracy: ' + str(transfer_test_accuarcy))
    
    #reset the test iterator before predictions
    test_iterator.reset()

    #perform predictions on the test set with the trained model
    probabilities = transfer_model.predict(test_iterator)

    #convert the predictions into the binary class values
    predicted_class_indices = np.where(probabilities > 0.5, 1, 0)

    #true labels of the test set
    labels=(test_iterator.classes)

    print(classification_report(labels, predicted_class_indices))

    conf_matrix = confusion_matrix(labels, predicted_class_indices)
    print('Confusion Matrix:')
    print(conf_matrix)

def DenseNetTesting():
    #reset the test iterator before evaluation
    test_iterator.reset()
    
    #calculate loss and accuracy of the model on the test set
    transfer_test_loss, transfer_test_accuarcy = transfer_model2.evaluate(test_iterator, steps = test_iterator.samples//batch_size)

    print('Loss: ' + str(transfer_test_loss))
    print('Accuracy: ' + str(transfer_test_accuarcy))
    
    #reset the test iterator before predictions
    test_iterator.reset()

    #perform predictions on the test set with the trained model
    probabilities2 = transfer_model2.predict(test_iterator)

    #convert the predictions into the binary class values
    predicted_class_indices2 = np.where(probabilities2 > 0.5, 1, 0)

    #true labels of the test set
    labels=(test_iterator.classes)

    print(classification_report(labels, predicted_class_indices2))

    conf_matrix2 = confusion_matrix(labels, predicted_class_indices2)
    print('Confusion Matrix:')
    print(conf_matrix2)
