
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time
import pyautogui

# ============================================================================
SHUTDOWN = False
SCREENSHOT = False
# ============================================================================

if(SHUTDOWN):
    input('WARNING: System will SHUTDOWN after the program conclusion.')

# Resize method: Nearest Neighbour
width = 224
height = 224

batch_size = 20
epochs_2phase = 100

folder = 'ISBI2016_ISIC_Part3_Training_Data'

# ============================================================================

try:

    from keras.applications import VGG16
    
    conv_base = VGG16(weights='imagenet',
                      include_top=True, # Include or not the fully connected layers.
                      input_shape=(width, height, 3))
    
    conv_base.layers.pop()
    # conv_base.summary()
    
    conv_base.trainable = False    
    
    # conv_base.summary()
    
    from keras import models
    from keras import layers
    # from keras.layers import Dropout
    
    model = models.Sequential()
    model.add(conv_base)    
    model.add(layers.Dense(1, activation='sigmoid')) 
    
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'vgg16':
            layer.trainable = True
            for sublayer in layer.layers:
                if sublayer.name == 'fc1': # Primeira camada densa
                    set_trainable = True
                if set_trainable:
                    sublayer.trainable = True
                else:
                    sublayer.trainable = False
                    
    # conv_base.summary()    
    model.summary()
    
    print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
    
    from keras.preprocessing.image import ImageDataGenerator
    from keras import optimizers
    
    full_datagen = ImageDataGenerator(
          rescale=1./255,
          #rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          validation_split = 0.2, # 20% validation
          horizontal_flip=True
          )
    
    train_generator = full_datagen.flow_from_directory(
            folder,
            target_size=(width, height),
            batch_size=batch_size,
            subset = 'training',
            class_mode='binary')
    
    validation_generator = full_datagen.flow_from_directory(
            folder,
            target_size=(width, height),
            batch_size=batch_size,
            subset = 'validation',
            shuffle=False,
            class_mode='binary')
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.001), # High learning rate
                  metrics=['accuracy'])
    
    from tensorflow.python.client import device_lib
    print('\n')
    print(device_lib.list_local_devices())
    
    history = model.fit_generator(
           train_generator,
           steps_per_epoch=900 // batch_size+1,
           epochs=20,
           validation_data=validation_generator,
           validation_steps=180 // batch_size+1,
           #verbose=2)
           )
    
    # Fine-tuning
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'vgg16':
            layer.trainable = True
            for sublayer in layer.layers:
                if sublayer.name == 'block5_conv3':
                    set_trainable = True
                if set_trainable:
                    sublayer.trainable = True
                else:
                    sublayer.trainable = False
     
    model.summary()
    
# =============================================================================
#     for layer in model.layers:
#         if layer.name == 'vgg16':
#             for sublayer in layer.layers:
#                 print(sublayer.name + str(sublayer.trainable))
# =============================================================================

    if(SCREENSHOT):
        time.sleep(2)
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(r'C:\Users\Pichau\Desktop\model_summary.png')
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5), # Much lower learning rate (fine-tuning)
                  metrics=['accuracy'])
    
    history = model.fit_generator(
           train_generator,
           steps_per_epoch=900 // batch_size+1,
           epochs=epochs_2phase,
           validation_data=validation_generator,
           validation_steps=180 // batch_size+1,
           #verbose=2)
           )
    
    
    # Exponential moving averages
    def smooth_curve(points, factor=0.8):
      smoothed_points = []
      for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
      return smoothed_points
    
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    # Plotting 
    
    import matplotlib.pyplot as plt
    
    # Smoothed
    # =============================================================================
    # plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
    # plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # 
    # plt.figure()
    # 
    # plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
    # plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # 
    # plt.show()
    # =============================================================================
    
    # Normal
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.show()
    
    if(SCREENSHOT):
        time.sleep(5)
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(r'C:\Users\Pichau\Desktop\graph1.png')
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
    if(SCREENSHOT):
        time.sleep(5)        
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(r'C:\Users\Pichau\Desktop\graph2.png')
    
    # =============================================================================
    # myScreenshot = pyautogui.screenshot()
    # myScreenshot.save(r'C:\Users\Pichau\Desktop\graphs.png')
    # =============================================================================
    
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Confusion Matrix and Classification Report
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('ISBI2016_ISIC_Part3_Test_Data',
                                                target_size = (width, height),
                                                batch_size = batch_size,
                                                class_mode = 'binary')
    
    Y_pred = model.predict_generator(test_set, 379 // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_set.classes, y_pred))
    print('Classification Report')
    target_names = ['Benign', 'Malignant']
    print(classification_report(test_set.classes, y_pred, target_names=target_names))
    
    if(SCREENSHOT):
        time.sleep(3)        
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(r'C:\Users\Pichau\Desktop\CM.png')
    
except Exception as e:
    print(e)
    if(SCREENSHOT):    
        time.sleep(1)
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save(r'C:\Users\Pichau\Desktop\ERROR.png')

if SHUTDOWN:   
    import os
    os.system('shutdown -s -f')

