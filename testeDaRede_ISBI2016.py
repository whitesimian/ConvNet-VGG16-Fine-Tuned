
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

directory = os.getcwd()

# 8 classes

# Resize method: Nearest Neighbour
width = 224
height = 224

batch_size = 20
epochs = 120
epochs_before = 20
validation_split = 0.2
lr_bef = 1e-3
lr_final = 1e-4

train_folder = 'dataset_final_training'
# test_folder = 'dataset_final_test'
test_folder = 'dataset_final_training'
save_file = '../../Desktop/Results/VGGs e EfficientNet/'
model_name = None

def get_amount_of_images(path):
    imagesQt = 0
    curFolder = directory + '/' + path
    for folder in os.listdir(curFolder):
        imagesQt += len(os.listdir(curFolder + '/' + folder))
    return imagesQt

# ============================================================================
    
for i in range(2):
    
    from keras.applications import VGG19
    from keras.applications import VGG16
    
    conv_base = None
    
    if i == 0:
        print('=========== VGG19:')
        conv_base = VGG19(weights='imagenet',
                          include_top=False, # Include or not the fully connected layers.
                          input_shape=(width, height, 3))
        save_file += 'VGG19/'
        model_name = 'vgg19'
    else:
        print('=========== VGG16:')
        conv_base = VGG16(weights='imagenet',
                          include_top=False, # Include or not the fully connected layers.
                          input_shape=(width, height, 3))
        save_file += 'VGG16/'
        model_name = 'vgg16'
        
    
    conv_base.trainable = False    
    
    # conv_base.summary()
    
    from keras import models
    from keras import layers
    
    model = models.Sequential()
    model.add(conv_base)
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 128, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units = len(os.listdir(directory + '/' + train_folder)), activation = 'softmax'))
    # model.summary()
    
    from keras.preprocessing.image import ImageDataGenerator
    from keras import optimizers
    
    full_datagen = ImageDataGenerator(
          rescale=1./255,
          #rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          # 20% validation
          validation_split = validation_split,
          horizontal_flip=True)
    
    train_generator = full_datagen.flow_from_directory(
            train_folder,
            target_size=(width, height),
            batch_size=batch_size,
            subset = 'training',
            class_mode='categorical')
    
    validation_generator = full_datagen.flow_from_directory(
            train_folder,
            target_size=(width, height),
            batch_size=batch_size,
            subset = 'validation',
            shuffle=False,
            class_mode='categorical')
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr_bef), # High learning Rate
                  metrics=['accuracy'])
    
    
    imagesQt = get_amount_of_images(train_folder)
    
    # Before fine-tune
    history = model.fit_generator(
           train_generator,
           steps_per_epoch=imagesQt // batch_size+1,
           epochs=epochs_before,
           validation_data=validation_generator,
           validation_steps= (imagesQt * validation_split) // batch_size+1,
           #verbose=2
           )
    
    # Fine-tune
    conv_base.trainable = True    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr_final), # Lower learning Rate (fine-tuning)
                  metrics=['accuracy'])
    
    history = model.fit_generator(
           train_generator,
           steps_per_epoch=imagesQt // batch_size+1,
           epochs=epochs,
           validation_data=validation_generator,
           validation_steps= (imagesQt * validation_split) // batch_size+1,
           #verbose=2
           )    
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    # Plotting    
    import matplotlib.pyplot as plt
    
    # Normal
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.savefig(save_file + '1.png')
    plt.show()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig(save_file + '2.png')
    plt.show()
    
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Confusion Matrix and Classification Report
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(test_folder,
                                                target_size = (width, height),
                                                batch_size = batch_size,
                                                class_mode = 'categorical')
    
    
    imagesQt = get_amount_of_images(test_folder)
    
    target_names = []
    for folder in os.listdir(directory + '/' + train_folder):
        target_names.append(folder.capitalize())    
    # target_names = ['Cobblestone', 'Globular', 'Hommogeneous', 'Multicomponent', 'Parallel', 'Reticular', 'Starburst', 'Unspecific']
    
    Y_pred = model.predict_generator(test_set, imagesQt // batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    
    f = open(save_file + 'CM.txt', "a")
    matrix = confusion_matrix(test_set.classes, y_pred)
    f.write('\nConfusion Matrix - Validation\n')
    for row in matrix:
        f.write('[')
        for number in row:
            f.write("{: >4}".format(number))
        f.write(']\n')
    report = classification_report(test_set.classes, y_pred, target_names=target_names)
    f.write('\nClassification Report\n' + report)
    f.close()
    
    model.save(model_name + '.h5')
    
# =============================================================================
#     print('Confusion Matrix')
#     print(confusion_matrix(test_set.classes, y_pred))
#     print('Classification Report')
#     print(classification_report(test_set.classes, y_pred, target_names=target_names))
# =============================================================================

