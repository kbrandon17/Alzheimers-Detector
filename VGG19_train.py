#Importing Libaries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pathlib

#Variable Setup

class_names = ['NonDemented', 'Demented']
num_classes = len(class_names)
batch_size = 32
input_shape = (208, 176, 3)



#Loading Training, Validation and Testing Data

#Below relative path needs to be adjusted to point towards training data (not included in github)
data_dir = pathlib.Path('./TrainingData') #model loaded below is trained on 'AugmentedAlzheimerDataset' found at https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

# horizontal flip and normalization are on (for better training)
datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                       #rotation_range=10,
                                                       rescale=1./255,
                                                       validation_split=0.2
                                                       #brightness_range=(0.8,1.1),
                                                       #zoom_range=0.2,
                                                       #width_shift_range=0.1,
                                                       #height_shift_range=0.1
                                                      )

train_generator = datagen.flow_from_directory(
        data_dir,
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size,
        target_size=input_shape[0:2],
        subset='training')

validation_generator = datagen.flow_from_directory(
        data_dir,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        target_size=input_shape[0:2],
        subset='validation')



#Construct and Display VGG19 Transfer Learning Model Structure

vgg = keras.applications.VGG19(
    include_top=False,
    input_shape=input_shape,
    pooling=max)


for layer in vgg.layers:
    layer.trainable = False

model = keras.models.Sequential([
        keras.Input(input_shape),
        vgg,
        layers.Flatten(),
        layers.Dropout(0.6),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
        ])

model.summary()



#Loading the Model (Optional, to use pre-existing instead of creating with above)

#model = keras.models.load_model("./vgg19_300.keras")



#Training the Model

epochs = 10
model.compile(loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)



#Testing Model and Plot Accuracy Post-Training

plt.title('VGG19 Brain Layer Selection Accuracy (Grayscale)')

plt.plot(history.history['accuracy'], label = "training accuracy")
plt.plot(history.history['val_accuracy'], label = "validation accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.xticks(np.arange(0, 10, 2))
plt.show()



# Saving Trained Model

model.save("./vgg19_300.keras")
