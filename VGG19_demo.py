#Importing Libaries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pathlib
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#Variable Setup

class_names = ['NonDemented', 'Demented']
num_classes = len(class_names)
batch_size = 32
input_shape = (208, 176, 3)



#Loading Training, Validation and Testing Data

#data_dir = pathlib.Path('path/to/training/data') #model loaded below is trained on 'AugmentedAlzheimerDataset' found at https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
data_dir_test = pathlib.Path('./TestSamples')

# only normalization is on (don't need to flip test images)
simple_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# load the test images just as they are using simple datagen
test_generator = simple_datagen.flow_from_directory(
        data_dir_test,
        class_mode='categorical',
        shuffle=False,
        batch_size=batch_size,
        target_size=input_shape[0:2])


# Loading and Evaluating of Trained Model

model = keras.models.load_model("./vgg19_300.keras")

print("Evaluation of Model:\n")
score = model.evaluate(test_generator)



#Predicting Single Image 

img = cv2.imread("./TestSamples/Demented/DementedExample.jpg")
# Resize to match the model's input size
img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))  

# Adding dimension for consistency
img1 = np.reshape(img_resized, (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))  

# Apply the preprocessing function used during training
img_rescaled = img1.astype(np.float32) / 255.0  

# Make the prediction
prediction = model.predict(img_rescaled)

if (prediction[0][0] >= prediction[0][1]):
    print('Patient has dementia.')
else:
    print('Patient does not has dementia.')
    
#Confidence of prediction [[demented percent, nondemented percent]]
print(prediction)



#Producing Confusion Matrix

target_names = ['NonDemented', 'Demented']

# Use the test generator for predictions
Y_pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

# Use the test generator for true labels
y_true = test_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)
disp.plot()
plt.show()

# Classification Report
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=target_names))
