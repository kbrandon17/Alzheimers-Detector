# SFU CMPT 353 Project -- Alzheimer Identifier

This project aims to utilize convolution neural networks (CNNs) to develop a classification model capable of discerning whether a patient is affected by Alzheimer’s disease.

## Table of Contents:

1. [Installation](#installation)

2. [Reproducing Project Results](#repro)

### What to Find Where:

```bash
repository
├── TestSamples                     ## Folder of Test Samples
    ├── Demented                    ## Folder of Demented Test Samples
        ├── DementedExample.jpg     ## Sample MRI image of Demented Patient
        ├── ...                     ## Other .jpg Demented Test Data for Evaluation
    ├── NonDemented                 ## Folder of NonDemented Test Samples
        ├── NonDementedExample.jpg  ## Sample MRI image of NonDemented Patient
        ├── ...                     ## Other .jpg Non Demented Test Data for Evaluation
├── .gitattributes                  ## Text file that gives attributes to pathnames
├── README.md                       ## Introduction and Installation/Reproduction Instructions
├── VGG19_demo.py                   ## Python file where Model is Loaded, Evaluated, Makes a Single Image Predicion, and Displays a Confusion Matrix on Test Data
├── VGG19_train.py                  ## Python file where Model is Created, Trained, and Accuracy Changes Plotted **CANNOT BE RUN WITHOUT DOWNLOADING TRAINING DATA**
├── vgg19_300.keras*                ## The Trained Keras Model
```
*The size of the Keras file containing the model exceeds Github's limit. This file was added to the Github using Git Large File Storage. Downloading large files requires Git LFS installed, as another option [Download the Keras Model Manually](https://drive.google.com/file/d/18VeSLPzMVDFHG6QxsjzX4AXl3JBfHtNy/view). If downloading the model manually, ensure that it is added to the repository so the file hierarchy is the same as above.

***Users MUST either download the model via Git LFS or manually add it to the folder and replace existing keras file, the Python files will NOT run otherwise***

<a name="installation"></a>

## 1. Installation:

### Libraries Used:

This project utilizes the following libraries:

- **TensorFlow**: An open-source machine learning framework developed by the Google Brain team. It is widely used for building and training deep learning models.
  - [TensorFlow Official Page](https://www.tensorflow.org/)
  - Install with: `pip install tensorflow`

- **NumPy**: A powerful library for numerical operations in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
  - [NumPy Official Page](https://numpy.org/)
  - Install with: `pip install numpy`

- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python. It is often used for data visualization.
  - [Matplotlib Official Page](https://matplotlib.org/)
  - Install with: `pip install matplotlib`

- **OpenCV (cv2)**: Open Source Computer Vision Library, a powerful computer vision and image processing library.
  - [OpenCV Official Page](https://opencv.org/)
  - Install with: `pip install opencv-python`

- **Keras**: A high-level neural networks API running on top of TensorFlow. It simplifies the process of building, training, and deploying deep learning models.
  - [Keras Official Page](https://keras.io/)
  - Install with: `pip install keras`

- **scikit-learn**: A machine learning library providing simple and efficient tools for data analysis and modeling, including classification and regression.
  - [scikit-learn Official Page](https://scikit-learn.org/stable/)
  - Install with: `pip install scikit-learn`

Make sure to install these libraries before running the code. You can use the provided links for more information and installation instructions.

### Dataset

This project utilizes the [Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset) available on Kaggle. The dataset is designed for Alzheimer's disease classification based on magnetic resonance imaging (MRI) scans. It includes augmented images to enhance model robustness and generalization.

#### Overview

- **Dataset Source**: [Augmented Alzheimer MRI Dataset on Kaggle](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
- **Description**: The dataset contains MRI scans of the brain, specifically focusing on patients with Alzheimer's disease. The augmentation techniques applied aim to diversify the dataset and improve the model's performance.

#### Citation

If you use this dataset in your work, please make sure to cite the original source:

> Uraninjo. "Augmented Alzheimer MRI Dataset." Kaggle, 2023. [https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

#### Training and Testing

**Since no categories for levels of dementia were used, the very mild, mild and moderate folders were combined to train under 'Demented'.**

The training for vgg19_300.keras was done on the 'AugmentedAlzheimerDataset' folder (with categories of dementia combined). 

A small sample of the 'OriginalDataset' folder is included under 'TestSamples' in this repo. Screenshots from more extensive training are included in the report.

To reproduce results identical to in the report, download the 'OriginalDataset' folder, combine categories so that folders are 'Demented' and 'NonDemented' and set path in notebook.

<a name="repro"></a>

## 2. Reproduction:
Once the dependencies have been installed, run the following commands:

```
git clone https://github.sfu.ca/kbrandon/Alzheimers-Detector.git
cd Alzheimers-Detector
```
Then insert the model into the same folder the README is in (as shown in the folder hierarchy above), either downloaded via Git LFS or manually using the link above.

```
python VGG19_demo.py
```

This script processes data in the TestSamples folder, prints predictions in the terminal, and displays a confusion matrix.
