# Traffic Signs Classification Project

This repository contains the code for both a **Web Application** and an **Android Application** for Traffic Signs Classification using a Convolutional Neural Network (CNN). The project is designed to recognize various types of traffic signs and classify them accurately, aiding in applications such as driver assistance and autonomous vehicle systems.

## Overview

This project leverages a trained CNN model to classify traffic signs into different categories. It includes:
- A **web application** built with Python, Flask, and Tailwind CSS.
- An **Android application** that uses the TensorFlow Lite version of the CNN model for on-device classification.

The CNN model is trained using Keras and TensorFlow and can classify traffic signs into 43 different categories.

## Introduction

Traffic sign recognition is crucial for the development of self-driving cars and other intelligent road systems. This project employs a CNN model to classify images of traffic signs into 43 distinct classes. Using image processing and data augmentation techniques, the model achieves robust performance, making it suitable for real-world applications.

![image](https://github.com/user-attachments/assets/2a16241c-b2b6-4849-a832-07cb34cbac42)

![image](https://github.com/user-attachments/assets/a93a1fac-cc18-41f0-a4f0-1cc8e550ce30)

---

## Dataset

The project uses the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset, which contains over 35,000 images of 43 different traffic sign classes. The images are of varying sizes, lighting conditions, and angles, making this a challenging classification task.

### Data Split
- **Training Set**: 80% of the dataset
- **Validation Set**: 20% of the training data
- **Test Set**: 20% of the total dataset

![image](https://github.com/user-attachments/assets/c5e6c8ad-35bf-4393-a829-d55ae2743341)


---

## Model Architecture

The CNN model has been built using the Keras framework. The architecture includes:

- **Convolutional Layers**: Extract features from the input images.
- **MaxPooling Layers**: Reduce the spatial dimensions of feature maps.
- **Dropout Layers**: Prevent overfitting by dropping nodes during training.
- **Dense Layers**: Perform final classification using a softmax activation function.

![image](https://github.com/user-attachments/assets/dd5a5249-3069-4f96-a83b-c6788bb5bdeb)

### Model Summary

The network includes:
- Two sets of Convolutional and MaxPooling layers.
- Dropout layers with a 50% drop rate to enhance generalization.
- A fully connected layer followed by a softmax output layer.

---

## Installation and Setup

### Prerequisites
Make sure you have the following packages installed:
- Python (>= 3.6)
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/waliulrayhan/Traffic-Signs-Classification.git
   cd Traffic-Signs-Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Training and Evaluation

### Preprocessing

Images are preprocessed using OpenCV:

- **Grayscale Conversion**: Simplifies the image by reducing color channels.
- **Histogram Equalization**: Normalizes the lighting conditions.
- **Normalization**: Scales pixel values to the [0, 1] range.

### Data Augmentation

To improve model generalization, data augmentation techniques such as width/height shifts, zoom, shear, and rotations are applied.

### Training

To train the model, run:
```bash
python TrafficSigns_main.py
```
The model is trained using the Adam optimizer and categorical cross-entropy loss function. The training history, including accuracy and loss curves, is plotted for analysis.

![image](https://github.com/user-attachments/assets/aed864f7-2d4f-42c9-9cd8-4c8bd70de15c)

### Evaluation
The model is evaluated on the test set to measure performance. The final model is saved as `model_trained.p` for later use.

---

## Results and Visualizations

### Accuracy Curves

![image](https://github.com/user-attachments/assets/cee45fd8-5ff3-43cf-a791-f05af97c834f)

### Loss Curves

![image](https://github.com/user-attachments/assets/28c550f4-f443-4579-abc6-918fd40d2568)

---

## Live Testing

You can test the model in real-time using a webcam. The model detects and classifies traffic signs with a specified probability threshold.

### Running the Live Test

```bash
python TrafficSigns_test.py
```
- Press `q` to quit the live demo.
- The `TrafficSigns_test.py` script captures images from the webcam, preprocesses them, and classifies the traffic signs using the trained model.

### Live Test Results

Below are some sample screenshots from the live testing:

| ![Live Test 1](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_1.png) | ![Live Test 2](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_2.png) | ![Live Test 3](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_3.png) |
|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
| ![Live Test 4](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_4.png) | ![Live Test 5](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_5.png) | ![Live Test 6](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_6.png) |

## Features

- **Web Application**:
  - Real-time traffic sign classification.
  - Displays the identified traffic sign with a probability score.
  - User-friendly interface built using Flask and styled with Tailwind CSS.

- **Android Application**:
  - Real-time traffic sign detection using the device's camera.
  - Intuitive and mobile-optimized user interface.
  - Efficient and fast on-device classification with TensorFlow Lite.

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, Tailwind CSS
- **Mobile Frameworks**: Android SDK, TensorFlow Lite, OpenCV, CameraX
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Data Handling and Visualization**: NumPy, Pandas, Matplotlib, OpenCV
- **Miscellaneous**: ImageDataGenerator for data augmentation, Pickle for model storage

## Contributing

We welcome contributions from the community! If you wish to contribute:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes and open a pull request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- **Keras and TensorFlow**: For providing an excellent framework to build and train deep learning models.
- **OpenCV**: For image preprocessing and handling.
- **Flask and Tailwind CSS**: For web app development.
- **Android SDK and TensorFlow Lite**: For mobile optimization.

## Contact

For any queries or issues, please contact [mdwaliulislamrayhan@gmail.com] or open an issue on the repository.
