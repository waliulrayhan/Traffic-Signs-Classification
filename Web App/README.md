# Traffic Signs Classification Web Application

This repository contains the code for the **Traffic Signs Classification Web Application**, developed using Python, Flask, HTML, and Tailwind CSS. The project leverages a Convolutional Neural Network (CNN) for image classification, which identifies various types of traffic signs. The application provides a user-friendly web interface for uploading images and viewing the classification results.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to assist in the identification and classification of traffic signs using a trained deep learning model. The web application is built to run the classification task seamlessly through a web browser, providing quick and accurate results. The CNN model used in this project is trained using Keras and TensorFlow on a dataset of traffic sign images.

## Features

- **Real-time Prediction:** The app displays the classified sign along with the probability score.
- **User Interface:** Built with Flask for backend services and Tailwind CSS for a responsive and modern frontend.
- **Model Persistence:** Uses a pre-trained model stored in `.p` format for efficient predictions.

## Technology Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, Tailwind CSS
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Data Handling and Visualization:** NumPy, Pandas, Matplotlib, OpenCV
- **Miscellaneous:** ImageDataGenerator for data augmentation, Pickle for model storage

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Traffic-Signs-Classification-WebApp.git
   cd Web App
2. **Create a virtual environment and activate it:**
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install the required dependencies:**
   ```bash
    pip install -r requirements.txt
4. **Run the Flask app:**
   ```bash
    python app.py
The web application will be accessible at http://127.0.0.1:5000/.

## Usage

1. **Navigate to the web app** using the provided URL.
2. **View the classification results**, which include the identified class and the confidence level.

## Model Training

The CNN model used in this project was developed using the following methodology:

- **Dataset**: Composed of traffic sign images, labeled appropriately for classification.
- **Preprocessing Techniques**:
  - Conversion to grayscale for uniform image analysis.
  - Histogram equalization to standardize lighting.
  - Normalization of pixel values for better model performance.
- **Image Augmentation**:
  - Employed techniques such as zooming, shifting, and rotating to enhance model generalization.
- **Model Architecture**:
  - Multiple convolutional layers paired with ReLU activation.
  - MaxPooling for feature map reduction.
  - Dropout layers to minimize overfitting.
  - Dense layers as the final decision-making layers with softmax activation for multi-class classification.

The training script can be found in `TrafficSigns_main.py`, and the real-time prediction script is in `TrafficSign_Test.py`.

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
- **Flask**: For enabling simple and powerful web app development.
- **Tailwind CSS**: For providing a modern, responsive design framework.

## Contact

For any queries or issues, please contact [mdwaliulislamrayhan@gmail.com] or open an issue on the repository.
