# Breast cancer Prediction
This project develops a deep learning model comparing two convolutional neural networks (CNN).</br>
The focus is on selecting an appropriate deep-learning model that can diagnose breast cancer images. The comparison is between designing and building a CNN model from scratch against using VGG-16 from transfer learning (discussed in Chapter 3: Methodology).
After comparing and evaluating these models, the best performing model is the VGG-16 from transfer learning using the ``SparseCategoricalCrossentropy" (as seen in Chapter 4: Results and Evaluation). This model is used for detecting and diagnosing breast cancer from histopathological images. The adjusted VGG-16 model is trained on a breast cancer histopathology image dataset obtained from Kaggle. The VGG-16 is adjusted by setting to include the connected and trainable layers to `False'. This is because the VGG-6 is trained with over 120 million IMAGENET which are natural images, and it has adequately learned from images for image classification. There is no need for it to start training over again, and furthermore, freezing the output layer to adopt the binary classification task of classifying the images as benign and malignant. Redefining the output layer changes the VGG-16, which initially has various output layer classes to suit this specific task. 

# Language and environment used
• Python: Python is a programming language that enables rapid completion of work, which makes this project more efficient and successful for system integration. </br>
• Jupyter: Jupyter is a free and most recent web-based interactive notebook used as an environment for data and software development. </br>
The model development for diagnosing breast cancer was tried out in different environments, like Kaggle and Google Colab. Although Kaggle and Google Colab gave a better option for GPU usage. Colab required repeatedly reloading and rerunning the code, and Kaggle did not support the VGG-16 model. However, the Jupyter coding environment has a better ability to train and retrain the code easily and efficiently.

# Machine learning libraries and frameworks
• TensorFlow: TensorFlow is a well-known deep learning framework used to create and train neural networks. TensorFlow, compared to Pytorch, is most effective in dataflow programming as it provides a range of options for different purposes. </br>
• Keras: A high-level neural network API library that operates on top of TensorFlow and offers a simple user interface for model construction. Keras synchronises well with Python, provides several pre-trained networks, and performs well. </br>
• Python Imaging Library (PIL): PIL is the standard image processing package for the Python language. It was used because it is capable of simple image processing, opening, and manipulation when the saved model was to be tested with the test images. </br>
• Open CV: An image processing and manipulation tool for machine learning and computer vision. </br>
• Matplotlib: This is a Python library used for creating visualisations. </br>
• Numpy: This Python library is used for computations and arrays. </br>

# Dataset
The dataset used to train the breast cancer diagnosis model is the breast cancer histopathology images, which provide a diverse range of breast cancer image cases (discussed in Section 3.2).

# Model development
The breast cancer diagnosis model is built using Python and TensorFlow-Keras frameworks. The model script provided includes a sequential process of installing and importing the required libraries and frameworks, loading and preprocessing the image dataset, and building and training the deep learning model architecture.

# Evaluation and testing
To evaluate the performance of the developed models, we accessed and compared them to identify the model with the best classification of true positives and true negatives as against false positives and false negatives (as discussed in Section 3.6 and seen in Section 4.2). Other evaluation metrics, such as precision, recall, and accuracy, were taken into consideration. Also, we compared their loss rates by calculating their error bar analysis to determine their statistical difference (Section 4.3). The code also includes a developed function to load the saved model, perform preprocessing on the test images, and evaluate the results.

# Running the code
To run the code and utilise the breast cancer diagnosis model, please follow the following steps: </br>
i) Install the necessary dependencies: TensorFlow, Keras, PIL, and OpenCV. </br>
ii) Download the breast cancer histopathology images (BHI) dataset from Kaggle and add it to the script folder. </br>
iii) The necessary modifications should be made to the code to include the file locations and reset any hyperparameters. </br>
iv) Run the code to train the diagnosis model on the BHI dataset. </br>
v) Once the model is trained, save the model for further testing. </br>
vi) Load the saved model and use unseen images to test the performance. </br>
vii) Evaluate and analyse the performance of the model results. </br>
For thorough directions on each step, kindly refer to the provided Python code and comments. Please feel free to modify and adapt the code and parameters to fit your requirements and datasets.
