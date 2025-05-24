# IMAGE-CLASSIFICATION-MODEL

COMPANY : CODTECH IT SOLUTIONS

NAME : NAVEEN DAGGUBATI

INTERN ID : CT06DL790

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTOSH

EXPLANATION OF THE CODE:

The given code implements an image classification pipeline using the CIFAR-10 dataset and a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

1. Importing Libraries

The code begins by importing essential Python libraries:

* `numpy` for numerical operations.
* `cv2` (OpenCV) for image processing.
* `matplotlib.pyplot` for image visualization.
* `tensorflow` and `keras` for building and training the deep learning model.

---

2. Loading and Preprocessing Data

The CIFAR-10 dataset is loaded using `datasets.cifar10.load_data()`. This dataset consists of 60,000 color images (32x32 pixels) across 10 classes, such as airplane, automobile, bird, etc. The dataset is automatically split into 50,000 training and 10,000 testing images and labels.

Images are then normalized by dividing each pixel value by 255 to scale them between 0 and 1. This helps improve the performance and convergence speed of the neural network.

---

3. Visualizing the Dataset

The code then visualizes the first 100 training images using `matplotlib`. Each image is labeled with its corresponding class name using the `class_names` list. This step is useful for verifying that the data has been loaded correctly and understanding what the dataset looks like.

---

4. CNN Model Construction

A Convolutional Neural Network (CNN) is built using the Keras Sequential API. The model includes the following layers:

* `Conv2D`: Extracts features from the images using 32 and then 64 filters.
* `MaxPooling2D`: Downsamples the feature maps to reduce computation and overfitting.
* `Flatten`: Converts the 2D feature maps to a 1D feature vector.
* `Dense`: Fully connected layers. The final layer uses `softmax` to output probabilities for each of the 10 classes.

---

5. Compiling and Training the Model

The model is compiled with:

* `adam` optimizer for efficient gradient descent.
* `sparse_categorical_crossentropy` as the loss function because the labels are integers.
* `accuracy` as a performance metric.

The model is then trained for 10 epochs using the training data and validated on the test data.

---

6. Evaluating and Saving the Model

After training, the model is evaluated on the test set using `model.evaluate()`, which returns the loss and accuracy.

The trained model is saved using `model.save()` in the `.keras` format and can be reloaded using `keras.models.load_model()`.

---

7. Predicting on an External Image

An external image (`Bird.jpg`) is loaded using OpenCV. Since OpenCV loads images in BGR format, it is converted to RGB using `cv2.cvtColor()`. The image is resized to 32x32 to match CIFAR-10 input dimensions and normalized.

The preprocessed image is reshaped into a batch of one image and passed to the model for prediction. The `np.argmax()` function is used to determine the class with the highest probability, and the predicted class name is displayed along with the image using `matplotlib`.

---

Conclusion

This code offers a full machine learning pipeline: loading data, preprocessing, training a CNN model, saving/loading the model, and testing it on real-world images. It serves as a practical foundation for anyone learning computer vision and deep learning using Python and TensorFlow.

For the classification of Images, we gave some images to classify the images into types like bird, truck, frog and so on...

DATASET IMAGE:

![Image](https://github.com/user-attachments/assets/597b0ebf-3677-49c7-ad9f-dacd465ad9a1)






For the classification, the input is given and the output is provided by our classification model.

INPUT IMAGE :

![Image](https://github.com/user-attachments/assets/9e76fa73-a0c4-4fbc-9bf7-46c603be4c05)





BY USING THE INPUT, OUR MODEL PREDICTED THAT IT IS A "BIRD".

OUTPUT:


"Bird"













