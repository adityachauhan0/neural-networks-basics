# Importing required libraries
import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

# Loading the Fashion MNIST dataset from Keras
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Defining class names for better interpretation of labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Displaying an example image from the dataset
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

# Normalizing pixel values to be in the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating a neural network model using Keras Sequential API
model = keras.Sequential([
    # Flattening the 28x28 images to a 1D array
    keras.layers.Flatten(input_shape=(28, 28)),
    # Adding a densely connected layer with 128 neurons and ReLU activation function
    keras.layers.Dense(128, activation='relu'),
    # Output layer with 10 neurons (classes) and softmax activation for multi-class classification
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the model with Adam optimizer and sparse categorical crossentropy as the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model on the training data for 5 epochs
model.fit(train_images, train_labels, epochs=5)

# Evaluating the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc: ", test_acc)

# Making predictions on the test data and visualizing a few examples

prediction = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: "+ class_names[test_labels[i]])
    plt.title("Prediction :"+ class_names[np.argmax(prediction[i])])
    plt.show()

