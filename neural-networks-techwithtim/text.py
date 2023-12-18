# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Loading the IMDB movie review dataset from Keras datasets
data = keras.datasets.imdb

# Splitting the dataset into training and testing sets
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# Getting the word index mapping from the dataset
word_index = data.get_word_index()

# Adjusting the word index by adding 3 to each value and adding special tokens
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Creating a reverse word index mapping
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Padding the sequences to ensure uniform length for input data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Function to decode a sequence of indices back to words
def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# Building the neural network model
model = keras.Sequential()
# Adding an embedding layer with input dimension 10000 and output dimension 15
model.add(keras.layers.Embedding(88000, 15))
# Adding a global average pooling layer to reduce dimensionality
model.add(keras.layers.GlobalAveragePooling1D())
# Adding a dense layer with 16 units and ReLU activation function
model.add(keras.layers.Dense(16, activation="relu"))
# Adding the output layer with 1 unit and sigmoid activation function
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Displaying the summary of the model architecture
model.summary()

# Compiling the model with Adam optimizer, binary crossentropy loss, and accuracy metric
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Separating the data into training and validation sets
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Training the model for 40 epochs with a batch size of 512, using validation data
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Evaluating the model on the test data
results = model.evaluate(test_data, test_labels)
print(results)


# Function to decode a sequence of indices back to words
def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# ...

test_review = test_data[0]
test_review = np.array([test_review])
predict = model.predict(test_review)
print("Review : ")
print(decode(test_review[0]))  # Access the first review in the batch
print("Prediction: " + str(predict[0][0]))
print("Actual: " + str(test_labels[0]))

print(results)

model.save("model.keras")
# def review_code(s):
#     encoded = [1]
#     for word in s:
#         if word.lower() in word_index:
#             encoded.append(word_index[word.lower()])
#         else:
#             encoded.append(2)
#     return encoded

# model = keras.models.load_model("model.h5")
# with open("test.txt", encoding= "utf-8") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
#         encode = review_code(nline)
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])

