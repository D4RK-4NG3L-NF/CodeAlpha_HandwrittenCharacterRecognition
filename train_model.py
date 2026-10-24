import tensorflow as tf 
from tensorflow.keras import layers, models

#Load the MNIST the data
(X_train, Y_train), (X_test, Y_test) = (
    tf.keras.datasets.mnist.load_data()
)

#Normlize pixel values to the range 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#Add a channel dimension for grayscale images
X_train = X_train[..., None]
X_test = X_test[..., None]

#Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

#Configure the model for training 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Train the model
model.fit(
    X_train,
    Y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

#Evaluate the image  on test images
test_loss, test_accuracy = model.evaluate(
    X_test, Y_test, verbose=2
)

print("Test accuracy:", round(test_accuracy * 100, 2), "%")

#Save the trained model
model.save("handwritter_digit_model.keras")
print("Model saved successfully!")
