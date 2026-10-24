import tensorflow as tf

#Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = (
    tf.keras.datasets.mnist.load_data()
)

#Normalize the pixels values 
X_train = X_train / 255.0
X_test = X_test / 255.0

#Check the prepare data
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())
print("Correct label of first image:", Y_train[0])