import tensorflow as tf
import matplotlib.pyplot as plt 

#Load the built-in MNIST dataset
(X_train, Y_train), (X_test, Y_test) = (
    tf.keras.datasets.mnist.load_data()
)

#Display the dataset information 
print("Training images:", X_train.shape)
print("Training labels:", Y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", Y_test.shape)

#Display the first training image
plt.figure(figsize=(3, 3))
plt.imshow(X_train[0], cmap="grey")
plt.title(f"Correct digit: {Y_train[0]}")
plt.axis("off")
plt.savefig("Sample_digit.png")
plt.close()
print("Image saved as sample_digit.png")
#plt.show()