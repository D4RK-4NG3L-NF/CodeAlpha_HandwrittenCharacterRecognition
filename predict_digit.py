import tensorflow as tf 

#Load the trained model
model = tf.keras.models.load_model(
    "handwritter_digit_model.keras"
)

#Load the MNIST dataset
(_, _), (X_test, Y_test) = (
    tf.keras.datasets.mnist.load_data()
)

#Prepare one test image
image = X_test[0].astype("float32") / 255.0
image = image.reshape(1, 28, 28, 1)

#Predict the digit
prediction = model.predict(image, verbose=0)
predicted_digit = prediction.argmax()

#Display the result
print("Predicted digit:", predicted_digit)
print("Correct digit:", Y_test[0])