import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist

(x_training, y_training), (x_test, y_test) = mnist.load_data()

x_training = x_training.reshape(60000, 784)

x_training = x_training.astype("float32")
x_training = x_training / 255

y_training = np_utils.to_categorical(y_training, 10)

model = Sequential()

model.add(Dense(800, input_dim=784,  kernel_initializer="normal", activation="relu"))
model.add(Dense(10,  kernel_initializer="normal", activation="softmax"))

model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

model.fit(x_training, y_training, batch_size=200, epochs=1, verbose=1)

#predictions = model.predict(x_training)
#predictions = numpy.argmax(predictions, axis=1)
