import keras
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy

data = sio.loadmat('dataset10k.mat')

X = numpy.array(data['inputs'])
Y = numpy.array(data['labels'][0])

layers = [6,4,3]

model = keras.Sequential()
model.add(Dense(9, input_dim=2, activation='sigmoid'))

for nodes in layers:
    model.add(Dense(nodes, activation='sigmoid'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=1000, batch_size=10000)

scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

testdata = sio.loadmat('dataset.mat')

testX = numpy.array(data['inputs'])
testY = numpy.array(data['labels'][0])

predictions = model.predict(testX)
print("Predictions: ")
print(predictions)
print("Actual: ")
print(testY)