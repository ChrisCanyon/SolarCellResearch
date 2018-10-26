import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math

name = 'model4x15'
try:
    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(name + ".h5")
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
except OSError as e:
    print('No saved model: ' + name)

testdata = sio.loadmat('dataset.mat')

testX = numpy.array(testdata['inputs'])
testY = numpy.array(testdata['labels'][0])

predictions = model.predict(testX)

print('Predictions: ', predictions)
print('Expected   : ', testY)