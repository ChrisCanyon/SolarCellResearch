import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math

def evaluate_model(model, testdatafile):
    testdata = sio.loadmat(testdatafile)

    testX = numpy.array(testdata['inputs'])
    testY = numpy.array(testdata['labels'][0])

    predictions = model.predict(testX)
    flat_list = [item for sublist in predictions for item in sublist]

    print('avg error in Volts for test data: %s' % testdatafile)
    totalError = 0
    i = 0
    for p in flat_list:
        error = abs(testY[i] - p)
        totalError += math.pow(error, 2)
        i += 1
    mse = (totalError/i)
    print("Preditions: ",flat_list)
    print("Actual: ", testY)
    print(mse)

def train_model(layers, dataset):
    data = sio.loadmat(dataset)

    X = numpy.array(data['inputs'])
    Y = numpy.array(data['labels'][0])

    model = keras.Sequential()
    model.add(Dense(layers[0], input_dim=2, activation='sigmoid'))

    for nodes in layers[1:]:
        model.add(Dense(nodes, activation='sigmoid'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    model.fit(X, Y, epochs=1000, batch_size=10000)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model

try:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
except OSError as e:
    print('No saved model, training new one')
    layers = [9,6,4,3]
    model = train_model(layers, 'dataset10k.mat')

evaluate_model(model, 'dataset.mat')