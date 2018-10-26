import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math

def array_increment(array, N)
    size = array.size
    for i in range(array.size)
        if array[size - 1 - i] == N
            #check if done
            if size-i = 0
                #done
                return array
            else
                #else roll over to 0
                array[size-i] = 0
        else 
        # add one and done
        array[size - 1 - i] += 1
        return array



def evaluate_model(model, testdatafile):
    testdata = sio.loadmat(testdatafile)

    testX = numpy.array(testdata['inputs'])
    testY = numpy.array(testdata['labels'][0])

    predictions = model.predict(testX)
    flat_list = [item for sublist in predictions for item in sublist]

    totalError = 0
    i = 0
    for p in flat_list:
        error = abs(testY[i] - p)
        totalError += math.pow(error, 2)
        i += 1
    averageErrorVolts = (totalError/i)

    print('avg error in Volts for test data: %s' % testdatafile)
    print(averageErrorVolts)

def train_model(layers, dataset, name):
    data = sio.loadmat(dataset)

    X = numpy.array(data['inputs'])
    Y = numpy.array(data['labels'][0])

    model = keras.Sequential()
    model.add(Dense(layers[0], input_dim=2, activation='sigmoid'))

    for nodes in layers[1:]:
        model.add(Dense(nodes, activation='sigmoid'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    model.fit(X, Y, epochs=1000, batch_size=10000, verbose=0)

    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")

    return model

L = 3
N = 10

layers = [15,15,15,15]
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
    print('No saved model, training new one')
    model = train_model(layers, 'dataset10k.mat', name)

evaluate_model(model, 'dataset1k.mat')