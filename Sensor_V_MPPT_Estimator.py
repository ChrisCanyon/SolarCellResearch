import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math

def array_increment(array, N):
    size = len(array)
    
    for i in range(size):
        if array[size - 1 - i] == N:
            #check if done
            if (size-i-1) == 0:
                #done
                return
            else:
                #else roll over to 0
                array[size - 1 - i] = 1
        else:
            # add one and done
            array[size - 1 - i] += 1
            return

def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")

def compare_models(model):
    avgVoltError = evaluate_model(model, 'dataset1k.mat')

    try:
        f = open('bestAvgVoltError', 'r')
        best = float(f.read())
        f.close()
        print("Old best:", best, "Current config:", avgVoltError)
        if best > avgVoltError:
            print("New best config. Saving")
            save_model(model, 'optimalNN')
            f = open('bestAvgVoltError', 'w')
            f.write("%f" % avgVoltError)
            f.close()
    except FileNotFoundError as e:
        print(e, "No old best, creating files...")
        f = open('bestAvgVoltError', 'w+')
        f.write("%f" % avgVoltError)
        f.close()
        save_model(model, 'optimalNN')

def configure_model(N, L):
    for i in range(L):
        layers = [1 for X in range(i+1)]
        for j in range(int(math.pow(N,i+1) - 1)):
            array_increment(layers, N)
            print("Training with layers:", layers)
            model = train_model(layers, 'dataset10k.mat', 'test')
            compare_models(model)

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

    print('avg error in Volts for test data: {0}, {1}'.format(testdatafile, averageErrorVolts))
    return averageErrorVolts

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
    save_model(model, name)

    return model

def load_model(name):
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(name + ".h5")
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

layers = [15,15,15,15]
goodname = 'model4x15'
name = 'optimalNN'
try:
    model = load_model(name)
except OSError as e:
    print('No saved model, training new one')
    configure_model(5,3)
    model = load_model(name)

evaluate_model(model, 'dataset1k.mat')