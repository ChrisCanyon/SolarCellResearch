# TODO: make NN save and load from a folder in a folder called NN
# TODO: ensure MSE is only calculated once per NN

import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math


# my attempt to make structure generation easier to read
# creates an array to represent a number with base N+1
# numbers roll over to 1 because 1 node is required

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

# train_model creates a keras.Sequential() NN with network structure 'layers'
# returns a trained model
# Paramters:
#  layers:   array of node lengths
#  dataset:  .mat file with training data. Created using saveDataset.m 
#  name:     filename for the new NN
#  batch:    batch_size
#  verbose:  0 = no training output 
#            1 = training output

def train_model(layers, dataset, name, batch, verbose):
    data = sio.loadmat(dataset)

    X = numpy.array(data['inputs'])
    Y = numpy.array(data['labels'][0])

    model = keras.Sequential()
    model.add(Dense(layers[0], input_dim=2, activation='sigmoid'))

    for nodes in layers[1:]:
        model.add(Dense(nodes, activation='sigmoid'))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(X, Y, epochs=1000, batch_size=batch, verbose=verbose)
    save_model(model, name)
    MSE = evaluate_model(model, 'dataset1k.mat')
    f = open(name, 'w+')
    f.write("%f" % MSE) 
    f.close()

    return model

# saves model to file with name 'name'
def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")

# load_model loads a model with name 'name'
# Returns a compiled model
# TODO: Make loss, optimizer, and metrics optional input variables

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

# evaluate_model calculates the MSE of model on testdatafile
def evaluate_model(model, testdatafile, verbose=0):
    testdata = sio.loadmat(testdatafile)

    testX = numpy.array(testdata['inputs'])
    testY = numpy.array(testdata['labels'][0])

    RawPredictions = model.predict(testX)
    predictions = [item for sublist in RawPredictions for item in sublist]

    totalError = 0
    i = 0
    for p in predictions:
        error = testY[i] - p
        totalError += math.pow(error, 2)
        i += 1
    MSE = (totalError/i)

    if verbose==1:
        print('MSE for test data: {0}, {1}'.format(testdatafile, MSE))
    return MSE

# compares 'model' with the model_name and saves the better as model_name
def compare_models(model, model2, dataset):
    MSE1 = evaluate_model(model, dataset)
    MSE2 = evaluate_model(model2, dataset)

    if MSE2 > MSE1:
        return 1
    return 0

def shutuplinter(model_name, MSE, model):
    try:
        # get model_name MSE from file
        f = open(model_name, 'r')
        best = float(f.read())
        f.close()

        print("Old best:", best, "Current config:", MSE)
        if best > MSE:
            print("New best config. Saving")
            save_model(model, model_name)
            f = open(model_name, 'w')
            f.write("%f" % MSE)
            f.close()
    except FileNotFoundError as e:
        print(e, "No old best, creating files...")
        f = open(model_name, 'w+')
        f.write("%f" % MSE) 
        f.close()
        save_model(model, model_name)

# searches through all possible layer node arrangments
# for L layers with up to N nodes each

def configure_model(N, L, trainingSet, batchSize=1000, verbose=0, testSet='dataset1k.mat'):
    bestModel = None
    bestMSE = None

    for i in range(L):
        layers = [1 for X in range(i+1)] #create array of layers with 1 node each

        # formula for possible sctructures with N nodes and i layers
        # used so array_increment goes through every possible option
        possibleStructures = int(math.pow(N,i+1) - 1)
        for j in range(possibleStructures):
            array_increment(layers, N) # assuming all 1 node layers is bad
            print("Training with layers:", layers)
            model = train_model(layers, trainingSet, 'test', batchSize, verbose)
            MSE = evaluate_model(model, testSet, 1)

            if bestModel == None:
                bestModel = model
                bestMSE = MSE

            if compare_models(model, bestModel, testSet) == 1:
                bestModel = model
                bestMSE = MSE

    return bestModel 

def sim(input, name="optimalNN"):
    try:
        model = load_model(name)
    except OSError as e:
        print('model:', name, 'not found')
        exit()

    return model.predict(input)

####################
##                ##
##  ~~~ Main ~~~  ##
##                ##
####################

configured_model = configure_model(2,1,'dataset1k.mat', 100, 0)
print('Ding: Config Done')
evaluate_model(configured_model, 'dataset10k.mat', 1)

