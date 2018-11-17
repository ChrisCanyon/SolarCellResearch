# Model IO

import os
import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import tensorflow as tf
import numpy
import math
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Disable tensorflow info logs

# my version of sklearn's Kfold
def Kfold(X,Y, folds):
    dataSize = len(X)
    foldSize = int(dataSize/folds)

    for i in range(0, dataSize, foldSize):
        X_evaluate = X[i:i+foldSize]
        Y_evaluate = Y[i:i+foldSize]
        X_train = numpy.array(list(X[0:i]) + list(X[(i+foldSize):dataSize]))
        Y_train = numpy.array(list(Y[0:i]) + list(Y[(i+foldSize):dataSize]))
        yield (X_train, Y_train, X_evaluate, Y_evaluate)

# saves model to file with name 'name'
# allows partial saves so CVtrain can use keras.backend.clear_session()
# TODO: Make loss, optimizer, and metrics save to file for a NN
def save_model(name, model = None, layers = None, MSE = None):
    if model != None:
        # serialize model to JSON
        model_json = model.to_json()
        with open("./NNs/" + name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("./NNs/" + name+ ".h5")
    
    if layers != None:
        # save layer structure
        with open("./NNs/" + name + "_structure", "w") as structure_file:
            for node in layers:
                structure_file.write(str(node) + ' ')
    
    if MSE != None:
        # save MSE
        with open("./NNs/" + name + "_MSE" , 'w') as f:
            f.write("%f" % MSE)

# load_model loads a model with name 'name'
# Returns a compiled model
# TODO: Make loss, optimizer, and metrics input from file
def load_model(name):
    json_file = open("./NNs/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./NNs/" + name + ".h5")
    print("Loaded model from disk")
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

# evaluate_model calculates the MSE of model on testdatafile
def evaluate_model(model, X, Y, verbose=0):
    RawPredictions = model.predict(X)
    predictions = [item for sublist in RawPredictions for item in sublist]

    totalError = 0
    i = 0
    for p in predictions:
        error = Y[i] - p
        totalError += math.pow(error, 2)
        i += 1
    MSE = (totalError/i)

    if verbose==1:
        print('MSE:', MSE)
    return MSE

# compares 2 models against the same dataset
# returns 1 if model1 has lower MSE than model2
def compare_models(model1, model2, dataset):
    X = numpy.array(dataset['inputs'])
    Y = numpy.array(dataset['labels'][0])
    MSE1 = evaluate_model(model1, X, Y)
    MSE2 = evaluate_model(model2, X, Y)

    if MSE2 > MSE1:
        return 1
    return 0

# compare models saved model
# allows comparison of AVG MSE computed by CVtrain
# returns 1 if model1 has lower MSE than model2
def compare_saved_models(model1, model2):
    MSE1 = getMSE(model1)
    MSE2 = getMSE(model2)

    if MSE2 > MSE1:
        return 1
    return 0

def sim(input, name):
    try:
        model = load_model(name)
    except OSError as e:
        print('model:', name, 'not found')
        exit()

    return model.predict(input)

# returns the saved MSE saved for a NN
def getMSE(name):
    f = open("./NNs/"+name+"_MSE", 'r')
    MSE = float(f.read())
    f.close()
    return MSE

def CVtrain(layers, trainingSet, folds=5, name="trash", batch=100, verbose=0, epochs=1000):
    X = numpy.array(trainingSet['inputs'])
    Y = numpy.array(trainingSet['labels'][0])
    datasize = len(Y)
    chunkSize = int(datasize/folds)
    
    if folds >= datasize:
        print("Error in CVtrain. Dataset not larger enough for {0}-fold CV".format(folds))
        return None
    if chunkSize < batch:
        print("Error in CVtrain. Batch size larger that chunk size for {0}-fold CV".format(folds))
        return None

    data = Kfold(X, Y, folds)

    #train model
    bestModel = None
    bestMSE = 1000
    totalMSE = 0
    for i in range(folds): 
        if verbose == 1:
            print("Fold", i)

        foldedData = next(data)
        X_train = foldedData[0]
        Y_train = foldedData[1]
        X_eval = foldedData[2]
        Y_eval = foldedData[3]

        if verbose==1:
            t0 = time.time()
        model = train_model(layers, X_train, Y_train, batch, 0, epochs)

        MSE = evaluate_model(model, X_eval, Y_eval, 0)
        if verbose==1:
            t1 = time.time()
            print("Fold {0} Train time: {1} minutes, MSE: {2}".format(i, (t1-t0)/60, MSE))

        totalMSE += MSE
        if MSE < bestMSE:
            bestModel = model
            bestMSE = MSE
            save_model(name, bestModel)
        keras.backend.clear_session() #Clearing sessions improves performance over multipe model trains

    avgMSE = totalMSE/folds
    save_model(name, model=None, layers=layers, MSE=avgMSE)
    return avgMSE 

# train_model creates a keras.Sequential() NN with network structure 'layers'
# returns a trained model
# Paramters:
#  layers:   array of node lengths
#  X, Y:     training data for use with model.fit
#  batch:    batch_size
#  verbose:  0 = no training output 
#            1 = training output
#  epochs:   number of epochs in model.fit
#  folds:    number of folds for K-fold CV
# TODO: add 'shuffle' to model.fit and test
def train_model(layers, X, Y, batch=100, verbose=0, epochs=1000):
    #create model
    model = keras.Sequential()

    model.add(Dense(layers[0], input_dim=2, activation='sigmoid'))
    for nodes in layers[1:]:
        model.add(Dense(nodes, activation='sigmoid'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    model.fit(X, Y, epochs=epochs, batch_size=batch, verbose=verbose, shuffle=True)

    return model