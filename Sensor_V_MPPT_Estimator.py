# TODO: make genetic learn

import keras
from keras.models import model_from_json
from keras.layers import Activation, Dense
import scipy.io as sio
import numpy
import math
from multiprocessing.dummy import Pool as ThreadPool
import random

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

# break l into n sized chunks
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# saves model to file with name 'name'
# TODO: Make loss, optimizer, and metrics save to file for a NN
def save_model(model, name, layers, MSE):
    print("Saving model:", name, "\nMSE:", MSE)
    # serialize model to JSON
    model_json = model.to_json()
    with open("./NNs/" + name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    # save layer structure
    with open("./NNs/" + name+"_structure", "w") as structure_file:
        for node in layers:
            structure_file.write(str(node) + ' ')
    # save MSE
    with open("./NNs/" + name+"_MSE" , 'w') as f:
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

# compares 2 models against the same dataset
# returns 1 if model1 has lower MSE than model2
def compare_models(model1, model2, dataset):
    MSE1 = evaluate_model(model1, dataset)
    MSE2 = evaluate_model(model2, dataset)

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

def CVtrain(layers, trainingSet, evaluateSet, folds=1, name="trash", batch=100, verbose=0, epochs=1000):
    data = sio.loadmat(evaluateSet)

    X = numpy.array(data['inputs'])
    Y = numpy.array(data['labels'][0])
    datasize = len(Y)
    chunkSize = int(datasize/folds)
    
    if folds >= datasize:
        print("Error in CVtrain. Dataset not larger enough for {0}-fold CV".format(folds))
        return None
    if chunkSize < batch:
        print("Error in CVtrain. Batch size larger that chunk size for {0}-fold CV".format(folds))
        return None

    Xchunks = chunks(X, chunkSize)
    Ychunks = chunks(Y, chunkSize)

    #train model
    bestModel = None
    bestMSE = 1000
    totalMSE = 0
    for i in range(folds):
        if verbose == 1:
            print("Fold", i)

        model = train_model(layers, next(Xchunks), next(Ychunks), batch, verbose, epochs)
        MSE = evaluate_model(model, evaluateSet, 0)

        totalMSE += MSE

        if MSE < bestMSE:
            bestModel = model
            bestMSE = MSE
    
    avgMSE = totalMSE/(i+1)
    print("Average MSE:", avgMSE)
    save_model(bestModel, name, layers, avgMSE)
    return bestModel

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
def train_model(layers, X, Y, batch=100, verbose=0, epochs=1000):
    #create model
    model = keras.Sequential()

    model.add(Dense(layers[0], input_dim=2, activation='sigmoid'))
    for nodes in layers[1:]:
        model.add(Dense(nodes, activation='sigmoid'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    model.fit(X, Y, epochs=epochs, batch_size=batch, verbose=verbose)

    return model

# searches through all possible layer node arrangments
# for L layers with up to N nodes each
def exhuastive_config(N, L, trainingSet, evaluateSet, batchSize=100, verbose=0, epochs=500):
    bestModel = None
    bestMSE = None
    bestLayers = None

    for i in range(L):
        layers = [1 for X in range(i+1)] #create array of layers with 1 node each

        # formula for possible sctructures with N nodes and i layers
        # used so array_increment goes through every possible option
        possibleStructures = int(math.pow(N,i+1) - 1)
        for j in range(possibleStructures):
            array_increment(layers, N) # assuming all 1 node layers is bad
            print("Training with layers:", layers)
            model = CVtrain(layers, trainingSet, evaluateSet, 5, 'test', batchSize, verbose, epochs)
            MSE = getMSE('test')
            if bestModel == None:
                bestModel = model
                bestMSE = MSE
                bestLayers = layers

            if  bestMSE > MSE:
                bestModel = model
                bestMSE = MSE
                bestLayers = layers


    print("Best model for N:", N, "L:", L, "\nMSE:", bestMSE, "\nlayers:", bestLayers)
    save_model(bestModel, str(N)+'x'+str(L)+"optimalNN", bestLayers, bestMSE)
    return bestModel 

# adds 0s to make layer arrays all the same size
def append_zeros(layers, L):
    x = len(layers)
    for i in range(L - x):
        layers.append(0)
    return layers

# removes dangling zeros from being the child of different depth NNs
def handle_zeros(layers):
    x = layers[:]
    try:
        while(1):
            x.remove(0)
    except:
        return x

# returns an array of layer arrays
def generate_initial_generation(N, L, size):
    generation = []
    for i in range(size):
        layers = []
        # choose a depth
        x = random.randint(1, L)
        for j in range(x):
            layers.append(random.randint(1, N))
        layers = append_zeros(layers, L)
        generation.append(layers)

    return generation

# model1 and model2 are same length
def reproduce(model1, model2):
    x = len(model1)

    flipIndex = random.randint(0, x-1)

    model3 = model2[:]

    for i in range(flipIndex):
        model3[i] = model1[i]
    
    return model3

def train_generation(gen, trainingSet, evaluateSet, batchSize, verbose, epochs):
    print("Training generation")
    MSEs = []
    for i in range(len(gen)):
        trainFriendlyLayer = handle_zeros(gen[i])
        MSEs.append(CVtrain(trainFriendlyLayer, trainingSet, evaluateSet, 5, "trash", batchSize, verbose, epochs))
    
    print("MSEs after training:", MSEs)
    return MSEs

# TODO: look into mutate options
# currently just chooses a random index and changes the nodes to a random number
def mutate(layers, N, L):
    x = len(layers)

    i = random.randint(0,x-1)

    lower = 1 - layers[i] 
    upper = N - layers[i]

    layers[i] += random.randint(lower, upper)

    return layers


def select_parent(fitness):
    print("Fitness:", fitness)
    total = sum(fitness)

    chosen = random.randint(0, total)

    for i in len(fitness):
        chosen = chosen - fitness[i]
        if chosen <= 0:
            return i
    
def generate_next_generation(lastGen, MSEs, N, L):
    # sort MSEs and lastGen so they are in order of best MSE to worst
    errors, structures = zip(*sorted(zip(MSEs, lastGen), key=lambda x: x[0], reverse=False))
    errors = list(errors)
    structures = list(structures)

    # convert errors to some fitness value
    maxError = max(errors)
    fitness = []
    for e in errors:
        fitness.append(maxError/e) #dividing maxError/e makes smaller errors end up with larger fitness scores

    # Save top X structures for next gen
    nextGen = structures[0:5]

    # loop to fill rest of the generation
    x = len(MSEs)
    for i in range(x - 5):

        # get mom and dad
        momIndex = select_parent(fitness)
        dadIndex = select_parent(fitness)

        mom = structures[momIndex]
        dad = structures[dadIndex]
        # make baby
        child = reproduce(mom, dad)
        # roll some chance to mutate
        if (random.randint(0,100) == 0):
            child = mutate(child, N, L)
        nextGen.append(child)

    return nextGen

def genetic_config(N, L, trainingSet, evaluateSet, batchSize=250, verbose=0, epochs=500):
    # create first list of models
    genepoolSize = 10
    currentGen = generate_initial_generation(N, L, genepoolSize)
    print("Gen1:", currentGen)
    #train a generation
    while(1):
        MSEs = train_generation(currentGen, trainingSet, evaluateSet, batchSize, verbose, epochs)
        currentGen = generate_next_generation(currentGen, MSEs, N, L)
        print("Next Gen:", currentGen)
        return

####################
##                ##
##  ~~~ Main ~~~  ##
##                ##
####################

genetic_config(5,3, "dataset10k.mat", "dataset1k.mat", 250, 0, 200)