from library.mio import *
import scipy.io as sio

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

# searches through all possible layer node arrangments
# for L layers with up to N nodes each
def exhuastive_config(N, L, trainingSetFile, evaluateSetFile, batchSize=100, verbose=0, epochs=500):
    trainingSet = sio.loadmat(trainingSetFile)
    evaluateSet = sio.loadmat(evaluateSetFile)

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
            CVtrain(layers, trainingSet, evaluateSet, 5, 'test', batchSize, verbose, epochs)
            MSE = getMSE('test')
            if bestMSE == None:
                bestMSE = MSE
                bestLayers = layers

            if  bestMSE > MSE:
                bestMSE = MSE
                bestLayers = layers


    print("Best model for N:", N, "L:", L, "\nMSE:", bestMSE, "\nlayers:", bestLayers)
    return bestLayers 
