from library.config.exhaustive import *
from library.config.genetic import *
from library.CVtrain import *
import scipy.io as sio

datasets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EPOCHS = 500
BATCH_SIZE = 150
POSSIBLE_NODES = 20
POSSIBLE_LAYERS = 5
outputStructures = []
outputMSEs = []

for N in datasets:
    outputStructures.append([])
    outputMSEs.append([])
    
i = 0
for N in datasets:
    t0 = time.time()

    TrainFile = "datasets/N" + str(N) + "dataset10k.mat"
    EvalFile = "datasets/N" + str(N) + "dataset100k.mat"
    ModeName = "N"  + str(N) + "geneticResult"

    print("Genetically searching for best layer structure...")
    print("  Training on:    ", TrainFile)
    print("  Evaluating with:", EvalFile)
    layers = genetic_config(POSSIBLE_NODES, POSSIBLE_LAYERS, TrainFile, 25, BATCH_SIZE, 0, EPOCHS)
    print("Result: ", layers)

    evalSet = sio.loadmat(EvalFile)
    MSE = CVtrain(layers, evalSet, 10, ModeName, batch=10, verbose=1, epochs=1000)
    t1 = time.time()
    print("MSE after 10-Fold CV:",MSE)
    print("Total Time:", (t1-t0)/60, "minutes")
    
    outputStructures[i].append(layers)
    outputMSEs[i].append(MSE)
    i = i + 1

print(outputStructures)
print(outputMSEs)