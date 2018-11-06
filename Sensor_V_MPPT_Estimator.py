from library.config.exhaustive import *
from library.config.genetic import *
from library.CVtrain import *
import scipy.io as sio

for i in range(3):
    t0 = time.time()

    N = 9 + i

    TrainFile = "datasets/N" + str(N) + "dataset10k.mat"
    EvalFile = "datasets/N" + str(N) + "dataset100k.mat"
    ModeName = "N"  + str(N) + "geneticResult"

    print("Genetically searching for best layer structure...")
    print("  Training on:    ", TrainFile)
    print("  Evaluating with:", EvalFile)
    layers = genetic_config(20, 5, TrainFile, 25, 150, 0, 500)
    print("Result: layers")

    evalSet = sio.loadmat(EvalFile)
    CVtrain([20,2,5], evalSet, 10, ModeName, batch=10, verbose=1, epochs=1000)
    t1 = time.time()
    print("MSE after 10-Fold CV:",getMSE(ModeName))
    print("Total Time:", (t1-t0)/60, "minutes")
