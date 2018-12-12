from library.config.exhaustive import *
from library.config.genetic import *
from library.CVtrain import *
import scipy.io as sio
import sys

if len(sys.argv) < 3:
    print("Error: please provide a number of cells per sensor and value of i")
    exit()

N = sys.argv[1]
i = sys.argv[2]
t0 = time.time()

TrainFile = "datasets/N" + str(N) + "dataset10k" + str(i) + ".mat"
EvalFile = "datasets/N" + str(N) + "dataset100k" + str(i) + ".mat"
ModeName = "N"  + str(N) + "geneticResult"
layers = [15, 5, 5]

evalSet = sio.loadmat(EvalFile)
[MSE, weightsAndBiases] = CVtrain(layers, evalSet, 5, ModeName, batch=10, verbose=1, epochs=1000)
finalErrors001 = compute_errors([MSE], [weightsAndBiases], 0.001)
finalErrors0025 = compute_errors([MSE], [weightsAndBiases], 0.0025)

t1 = time.time()
print("Total Time:", (t1-t0)/60, "minutes")
print("CellsPerSensor:", N)
print(" Resulting Network Architecture:", layers)
print(" MSE after 5-Fold CV:", MSE)
print(" Weights and Biases:", weightsAndBiases)
print(" Objective Function Value(0.001):", finalErrors001[0])
print(" Objective Function Value(0.0025):", finalErrors0025[0])
