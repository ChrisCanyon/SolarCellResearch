from library.config.exhaustive import *
from library.config.genetic import *
from library.CVtrain import *
import scipy.io as sio

condensedOutput = [[18, 18, 4, 6, 0], [20, 3, 10, 0, 0], [17, 16, 14, 0, 0], [20, 14, 5, 0, 0], [19, 20, 9, 0, 0], [20, 2, 11, 0, 0], [19, 4, 13, 7, 0], [16, 20, 6, 0, 0], [17, 9, 0, 0, 0], [19, 5, 4, 4, 0], [18, 7, 10, 12, 0], [19, 17, 10, 0, 0], [20, 8, 7, 0, 0], [20, 3, 6, 0, 0], [13, 8, 12, 0, 0], [20, 19, 0, 0, 0], [19, 16, 20, 0, 0], [18, 20, 5, 0, 0], [19, 9, 0, 0, 0], [14, 14, 0, 0, 0]]

MSEs = []
total = 0
for i,layers in enumerate(condensedOutput):
    t0 = time.time()
    N = 15
    EvalFile = "datasets/N" + str(N) + "dataset100k.mat"

    evalSet = sio.loadmat(EvalFile)
    CVtrain(handle_zeros(layers), evalSet, 10, 'trash', batch=10, verbose=1, epochs=1000)
    t1 = time.time()
    MSEs.append(getMSE('trash'))
    total = total + getMSE('trash')
    print("MSE after 10-Fold CV:",getMSE('trash'))
    print("current avg MSE:", total/(i+1))
    print("Total Time:", (t1-t0)/60, "minutes")

'''

def sortByMSE(layers, MSEs):
   errors, structures = zip(*sorted(zip(MSEs, layers), key=lambda x: x[0], reverse=False))
   errors = list(errors)
   structures = list(structures)

   return structures


for i in range(10):
    N = 20
    L = 3
    generation = generate_initial_generation(N, L, 10)

    trainingSet = "dataset10k.mat"
    evaluateSet = "dataset1k.mat"

    smallBatch = 100
    bigBatch = 1000

    smallEpochs = 200
    bigEpochs = 500

    print("Layers options sorted by MSE")

    # with const num epochs
    smallBatchMSEs = train_generation(generation, trainingSet, evaluateSet, smallBatch, 0, bigEpochs)
    bigBatchMSEs = train_generation(generation, trainingSet, evaluateSet, bigBatch, 0, bigEpochs)

    print("With num epochs: ", bigEpochs)
    print("\t", smallBatch, "batchSize:", sortByMSE(generation, smallBatchMSEs))
    print("\t", bigBatch, "batchSize:", sortByMSE(generation, bigBatchMSEs))

    #with const batchSize
    smallEpochsMSEs = train_generation(generation, trainingSet, evaluateSet, bigBatch, 0, smallEpochs)
    bigEpochsMSEs = train_generation(generation, trainingSet, evaluateSet, bigBatch, 0, bigEpochs)

    print("With batchSize: ", bigBatch)
    print("\t", smallEpochs,"epochs:", sortByMSE(generation, smallEpochsMSEs))
    print("\t", bigEpochs, "epochs:", sortByMSE(generation, bigEpochsMSEs))

    # big big vs small small
    smallMSEs = train_generation(generation, trainingSet, evaluateSet, smallBatch, 0, smallEpochs)
    bigMSEs = train_generation(generation, trainingSet, evaluateSet, bigBatch, 0, bigEpochs)

    print("Large vs Small")
    print("\tSmall:", sortByMSE(generation, smallMSEs))
    print("\tLarge:", sortByMSE(generation, bigMSEs))
    
'''