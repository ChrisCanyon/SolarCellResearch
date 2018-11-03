from library.config.genetic import *

for i in range(100):
    print(mutate([16, 17, 0, 0, 10], 10, 5))



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