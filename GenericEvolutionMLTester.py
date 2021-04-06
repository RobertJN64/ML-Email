import MachineLearning.GeneticEvolution as ge
import MachineLearning.GeneticNets as gn
import MachineLearning.NetRender as nr

#config
fname = "titanic.json"
generations = 20
populationSize = 20
trainSize = 400
testSize = 100
evoRate = 4

#specific config
midDepth = 10
midWidth = 1

screen = nr.screen()
rSettings = nr.stdSettings(screen)
rSettings["settings"]["vdis"] = 15

dataset, trainset, testset, metadata = ge.loadDataset(fname, testSize)


DB = gn.Random(metadata["inputs"], metadata["outputs"], populationSize, midWidth, midDepth)

bests = []

import time
startTime = time.time()

for genCount in range(0, generations):
    evoRate = evoRate * 0.95
    DB, best, bestscore, truescore = ge.Test(DB, dataset, trainset, trainSize, testset, renderSettings = rSettings, testMode="Absolute")
    bests.append([best,truescore])
    DB = ge.evolve(DB, evoRate)
    screen.bestNet(best, bestscore, truescore)

best, row = ge.getHighest(bests)
gn.saveNets([best], "generic-net-save", "Nets from file: " + fname, 2)
nr.stop()

print(time.time() - startTime)

