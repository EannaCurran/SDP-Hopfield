import time
import pandas as pd

from Utils.utils import *
from Utils.hopfield import HopfieldNetworkCut
import scipy
import os
import networkx as nx
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(1)
graphType = ["CustomCut", "SF", "Twitter", "CustomCutSizes"]
size = 256
sizeExp = False
currentGraphType = graphType[2]
maxCutSizes = dict()
os.chdir("..")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if not sizeExp:
    with open(f"Graphs/{currentGraphType}/cutSize.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            maxCutSizes[line[0]] = int(line[1])
    graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")

else:
    with open(f"Graphs/{currentGraphType}/{size}/cutSize.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            maxCutSizes[line[0]] = int(line[1])
        graphFolder = os.listdir(f"./Graphs/{currentGraphType}/{size}/Graphs")

#percisionRange = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
#roundRange = [1,2,3,4,5,6,7,8,9]
percisionRange = [1]
executionTimes = []

for m in reversed(percisionRange):
    hopCut = []
    normCut = []
    randCut = []
    convergenceCount = 0
    for graphFile in graphFolder:

        print(graphFile)
        if not sizeExp:
            G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
            graphSDP = pd.read_csv(f"./SDPCut/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
            #graphSDP = graphSDP.applymap(lambda x: x + random.uniform(-1, 1) / m)
            #graphSDP = graphSDP.round(graphSDP, m)
        else:
            G = nx.read_edgelist(f"./Graphs/{currentGraphType}/{size}/Graphs/{graphFile}", create_using=nx.Graph())
            G = nx.convert_node_labels_to_integers(G)
            graphSDP = pd.read_csv(f"./SDPCut/{currentGraphType}/{size}/{graphFile}".replace(".txt", ".csv"), header=None)
            #graphSDP = generate_sdp_relaxation_cut(G, graphFile, currentGraphType, size)
            #graphSDP = graphSDP.applymap(lambda x: x + random.uniform(-1, 1) / m)
            #graphSDP = graphSDP.round(graphSDP, m)

        embeddingDecomposition, _, _ = scipy.linalg.ldl(graphSDP)
        bestCut = 0
        for k in range(0, 5):
            startTime = time.time()
            cut = get_partition(embeddingDecomposition)
            value = get_cut_value(G, cut)
            endTime = time.time()
            executionTime = endTime - startTime
            cutSize = 0
            for edge in G.edges():
                if cut[int(edge[0])] != cut[int(edge[1])]:
                    cutSize += 1
            if cutSize > bestCut:
                bestCut = cutSize
        cutSize = bestCut

        bestCut = 0
        convergenceCheck = False

        for k in range(0, 5):
            processedGraphSDP = process_graph_sdp_cut(graphSDP)
            hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 10, 0)
            hopfieldNetwork.train()
            hopfieldCut, Con = hopfieldNetwork.get_partition()
            hopfieldCutSize = 0
            for edge in G.edges():
                if hopfieldCut[int(edge[0])] != hopfieldCut[int(edge[1])]:
                    hopfieldCutSize += 1
            if hopfieldCutSize > bestCut:
                bestCut = hopfieldCutSize
                executionTimes.append(executionTime)

            if Con == 0:
                convergenceCheck = True
        if not convergenceCheck:
            convergenceCount += 1

        hopfieldCutSize = bestCut
        bestCut = 0

        for k in range(0, 5):
            hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 10, 0)
            randomPartition, _ = hopfieldNetwork.get_partition()
            randomCutSize = 0
            for edge in G.edges():
                if randomPartition[int(edge[0])] != randomPartition[int(edge[1])]:
                    randomCutSize += 1
            if randomCutSize > bestCut:
                bestCut = randomCutSize
        randomCutSize = bestCut

        optimalCut = int(maxCutSizes[graphFile.replace('.txt', '')])
        hopCut.append(int(hopfieldCutSize) / optimalCut)
        normCut.append(int(cutSize) / optimalCut)
        randCut.append(int(randomCutSize) / optimalCut)
        print(optimalCut)
        print(f"Best GW Cut: {cutSize} Best Hopfield Cut: {hopfieldCutSize}  Best Random Cut:{randomCutSize} Optimal Cut: {optimalCut} GW Optimality Gap: {int(cutSize) / optimalCut} Hopfield Optimality Gap: {int(hopfieldCutSize) / optimalCut} Random Optimality Gap: {int(randomCutSize) / optimalCut}")

    print(f"{1/(m*10)}: Average Best GW Cut: {np.mean(normCut)}+-{np.std(normCut)} Average Best Hopfield Cut: {np.mean(hopCut)}+-{np.std(hopCut)} Average Best Random Cut: {np.mean(randCut)}+-{np.std(randCut)}")
    print(f"Average runtime for Hopfield:{np.mean(executionTimes)}+-{np.std(executionTimes)}")
