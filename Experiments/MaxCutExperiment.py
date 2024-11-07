import pandas as pd
from Utils.utils import *
from Utils.hopfield import HopfieldNetworkCut, HopfieldNetworkCutTorch
import scipy
import os
import networkx as nx
import random
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(1)
graphType = ["CustomCut", "SF", "Twitter"]

currentGraphType = graphType[1]
maxCutSizes = dict()
os.chdir('..')
torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

with open(f"Graphs/{currentGraphType}/cutSize.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n").split(" ")
        maxCutSizes[line[0]] = int(line[1])

graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")
precisionRange = [1]

for m in reversed( precisionRange):
    hopCut = []
    convergenceCount = []
    normCut = []
    randCut = []
    times = []
    nonConvergenceCount = 0
    for graphFile in graphFolder:

        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
        #G = nx.convert_node_labels_to_integers(G, first_label=0)
        graphSDP = pd.read_csv(f"./SDPCut/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        #time1 = time.time()
        #graphSDP = generate_sdp_relaxation_cut(G, graphFile, currentGraphType)
        #time2 = time1 - time.time()
        #times.append(time2)
        embedd, _, _ = scipy.linalg.ldl(graphSDP)
        bestCut = 0

        for k in range(0, 5):
            cut = get_partition(embedd)
            cutSize = 0
            for edge in G.edges():
                if cut[int(edge[0])] != cut[int(edge[1])]:
                    cutSize += 1
            if cutSize > bestCut:
                bestCut = cutSize
        cutSize = bestCut
        bestCut = 0

        processedGraphSDP = process_graph_sdp_cut(graphSDP)
        convergenceCheck = False
        averageConvergence = 0

        for k in range(0, 5):
            hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 20, 0)
            hopfieldNetwork.train()
            hopfieldCut, Con = hopfieldNetwork.get_partition()
            averageConvergence += Con
            hopfieldCutSize = 0
            for edge in G.edges():
                if hopfieldCut[int(edge[0])] != hopfieldCut[int(edge[1])]:
                    hopfieldCutSize += 1
            if hopfieldCutSize > bestCut:
                bestCut = hopfieldCutSize
            if Con == 0:
                convergenceCheck = True

        if not convergenceCheck:
            nonConvergenceCount += 1
        hopfieldCutSize = bestCut

        bestCut = 0
        for k in range(0, 5):
            randomCutSize = 0
            randomPartition = np.random.randint(0, 2, nx.number_of_nodes(G))
            randomPartition[randomPartition == 0] = -1
            for edge in G.edges():
                if randomPartition[int(edge[0])] != randomPartition[int(edge[1])]:
                    randomCutSize += 1
            if randomCutSize > bestCut:
                bestCut = randomCutSize
        randomCutSize = bestCut

        optimalCut = int(maxCutSizes[graphFile.replace('.txt', '')])
        hopCut.append(int(hopfieldCutSize) / optimalCut)
        convergenceCount.append(averageConvergence)
        normCut.append(int(cutSize) / optimalCut)
        randCut.append(int(randomCutSize) / optimalCut)

        print(f"Best GW Cut: {cutSize} Best Hopfield Cut: {hopfieldCutSize}  Best Random Cut:{randomCutSize} Optimal Cut: {optimalCut} GW Optimality Gap: {int(cutSize) / optimalCut} Hopfield Optimality Gap: {int(hopfieldCutSize) / optimalCut} Random Optimality Gap: {int(randomCutSize) / optimalCut}")

    print(f"{1/(m*10)}:Average Best GW Cut: {np.mean(normCut)}+-{np.std(normCut)} Average Best Hopfield Cut: {np.mean(hopCut)}+-{np.std(hopCut)} Average Best Random Cut: {np.mean(randCut)}+-{np.std(randCut)}")
    print(f"Average Best Hopfield Cut: {np.mean(hopCut)}+-{np.std(hopCut)} Average Convergence Count:{np.mean(convergenceCount)}+-{np.std(convergenceCount)}")
