import time

import numpy as np

from Utils.utils import *
from Utils.hopfield import HopfieldNetworkClique
import os
import networkx as nx
import pandas as pd
import random

os.chdir('..')
graphType = ["IMDB-BINARY", "COLLAB", "Twitter", "CustomClique", "CustomCliqueSizes"]
currentGraphType = graphType[1]
random.seed(1)
maxCliqueSizes = dict()
size = 1024

sizeExperiment = False
if currentGraphType == 'CustomCliqueSizes':
    with open(f"Graphs/{currentGraphType}/{size}/cliqueSize.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            maxCliqueSizes[line[0]] = int(line[1])
else:
    with open(f"Graphs/{currentGraphType}/cliqueSize.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            maxCliqueSizes[line[0]] = int(line[1])

graphFolder = os.listdir(f"./Graphs/{currentGraphType}/{size}/Graph") if sizeExperiment else os.listdir(f"./Graphs/{currentGraphType}/Graph")
maxCliqueNotFound = 0
invalidClique = 0
nonConvergenceCount = 0
hopCliquesOpt = []
executionTimes = []
for graphFile in graphFolder:
    if sizeExperiment:
        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/{size}/Graph/{graphFile}", create_using=nx.Graph())
        graphSDP = pd.read_csv(f"./SDPClique/{size}/{graphFile}".replace(".txt", ".csv"), header=None)

    else:
        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
        graphSDP = pd.read_csv(f"./SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)


    #graphSDP = generate_sdp_relaxation_clique(G, graphFile, size)
    bestCliqueSize = 0

    for t in range(0, 5):
        maxClique = maxCliqueSizes[graphFile.replace(".txt", "")]
        cliques = [len(c) for c in nx.find_cliques(G)]
        maxClique = max(cliques)
        startTime = time.time()
        processedGraphSCP, dummyNode = process_graph_sdp_clique(graphSDP, G)
        hopfieldNetwork = HopfieldNetworkClique(processedGraphSCP, dummyNode, 30, 0)
        hopfieldNetwork.train()
        endTime = time.time()
        executionTime = endTime - startTime
        print(executionTime)

        executionTimes.append(executionTime)
        hopfieldPartition, Con = hopfieldNetwork.get_partition()
        index = np.where(hopfieldPartition == 1)[0]
        hopfieldCount = np.count_nonzero(hopfieldPartition == 1)

        if Con == -1:
            nonConvergenceCount += 1

        currentScore = 0
        if not check_clique(index, G):
            invalidClique += 1
            hopCliquesOpt.append(0)
            currentScore = 0
        elif hopfieldCount != maxClique:
            maxCliqueNotFound += 1
            hopCliquesOpt.append(hopfieldCount/maxClique)
            currentScore = hopfieldCount/maxClique
        else:
            hopCliquesOpt.append(1)
            currentScore = 1

        if currentScore > bestCliqueSize:
            bestCliqueSize = currentScore

        print(hopfieldCount)
    if bestCliqueSize == 0:
        invalidClique += 1
    if bestCliqueSize != 1:
        maxCliqueNotFound += 1

    hopCliquesOpt.append(bestCliqueSize)
    print(graphFile)
    print(f"Max Clique Size:{maxClique} Hopfield Opt Gap:{bestCliqueSize} Graph Nodes:{nx.number_of_nodes(G)}")

print(f"Max Cliques Not Found:{maxCliqueNotFound}  Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}")
print(f"Average Clique Opt Ratio: {np.mean(hopCliquesOpt)}+-{np.std(hopCliquesOpt)}")
print(f"Average runtime for Hopfield:{np.mean(executionTimes)}+-{np.std(executionTimes)}")

