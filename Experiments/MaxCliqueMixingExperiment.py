from Utils.utils import *
from Utils.hopfield import HopfieldNetworkClique
import os
import networkx as nx
import pandas as pd
import random

os.chdir('..')
graphType = ["IMDB-BINARY", "COLLAB", "Twitter", "CustomClique"]
currentGraphType = graphType[0]
random.seed(1)
maxCliqueSizes = dict()

with open(f"Graphs/{currentGraphType}/cliqueSize.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n").split(" ")
        maxCliqueSizes[line[0]] = int(line[1])

graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")
maxCliqueNotFound = 0
invalidClique = 0
nonConvergenceCount = 0
hopCliquesOpt = []

randomCliqueOpt = []
for graphFile in graphFolder:
    G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
    #G = nx.convert_node_labels_to_integers(G, first_label=0)
    k = nx.number_of_nodes(G)
    graphSDP = generate_sdp_relaxation_mixing_clique(G, k, 100)
    graphSDP = pd.DataFrame(graphSDP)
    graphSDP2 = pd.read_csv(f"./SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)


    bestCliqueSize = 0
    bestRandomCliqueSize = 0

    cliques = [len(c) for c in nx.find_cliques(G)]
    maxClique = max(cliques)

    for n in range(0,1):
        processedGraphSCP, dummyNode = process_graph_sdp_clique(graphSDP, G)
        hopfieldNetwork = HopfieldNetworkClique(processedGraphSCP, dummyNode, 5, 0)

        hopfieldNetwork.train()
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

    for t in range(0,5):

        currentScore = len(greedy_max_clique(G))/maxClique
        if currentScore > bestRandomCliqueSize:
            bestRandomCliqueSize = currentScore

    hopCliquesOpt.append(bestCliqueSize)
    randomCliqueOpt.append(bestRandomCliqueSize)
    print(graphFile)
    print(f"Max Clique Size:{maxClique} Hopfield Opt Gap:{bestCliqueSize} Random Clique Opt Gap:{bestRandomCliqueSize} Graph Nodes:{nx.number_of_nodes(G)}")

print(f"Max Cliques Not Found:{maxCliqueNotFound}  Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}")
print(f"Average Clique Opt Ratio: {np.mean(hopCliquesOpt)}+-{np.std(hopCliquesOpt)}")
print(f"Average Random Clique Opt Ratio: {np.mean(randomCliqueOpt)}+-{np.std(randomCliqueOpt)}")
