import copy
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim, nn
import tqdm

from Utils.utils import *
from Utils.hopfield import HopfieldNetworkClique
import os
import networkx as nx
import pandas as pd
import random

os.chdir('..')
graphType = ["IMDB-BINARY", "COLLAB", "Twitter", "CustomClique"]
currentGraphType = graphType[3]
random.seed(1)
maxCliqueSizes = dict()
createDataset = True
device = torch.device('cpu')

with open(f"Graphs/{currentGraphType}/cliqueSize.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n").split(" ")
        maxCliqueSizes[line[0]] = int(line[1])

if createDataset:

    graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")
    graphs = []

    for graphFile in graphFolder:

        print(graphFile)
        originalGraph = nx.read_edgelist(f"Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph(), nodetype=int)
        graphSDPValues = pd.read_csv(f"SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        graphSDP = nx.complete_graph(nx.number_of_nodes(originalGraph)+1)
        edgeFeatures = process_graph_sdp_clique_model(graphSDP, originalGraph, graphSDPValues, graphFile)
        for currentGraphEdge in edgeFeatures:
            graphs.append(currentGraphEdge)

    df = pd.DataFrame(graphs, columns= ['Graph Name', 'Edge 1', 'Edge 2', 'Xi', 'Xi^2', 'Xi^3', 'Di', 'Ii'])
    df.to_pickle(f'./Dataframe/SDPClique/{currentGraphType}.pkl')

df = pd.read_pickle(f'./Dataframe/SDPClique/{currentGraphType}.pkl')

class ModifiedSequentialModel(nn.Module):
    def __init__(self):
        super(ModifiedSequentialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.model(x)
        return x

    def ModRange(self, x):
        scaled_output = 2 * x - 1
        return scaled_output

model = ModifiedSequentialModel()
torch.nn.init.xavier_uniform_(model.model[0].weight)
torch.nn.init.xavier_uniform_(model.model[2].weight)
torch.nn.init.xavier_uniform_(model.model[4].weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train = False

graphCount = df['Graph Name'].nunique()
trainSize = int(np.floor(graphCount * 0.7))
testSize = int(graphCount - trainSize)
graphNames = list(df['Graph Name'].unique())
trainGraphs = random.sample(graphNames, trainSize)
testGraphs = [n for n in graphNames if n not in trainGraphs]
testGraphs = graphNames

trainDf = df[df['Graph Name'].isin(trainGraphs)]
testDf = df[df['Graph Name'].isin(testGraphs)]


n_epochs = 100
best_ratio = 0
best_weights = None
history = []
if train:

    for epoch in range(n_epochs):

        print(epoch)
        model.train()

        maxCliqueNotFound = 0
        invalidClique = 0
        nonConvergenceCount = 0
        hopCliquesOpt = []

        for graphFile in trainGraphs[0:700]:
            G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
            graphSDP = pd.read_csv(f"./SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
            cliques = [c for c in nx.find_cliques(G)]
            cliqueSize = [len(c) for c in cliques]
            cliqueIndex = max(cliques, key=len)
            maxClique = max(cliqueSize)
            bestCliqueSize = 0
            currentGraphDataframe = df.loc[df['Graph Name'] == graphFile]
            processedGraphSDP, dummyNode, tensorsSDP, tensorDummy = process_graph_sdp_clique_model_train(G, model, currentGraphDataframe)

            hopfieldNetwork = HopfieldNetworkClique(processedGraphSDP, dummyNode, 5, 0)
            hopfieldNetwork.train()
            hopfieldPartition, Con = hopfieldNetwork.get_partition()
            index = np.where(hopfieldPartition == 1)[0]
            hopfieldClique = np.count_nonzero(hopfieldPartition == 1)

            if Con == -1:
                nonConvergenceCount += 1

            if not check_clique(index, G):
                invalidClique += 1
                hopCliquesOpt.append(0)
            elif hopfieldClique != maxClique:
                maxCliqueNotFound += 1
                hopCliquesOpt.append(hopfieldClique / maxClique)
            else:
                hopCliquesOpt.append(1)

            if not check_clique(index, G):
                if len(index) < maxClique:
                    hopfieldError = -1
                    hopfieldErrorBias= -1
                else:
                    hopfieldError = 1
                    hopfieldErrorBias= 1
            else:
                hopfieldError = -(1 - (hopfieldClique / maxClique))
                hopfieldErrorBias = -(1 - (hopfieldClique / maxClique))

            hopfieldNetwork.back_propagate(hopfieldError)
            startingGradients = hopfieldNetwork.get_history()[0]
            hopfieldStartingGradients = startingGradients['weight error']

            hopfieldNetwork.back_propagate(hopfieldErrorBias)
            startingGradients = hopfieldNetwork.get_history()[0]
            hopfieldBiasGradients = startingGradients['bias error']

            optimizer.zero_grad()
            norm = ((len(tensorsSDP)) * (len(tensorsSDP)+1)) / 2
            if hopfieldError != 0:

                for m in range(0, len(hopfieldStartingGradients)):
                    for n in range(0, len(hopfieldStartingGradients)):
                        if m != n:
                            tensor = tensorsSDP[m][n]
                            gradient = hopfieldStartingGradients[m][n]
                            gradient = torch.tensor([[gradient]])
                            tensor.backward(gradient=gradient/norm)

                for m in range(0, len(hopfieldStartingGradients)):
                    tensor = tensorDummy[m]
                    gradient = hopfieldBiasGradients[m]
                    gradient = torch.tensor([[gradient]])
                    tensor.backward(gradient=gradient/(norm*norm))

                optimizer.step()
            optimizer.zero_grad()

            print(f"{graphFile.replace('.txt,','')}, Number of Nodes:{nx.number_of_nodes(G)} Hopfield Count:{hopfieldClique} Max Clique Size:{maxClique} Invalid Clique Counter:{invalidClique} Convergence Count:{Con}")
        message = f"Max Cliques Not Found:{maxCliqueNotFound} Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}"
        print(message)
        print(f"Average Clique Opt Ratio: {np.mean(hopCliquesOpt)}")

        with open("./results.txt", "a") as myfile:
            myfile.write(f"{np.mean(hopCliquesOpt)} {message}\n")

        if np.mean(hopCliquesOpt) > best_ratio:
            best_weights = copy.deepcopy(model.state_dict())
            best_ratio = np.mean(hopCliquesOpt)

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f'./Models/MaxClique/{currentGraphType}_Model2.pt')

else:
        model_loaded = torch.load(f'./Models/MaxClique/IMDB-BINARY_Model2.pt')
        model.load_state_dict(model_loaded)
        model.eval()

        hopClique = []
        maxCliqueNotFound = 0
        invalidClique = 0
        nonConvergenceCount = 0
        hopCliquesOpt = []
        executionTimes = []
        invalidCount = 0
        testGraphs = testGraphs[0:300]
        for graphFile in testGraphs:
            G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
            graphSDP = pd.read_csv(f"./SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)

            currentGraphDataframe = df.loc[df['Graph Name'] == graphFile]
            cliques = [len(c) for c in nx.find_cliques(G)]
            maxClique = max(cliques)

            startTime = time.time()
            processedGraphSDP, dummyNode, _, _ = process_graph_sdp_clique_model_train(G, model, currentGraphDataframe)
            endTime = time.time()
            executionTime = endTime - startTime
            bestCliqueSize = 0
            currentScore = 0

            for k in range(0, 5):
                startTime = time.time()
                hopfieldNetwork = HopfieldNetworkClique(processedGraphSDP, dummyNode, 5, 0)
                hopfieldNetwork.train()
                endTime = time.time()
                hopTime = endTime - startTime
                hopfieldPartition, Con = hopfieldNetwork.get_partition()
                index = np.where(hopfieldPartition == 1)[0]
                hopfieldClique = np.count_nonzero(hopfieldPartition == 1)

                if Con == -1:
                    nonConvergenceCount += 1

                if not check_clique(index, G):
                    currentScore = 0
                    if bestCliqueSize == 0:
                        bestTime = executionTime + hopTime
                else:
                    currentScore = min(hopfieldClique/maxClique, 1)
                    if currentScore > bestCliqueSize:
                        bestCliqueSize = currentScore
                        bestTime = executionTime + hopTime

            if currentScore == 0:
                invalidCount += 1
            else:
                hopCliquesOpt.append(bestCliqueSize)
            executionTimes.append(bestTime)

            print(f"{graphFile.replace('.txt,', '')}, Number of Nodes:{nx.number_of_nodes(G)} Hopfield Count:{bestCliqueSize} Max Clique Size:{maxClique} Invalid Clique Counter:{invalidClique} Convergence Count:{Con}")
        message = f"Max Cliques Not Found:{maxCliqueNotFound} Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}"

        print(f"Max Cliques Not Found:{maxCliqueNotFound}  Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}")
        print(f"Percentage of Invalid Cliques found:{invalidCount/len(testGraphs)}")
        print(f"Average Clique Opt Ratio: {np.mean(hopCliquesOpt)}+-{np.std(hopCliquesOpt)}")
        print(f"Average runtime for Hopfield:{np.mean(executionTimes)}+-{np.std(executionTimes)}")