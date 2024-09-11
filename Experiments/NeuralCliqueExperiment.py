import copy
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
createDataset = False
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
        processedGraphSDP = process_graph_sdp_cut(graphSDPValues)
        graphSDP = nx.complete_graph(nx.number_of_nodes(originalGraph))

        for edge in graphSDP.edges():
            if graphSDPValues[int(edge[0])][int(edge[1])] != 0:
                currentGraph = [graphFile, edge[0], edge[1]]

                currentValueSDP = graphSDPValues[int(edge[0])][int(edge[1])]
                currentGraph.append(currentValueSDP)
                currentGraph.append(currentValueSDP * currentValueSDP)
                currentGraph.append(currentValueSDP * currentValueSDP * currentValueSDP)
                currentGraph.append(processedGraphSDP[int(edge[0])][int(edge[1])])
                graphs.append(currentGraph)

    df = pd.DataFrame(graphs, columns= ['Graph Name', 'Edge 1', 'Edge 2', 'Xi', 'Xi^2', 'Xi^3', 'Y'])
    df.to_pickle(f'./Dataframe/SDPClique/{currentGraphType}.pkl')

df = pd.read_pickle(f'./Dataframe/SDPClique/{currentGraphType}.pkl')

model = nn.Sequential(
    nn.Linear(3, 12),
    nn.ReLU(),
    nn.Linear(12, 1)
)

lossFn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train = False

if train:
    X = df[['Xi', 'Xi^2', 'Xi^3']]
else:
    X = df[['Graph Name', 'Xi', 'Xi^2', 'Xi^3']]


y = df[['Y']]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.7, shuffle=False)
graphNames = set([item[0] for item in X_test])
n_epochs = 10
batch_size = 100
best_mse = np.inf
best_weights = None
history = []

if train:

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    batch_start = torch.arange(0, len(X_train), batch_size)
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                y_pred = model(X_batch)
                loss = lossFn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(mse=float(loss))

        model.eval()
        y_pred = model(X_test)
        mse = lossFn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f'./Models/MaxClique/{currentGraphType}_Model.pt')
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()

else:
    model_loaded = torch.load(f'./Models/MaxClique/{currentGraphType}_Model.pt')
    model.load_state_dict(model_loaded)
    model.eval()

    hopClique = []
    maxCliqueNotFound = 0
    invalidClique = 0
    nonConvergenceCount = 0
    hopCliquesOpt = []

    for graphFile in graphNames:
        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
        graphSDP = pd.read_csv(f"./SDPClique/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)

        cliques = [len(c) for c in nx.find_cliques(G)]
        maxClique = max(cliques)
        bestCliqueSize = 0
        processedGraphSCP, dummyNode = process_graph_sdp_clique_model(graphSDP, G, model)
        hopfieldNetwork = HopfieldNetworkClique(processedGraphSCP, dummyNode, 20, 0)
        hopfieldNetwork.train()
        hopfieldPartition, Con = hopfieldNetwork.get_partition()
        index = np.where(hopfieldPartition == 1)[0]
        hopfieldCount = np.count_nonzero(hopfieldPartition == 1)

        if Con == -1:
            nonConvergenceCount += 1

        if not check_clique(index, G):
            invalidClique += 1
            hopCliquesOpt.append(0)
        elif hopfieldCount != maxClique:
            maxCliqueNotFound += 1
            hopCliquesOpt.append(hopfieldCount / maxClique)
        else:
            hopCliquesOpt.append(1)

        print(graphFile)
        print(f"Max Clique Size:{maxClique} Hopfield Number of 1s:{np.count_nonzero(hopfieldPartition == 1)} Iterations:{Con} Graph Nodes:{nx.number_of_nodes(G)}")

    print(f"Max Cliques Not Found:{maxCliqueNotFound}  Invalid Cliques Found:{invalidClique} Non Convergence:{nonConvergenceCount}")
    print(f"Average Clique Opt Ratio: {np.mean(hopCliquesOpt)}")
