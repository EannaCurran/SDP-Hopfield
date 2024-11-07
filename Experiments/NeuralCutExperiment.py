
import copy
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from Utils.hopfield import HopfieldNetworkCut
from Utils.utils import *

graphType = ["CustomCut", "SF", "Twitter"]
currentGraphType = graphType[1]
os.chdir('..')
createDataset = False
device = torch.device('cuda')
random.seed(1)
maxCutSizes = dict()

with open(f"./Graphs/{currentGraphType}/cutSize.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n").split(" ")
        maxCutSizes[line[0]+'.txt'] = int(line[1])

if createDataset:

    graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")
    graphs = []

    for graphFile in graphFolder:
        print(graphFile)
        originalGraph = nx.read_edgelist(f"Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph(),
                                         nodetype=int)
        graphSDPValues = pd.read_csv(f"SDPCut/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        graphSDP = nx.complete_graph(nx.number_of_nodes(originalGraph) + 1)
        edgeFeatures = process_graph_sdp_cut_model(graphSDP, originalGraph, graphSDPValues, graphFile)
        for currentGraphEdge in edgeFeatures:
            graphs.append(currentGraphEdge)

    df = pd.DataFrame(graphs, columns= ['Graph Name', 'Edge 1', 'Edge 2', 'Xi', 'Xi^2', 'Xi^3', 'Ii'])
    df.to_pickle(f'./Dataframe/SDPCut/{currentGraphType}.pkl')

df = pd.read_pickle(f'./Dataframe/SDPCut/{currentGraphType}.pkl')

model = nn.Sequential(
    nn.Linear(3, 6),
    nn.Tanh(),
    nn.Linear(6, 6),
    nn.Tanh(),
    nn.Linear(6, 1),
    nn.Tanh()
)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
torch.nn.init.xavier_uniform_(model[0].weight)
torch.nn.init.xavier_uniform_(model[2].weight)
torch.nn.init.xavier_uniform_(model[4].weight)

train = False

graphCount = df['Graph Name'].nunique()
trainSize = int(np.floor(graphCount * 0.7))
testSize = int(graphCount - trainSize)
graphNames = list(df['Graph Name'].unique())
trainGraphs = random.sample(graphNames, trainSize)
testGraphs = [n for n in graphNames if n not in trainGraphs]

trainDf = df[df['Graph Name'].isin(trainGraphs)]
testDf = df[df['Graph Name'].isin(testGraphs)]

n_epochs = 50
best_ratio = 0
best_weights = None

if train:

    model.train()
    nonConvergenceCount = 0

    for epoch in range(n_epochs):

        print(epoch)
        results = []
        for graphFile in trainGraphs[0:700]:

            G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
            graphSDP = pd.read_csv(f"./SDPCut/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
            currentGraphDataframe = trainDf.loc[trainDf['Graph Name'] == graphFile]

            processedGraphSDP, tensorsSDP = process_graph_sdp_cut_model_train(G, model, currentGraphDataframe)
            hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 5, 0)
            hopfieldNetwork.train()
            hopfieldCut, Con = hopfieldNetwork.get_partition()
            hopfieldCutSize = 0

            for edge in G.edges():
                if hopfieldCut[int(edge[0])] != hopfieldCut[int(edge[1])]:
                    hopfieldCutSize += 1
            if Con == -1:
                nonConvergenceCount += 1

            maxCutSize = maxCutSizes[graphFile]
            cutError = 1-(hopfieldCutSize/maxCutSize)
            results.append(hopfieldCutSize/maxCutSize)
            hopfieldNetwork.back_propagate(cutError)

            startingGradients = hopfieldNetwork.get_history()[0]

            hopfieldStartingGradients = startingGradients['weight error']

            optimizer.zero_grad()
            norm = (len(tensorsSDP) * (len(tensorsSDP)-1))/2

            for m in range(0, len(tensorsSDP)):
                 for n in range(m, len(tensorsSDP)):
                    if n != m:
                        tensor = tensorsSDP[m][n]
                        gradient = hopfieldStartingGradients[m][n]
                        gradient = torch.tensor([[gradient]])
                        tensor.backward(gradient=gradient/norm)

            optimizer.step()

            print(f"{graphFile.replace('.txt,', '')} Number of Nodes:{nx.number_of_nodes(G)} Hopfield Cut Size:{hopfieldCutSize} Cut Size:{maxCutSize} Opt Ratio:{hopfieldCutSize/maxCutSize} Non-convergence Count:{nonConvergenceCount}")

        print(f"Average Ratio:{np.mean(results)}")

        with open("./result.txt", "a") as myfile:
            myfile.write(f"{np.mean(results)}\n")

        if np.mean(results) > best_ratio:
            best_weights = copy.deepcopy(model.state_dict())
            best_ratio = np.mean(results)

    model.load_state_dict(best_weights)
    #torch.save(model.state_dict(), f'./Models/MaxCut/{currentGraphType}23_Model.pt')

else:
    results = []
    nonConvergenceCount = 0
    model.load_state_dict(torch.load(f'./Models/MaxCut/{currentGraphType}23_Model.pt'))
    model.eval()

    for graphFile in testGraphs[0:300]:

        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
        graphSDP = pd.read_csv(f"./SDPCut/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        currentGraphDataframe = testDf.loc[testDf['Graph Name'] == graphFile]

        processedGraphSDP, tensorsSDP = process_graph_sdp_cut_model_train(G, model, currentGraphDataframe)
        hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 10, 0)
        hopfieldNetwork.train()
        hopfieldCut, Con = hopfieldNetwork.get_partition()
        hopfieldCutSize = 0

        for edge in G.edges():
            if hopfieldCut[int(edge[0])] != hopfieldCut[int(edge[1])]:
                hopfieldCutSize += 1

        if Con == -1:
            nonConvergenceCount += 1

        maxCutSize = maxCutSizes[graphFile]
        results.append(hopfieldCutSize / maxCutSize)
        print(f"{graphFile.replace('.txt,', '')} Number of Nodes:{nx.number_of_nodes(G)} Hopfield Cut Size:{hopfieldCutSize} Cut Size:{maxCutSize} Opt Ratio:{hopfieldCutSize / maxCutSize} Non-convergence Count:{nonConvergenceCount}")

    print(f"Average Ratio for Test Set:{np.mean(results)}")