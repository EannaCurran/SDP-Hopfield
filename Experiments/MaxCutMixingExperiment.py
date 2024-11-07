import math
import os
import time
import random
from Utils.utils import *
from Utils.hopfield import HopfieldNetworkCut, HopfieldNetworkCutTorch

random.seed(1)
graphType = ["CustomCut", "SF", "Twitter", "GSET"]
currentGraphType = graphType[3]
maxCutSizes = dict()
os.chdir('..')
torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if currentGraphType != "GSET":
    with open(f"Graphs/{currentGraphType}/cutSize.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            maxCutSizes[line[0]] = int(line[1])

graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")
hopCut = []

for graphFile in graphFolder:

    G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph(), nodetype=int)
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    print(f"{graphFile} Nodes:{nx.number_of_nodes(G)} Edges:{nx.number_of_edges(G)}")

    k = math.ceil(math.sqrt(2*nx.number_of_nodes(G)))
    timer = time.time()
    graphSDP = generate_sdp_relaxation_mixing_cut(G, k, 100)
    SDPTime = time.time() - timer

    bestCut = 0
    timer = time.time()

    processedGraphSDP = process_graph_sdp_cut(graphSDP)
    for k in range(0, 5):
        hopfieldNetwork = HopfieldNetworkCut(processedGraphSDP, 20, 0)
        hopfieldNetwork.train()
        hopfieldCut, Con = hopfieldNetwork.get_partition()
        hopfieldCutSize = 0
        for edge in G.edges():
            if hopfieldCut[int(edge[0])] != hopfieldCut[int(edge[1])]:
                hopfieldCutSize += 1
        if hopfieldCutSize > bestCut:
            bestCut = hopfieldCutSize

    HopfieldTime = time.time() - timer

    hopfieldCutSize = bestCut
    if currentGraphType != "GSET":

        optimalCut = int(maxCutSizes[graphFile.replace('.txt', '')])
    else:
        optimalCut = 1

    hopCut.append(int(hopfieldCutSize) / optimalCut)
    print(f"Best Hopfield Cut: {hopfieldCutSize} Optimal Cut: {optimalCut} Hopfield Optimality Gap: {int(hopfieldCutSize) / optimalCut}")
    print(f"Mixing time:{SDPTime} Hopfield Time:{HopfieldTime} \n")
print(f"Average Best Hopfield Cut: {np.mean(hopCut)}+-{np.std(hopCut)}")
