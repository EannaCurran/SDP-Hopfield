from Utils.utils import *
from Utils.hopfield import HopfieldNetworkColour
import os
import networkx as nx
import pandas as pd
import numpy as np
import time

os.chdir('..')
np.set_printoptions(suppress=True)
graphType = ["ColourGraphs"]
currentGraphType = graphType[0]
graphFolder = sorted(os.listdir(f"./Graphs/{currentGraphType}/Graph2"))

for graphFile in graphFolder:
    invalid = False
    k = 17
    while not invalid:
        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph2/{graphFile}", create_using=nx.Graph())
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        graphSDP = pd.read_csv(f"./SDPColouring/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        time1 = time.time()
        #graphSDP = generate_sdp_relaxation_colouring(G,graphFile,"temp")

        invalid=True

        processedGraphSDP = process_graph_sdp_colour(graphSDP, k)
    
        hopfieldNetwork = HopfieldNetworkColour(processedGraphSDP, 50, 0, k, G)
        hopfieldNetwork.train()
        colouring = hopfieldNetwork.get_colouring()
        time2 = time.time()
        if check_valid_colouring(G, colouring):
            k-=1
        else:
            print(f"{graphFile}: Invalid {k} colouring best {k+1}")
            print(f"Time taken {time2 - time1}")
            invalid = True
