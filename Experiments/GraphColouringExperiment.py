import time

from Utils.utils import *
from Utils.hopfield import HopfieldNetworkColour
import os
import networkx as nx
import pandas as pd
import numpy as np

os.chdir('..')
np.set_printoptions(suppress=True)
graphType = ["ColourGraphs"]
currentGraphType = graphType[0]
graphFolder = sorted(os.listdir(f"./Graphs/{currentGraphType}/Graph2"))
executionTimes = []

for graphFile in graphFolder:
    invalid = False
    k = 17
    while not invalid:
        G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph2/{graphFile}", create_using=nx.Graph())
        graphSDP = pd.read_csv(f"./SDPColouring/{currentGraphType}/{graphFile}".replace(".txt", ".csv"), header=None)
        startTime = time.time()
        processedGraphSDP = process_graph_sdp_colour(graphSDP, k)
        hopfieldNetwork = HopfieldNetworkColour(processedGraphSDP, 100, 0, k, G)
        hopfieldNetwork.train()
        endTime = time.time()
        colouring = hopfieldNetwork.get_colouring()
        if check_valid_colouring(G, colouring):
            k-=1
        else:
            print(f"{graphFile}: Invalid {k} colouring best {k+1}")
            print(f"{graphFile} Time: {endTime-startTime} colouring best {k+1}")
            invalid = True
