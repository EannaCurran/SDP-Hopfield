import torch
import os
import networkx as nx
os.chdir('..')

graphType = ["ColourGraphs"]
currentGraphType = graphType[0]
graphFolder = os.listdir(f"./Graphs/{currentGraphType}/Graph")

for graphFile in graphFolder:
    G = nx.read_edgelist(f"./Graphs/{currentGraphType}/Graph/{graphFile}", create_using=nx.Graph())
    G2 = nx.convert_node_labels_to_integers(G)
    nx.write_edgelist(G2, f"./Graphs/ColourGraphs/GraphClean/{graphFile}", data=False)