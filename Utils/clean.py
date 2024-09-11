import torch
import os
import networkx as nx
os.chdir('..')

graphType = ["SF", "Twitter", "CustomCliqueSizes"]
size = 1024
relabel = dict()
currentGraphType = graphType[2]
graphFolder = os.listdir(f"./Graphs/{currentGraphType}/{size}/Graph")

for n in range(1,1025):
    relabel[str(n)] = n-1

for graphFile in graphFolder:
    G = nx.read_edgelist(f"./Graphs/{currentGraphType}/{size}/Graph/{graphFile}", create_using=nx.Graph())
    nx.relabel_nodes(G, relabel, copy=False)
    nx.write_edgelist(G, f"./Graphs/{currentGraphType}/{size}/Graph/{graphFile}", data=False)

