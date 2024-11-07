import random

import networkx as nx
import numpy as np
import cvxpy as cp
import csv
import sys

import numpy.random
import torch
np.set_printoptions(threshold=sys.maxsize)


def nearest_psd(matrix):

    if is_psd(matrix):
        return matrix

    spacing = np.spacing(np.linalg.norm(matrix))
    identity = np.identity(len(matrix))
    k = 1

    while not is_psd(matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        matrix += identity * (- min_eig * (k ** 2) + spacing)
        k += 1

    return matrix


def is_psd(matrix):

    try:
        _ = np.linalg.cholesky(matrix)
        return True

    except np.linalg.LinAlgError:
        return False


def process_graph_no_sdp(graph, graphSDP):

    new_matrix = np.zeros(graphSDP.shape)

    for edge in graph.edges():
        new_matrix[int(edge[0]), int(edge[1])] = -1

    return new_matrix


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def generate_sdp_relaxation_cut(g, fileName, folderName):

    n = g.number_of_nodes()
    x = cp.Variable((n, n), PSD=True)

    obj = sum(0.5 * (1 - x[int(i), int(j)]) for i, j in g.edges)
    constr = [cp.diag(x) == 1]
    problem = cp.Problem(cp.Maximize(obj), constraints=constr)
    problem.solve(solver=cp.SCS)

    embedding = x.value

    with open(f"./SDPCut/{folderName}/{fileName.replace('.txt', '.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)
    return embedding


def process_graph_sdp_cut(graphSDP):

    angleMatrix = np.empty((graphSDP.shape[0],graphSDP.shape[0]))
    for m in range(0, graphSDP.shape[0]):
        for n in range(0, graphSDP.shape[0]):
            if n != m:
                angleMatrix[m][n] = angle_between_cut(np.dot(graphSDP[m], graphSDP[n]))
            else:
                angleMatrix[m][n] = 0

    return angleMatrix


def get_cut_value(graph, partition):

    in_cut = sum(1 for u, v in graph.edges() if partition[int(u)] != partition[int(v)])
    total = .5 * nx.adjacency_matrix(graph).sum()

    return in_cut / total


def get_partition(vectors):
    random = np.random.randn(vectors.shape[1])
    return np.sign(np.dot(vectors, random))


def angle_between_cut(dot):
    return 1-(2*(np.arccos(np.clip(dot, -1.0, 1.0)))) / np.pi


def generate_sdp_relaxation_clique(g, fileName, folderName):

    n = nx.number_of_nodes(g)+1
    X = cp.Variable((n, n), symmetric=True)
    I = np.identity(n)
    A = np.zeros((n, n))
    A[0][0] = 1
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    B[0][0] = 0
    constraints = [cp.trace(A @ X) == 1, cp.trace(B @ X) == 0, cp.multiply(C, X) == 0, X >> 0]

    for x in range(1, n):
        for y in range(1, n):
            if not g.has_edge(str(x-1), str(y-1)) and x-1 != y-1:
                C[x][y] = 1
                C[y][x] = 1

        B[x][x] = -1
        B[x][0] = 1
        C[0][x] = 0
        C[x][0] = 0

    obj = cp.Maximize(cp.trace(I @ X))
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)
    embedding = problem.variables()[0].value
    print(f"SDP Objective Value:{problem.value - 1}")

    with open(f"./temp/{fileName.replace('.txt', '.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)

    return embedding


def process_graph_sdp_clique(graph_sdp, g):

    dummy_vector = graph_sdp[:1].to_numpy()
    graph_sdp = graph_sdp.iloc[1:, 1:].to_numpy()
    angle_matrix = np.empty(graph_sdp.shape)
    dummy_vector = dummy_vector[0][1:]
    dummy_vector_copy = dummy_vector.copy()

    for m in range(0, len(graph_sdp)):
        for n in range(0, len(graph_sdp)):
            if g.has_edge(str(n), str(m)):
                angle_matrix[m][n] = 1 if graph_sdp[m][n] > 0.5 else 0

            elif n == m:
                angle_matrix[m][n] = 0

            else:
                angle_matrix[m][n] = -1

        dummy_vector[m] = angle_between_clique(dummy_vector_copy, graph_sdp[m])
    return angle_matrix, dummy_vector


def angle_between_clique(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return 2*np.dot(v1_u, v2_u)-1



def check_clique(index, graph):

    for n in range(0, len(index)):
        for m in range(n+1, len(index)):
            if not graph.has_edge(str(index[n]), str(index[m])):

                return False
    return True


def generate_sdp_relaxation_colouring(g, fileName, folderName):

    n = nx.number_of_nodes(g)
    X = cp.Variable((n, n), symmetric=True)
    t = cp.Variable(1, pos=False)
    constraints = []

    for x in range(0, n):
        for y in range(0, n):
            if g.has_edge(x, y):
                constraints.append(X[x, y] <= t)
            if x == y:
                constraints.append(X[x, y] == 1)
    constraints.append(X >> 0)
    obj = cp.Minimize(t)
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS, qcp=True)
    embedding = problem.variables()[1].value

    t = problem.value
    K = -(1/t)+1

    #print(f"Solver optimal solution:{t} Graph Colouring Objective Value:{K}")
    """
    with open(f"./SDPColouring/{folderName}/{fileName.replace('.txt', '.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)

    with open(f"./Graphs/{folderName}/resultsSDP.txt", "a") as file:
        file.write(f"{fileName.replace('.txt', '')} {K} \n")
    """
    return embedding


def process_graph_sdp_colour(graphSDP, k):

    angle_matrix = np.empty(graphSDP.shape)
    t = (-1)/(k-1) + 0.001

    for m in range(0, len(graphSDP)):
        for n in range(0, len(graphSDP)):
            if graphSDP[m][n] < t:
                angle_matrix[m][n] = -100
            elif m != n:
                angle_matrix[m][n] = graphSDP[m][n]
            else:
                angle_matrix[m][n] = 0

    return angle_matrix


def check_valid_colouring(G, colouring):

    for edge in G.edges():
        if (colouring[int(edge[0])] == colouring[int(edge[1])]).all():
            print(f"{edge[0]} {colouring[int(edge[0])]} {edge[1]} {colouring[int(edge[1])]} ")
            return False

    return True


def random_rounding(values):
    return [1 if x > 0 else -1 for x in values]


def process_graph_sdp_clique_model(graphSDP, g, graphValuesSDP, graphName):

    edge_features = []
    k = len(graphSDP)

    for n in range(0, k):
        for m in range(0, k):

            currentEdgeFeatures = [graphName, m, n, graphValuesSDP[m][n],
                                     (graphValuesSDP[m][n] * graphValuesSDP[m][n]) / 2,
                                     (graphValuesSDP[m][n] * graphValuesSDP[m][n] * graphValuesSDP[m][n]) / 3]

            if m == 0 or n == 0:
                # Is dummy node edge
                currentEdgeFeatures.append(1)
                # Not in original graph
                currentEdgeFeatures.append(0)

            else:
                # Is not dummy node edge
                currentEdgeFeatures.append(0)
                # Increment down to match with edge labels and already handling if m/n is 0

                if g.has_edge(m, n):
                    # In original graph
                    currentEdgeFeatures.append(1)
                else:
                    # Not in original graph
                    currentEdgeFeatures.append(0)

            currentEdgeFeatures.append(k-1)
            edge_features.append(currentEdgeFeatures)

    return edge_features


def process_graph_sdp_clique_model_train(G, model, currentGraphDataframe):

    nodeCount = G.number_of_nodes()
    angleMatrix = np.empty((nodeCount, nodeCount))
    dummyVector = np.empty(nodeCount)
    dummyVectorTensors = []
    angleMatrixTensors = []

    for m in range(1, nodeCount+1):
        row_tensors = []
        for n in range(1, nodeCount+1):
            if m != n:
                current_edge = currentGraphDataframe.loc[(currentGraphDataframe['Edge 1'] == m-1) & (currentGraphDataframe['Edge 2'] == n-1)]
                current_edge = torch.tensor(current_edge[['Xi', 'Xi^2', 'Xi^3', 'Di', 'Ii', 'Si']].values, dtype=torch.float32, requires_grad=True)
                value = model(current_edge)
                angleMatrix[m-1][n-1] = value
                row_tensors.append(value)
            else:
                angleMatrix[m-1][n-1] = 0
                current_edge = torch.tensor(0.0)
                row_tensors.append(current_edge)
        angleMatrixTensors.append(row_tensors)

    for m in range(0, nodeCount):
        current_edge = currentGraphDataframe.loc[(currentGraphDataframe['Edge 1'] == 0) & (currentGraphDataframe['Edge 2'] == m)]
        current_edge = torch.tensor(current_edge[['Xi', 'Xi^2', 'Xi^3', 'Di', 'Ii', 'Si']].values, dtype=torch.float32, requires_grad=True)
        value = model(current_edge)
        dummyVectorTensors.append(value)
        dummyVector[m] = value

    return angleMatrix, dummyVector, angleMatrixTensors, dummyVectorTensors


def process_graph_sdp_cut_model(graphSDP, g, graphValuesSDP, graphName):

    edgeFeatures = []
    for n in range(0, len(graphSDP) - 1):
        for m in range(0, len(graphSDP) - 1):

            currentEdgeFeatures = [graphName, n, m, graphValuesSDP[m][n],
                                     (graphValuesSDP[m][n] * graphValuesSDP[m][n])/2,
                                     (graphValuesSDP[m][n] * graphValuesSDP[m][n] * graphValuesSDP[m][n])/3]

            if g.has_edge(n, m):
                # In original graph
                currentEdgeFeatures.append(1)
            else:
                # Not in original graph
                currentEdgeFeatures.append(0)

            edgeFeatures.append(currentEdgeFeatures)

    return edgeFeatures


def process_graph_sdp_cut_model_train(G, model, currentGraphDataframe):

    nodeCount = G.number_of_nodes()
    angleMatrix = np.empty((nodeCount, nodeCount))
    angleMatrixTensors = []

    for m in range(0, nodeCount):
        rowTensors = []
        for n in range(0, nodeCount):
            if m != n:
                currentEdge = currentGraphDataframe.loc[(currentGraphDataframe['Edge 1'] == m) & (currentGraphDataframe['Edge 2'] == n)]
                currentEdge = torch.tensor(currentEdge[['Xi', 'Xi^2', 'Xi^3']].values, dtype=torch.float32, requires_grad=True)
                value = model(currentEdge)
                angleMatrix[m][n] = value
                rowTensors.append(value)
            else:
                angleMatrix[m][n] = 0
                currentEdge = torch.tensor(0.0)
                rowTensors.append(currentEdge)

        angleMatrixTensors.append(rowTensors)

    return angleMatrix, angleMatrixTensors


def greedy_max_clique(G):

    nodesByDegree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

    clique = [nodesByDegree[0]]
    possibleNodes = set(G.neighbors(clique[0]))

    for node in nodesByDegree[1:]:
        if node in possibleNodes:
            clique.append(node)
            possibleNodes.intersection_update(set(G.neighbors(node)))
            if not possibleNodes:
                break

    return clique


def read_file(path):

    newLines = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(" ")
            line = list(map(float, line[:-1]))
            newLines.append(line)
        return newLines


def generate_sdp_relaxation_mixing_cut(graph, k, iterations):

    vectors = []

    for node in graph.nodes():
        initialVector = numpy.random.randn(k)
        initialVector /= numpy.linalg.norm(initialVector)
        vectors.append(initialVector)

    for i in range(iterations):
        for node in graph.nodes():
            newVector = np.zeros(k)
            for neighbor in graph.neighbors(node):
                newVector += vectors[neighbor]
            newVector = -newVector/numpy.linalg.norm(newVector)
            vectors[node] = newVector

    return np.array(vectors)


def read_graph(path):

    graph = nx.Graph()
    sets = set()
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(" ")
            sets.add(int(line[0]))
            sets.add(int(line[1]))

            graph.add_node(int(line[0]))
            graph.add_node(int(line[1]))
            graph.add_edge(int(line[0]), int(line[1]))
    print(f"Number of nodes: {len(sets)}")
    return graph

# Not working :/
def generate_sdp_relaxation_mixing_clique(graph, k, iterations):
    vectors = []

    dummyVector = numpy.random.randn(k)

    for node in graph.nodes():
        initialVector = [value * random.uniform(0.9,1.1) for value in dummyVector]
        vectors.append(initialVector)

    for i in range(iterations):
        for node in graph.nodes():

            nonNeighbours = nx.non_neighbors(graph, node)
            num = len(list(nonNeighbours))

            if num != 0:
                nonNeighbours = nx.non_neighbors(graph, node)
                newVector = np.zeros(k)
                for nonNeighbour in nonNeighbours:
                    newVector += vectors[int(nonNeighbour)]
                newVector /= num
                newVector = newVector - np.dot(newVector, dummyVector)
                vectors[int(node)] = newVector
    dummyVectorCopy = np.copy(dummyVector)

    for node in graph.nodes():
        dummyVector = numpy.dot(vectors[int(node)], dummyVectorCopy)

    vectors.insert(0, dummyVector)
    return vectors