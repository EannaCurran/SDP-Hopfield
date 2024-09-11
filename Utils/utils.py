import networkx as nx
import numpy as np
import cvxpy as cp
import pandas as pd
import csv
import sys
import torch
import gurobipy as gp
from gurobipy import GRB
import warnings
from torch import nn
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


def process_graph_no_sdp(graph, graph_sdp):

    new_matrix = np.zeros(graph_sdp.shape)

    for edge in graph.edges():
        new_matrix[int(edge[0]), int(edge[1])] = -1

    return new_matrix


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def generate_sdp_relaxation_cut(g, file_name, folder_name,size):

    n = g.number_of_nodes() + 1
    x = cp.Variable((n, n), PSD=True)

    obj = sum(0.5 * (1 - x[int(i), int(j)]) for i, j in g.edges)
    constr = [cp.diag(x) == 1]
    problem = cp.Problem(cp.Maximize(obj), constraints=constr)
    problem.solve(solver=cp.SCS)

    embedding = problem.variables()[0].value

    with open(f"./SDPCut/{folder_name}/{size}/{file_name.replace('.txt','.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)
    return embedding


def process_graph_sdp_cut(graph_sdp):

    angle_matrix = np.empty(graph_sdp.shape)

    for m in range(0, len(graph_sdp)):
        for n in range(0, len(graph_sdp)):
            if n != m:
                angle_matrix[m][n] = angle_between_cut(graph_sdp[m][n])
            else:
                angle_matrix[m][n] = 0

    return angle_matrix


def get_cut_value(graph, partition):

    in_cut = sum(1 for u, v in graph.edges() if partition[int(u)] != partition[int(v)])
    total = .5 * nx.adjacency_matrix(graph).sum()

    return in_cut / total


def get_partition(vectors):

    random = np.random.normal(size=vectors.shape[1])
    random /= np.linalg.norm(random, 2)

    return np.sign(np.dot(vectors, random))


def angle_between_cut(dot):
    return 1-(2*(np.arccos(np.clip(dot, -1.0, 1.0)))) / np.pi


def generate_sdp_relaxation_clique(g, file_name, folder_name):

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
    print(f"SDP Objective Value:{problem.value-1}")
    with open(f"./SDPClique/{folder_name}/{file_name.replace('.txt', '.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)
    return embedding


def process_graph_sdp_clique(graph_sdp, g):

    dummy_vector = graph_sdp[:1].to_numpy() if isinstance(graph_sdp, pd.DataFrame) else graph_sdp[:1]
    graph_sdp = graph_sdp.iloc[1:, 1:].to_numpy() if isinstance(graph_sdp, pd.DataFrame) else graph_sdp[1:, 1:]
    angle_matrix = np.empty(graph_sdp.shape)
    dummy_vector = dummy_vector[0][1:]
    dummy_vector_copy = dummy_vector.copy()
    dummy_vector_copy_copy = dummy_vector.copy()

    for m in range(0, len(graph_sdp)):
        for n in range(0, len(graph_sdp)):
            if g.has_edge(str(n), str(m)):
                angle_matrix[m][n] = 0

            elif n == m:
                angle_matrix[m][n] = 0

            else:
                angle_matrix[m][n] = -1

        dummy_vector_copy_copy[m] = angle_between_clique(dummy_vector_copy, graph_sdp[m])

    return angle_matrix, dummy_vector_copy_copy


def angle_between_clique(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return  2 * np.dot(v1_u, v2_u) - 1


def check_clique(index, graph):
    for n in range(0, len(index)):
        for m in range(n+1, len(index)):
            if not graph.has_edge(str(index[n]), str(index[m])):
                return False
    return True


def generate_sdp_relaxation_colouring(g, file_name, folder_name):

    n = nx.number_of_nodes(g)
    X = cp.Variable((n, n), symmetric=True)
    t = cp.Variable(1, pos=False)
    constraints = [X >> 0]

    for x in range(0, n):
        for y in range(0, n):
            if g.has_edge(str(x), str(y)):
                constraints.append(X[x, y] <= t)
            if x == y:
                constraints.append(X[x, y] == 1)

    obj = cp.Maximize(1/t)
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS, qcp=True)
    embedding = problem.variables()[1].value
    t = problem.value
    K = -(1/t)+1
    print(f"Solver optimal solution:{t} Graph Colouring Objective Value:{K}")
    '''
    with open(f"./SDPColouring/{folder_name}/{file_name.replace('.txt', '.csv')}", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(embedding)

    with open(f"./Graphs/{folder_name}/resultsSDP.txt", "a") as file:
        file.write(f"{file_name.replace('.txt', '')} {K} \n")
    '''
    return embedding


def process_graph_sdp_colour(graph_sdp, k):
    angle_matrix = np.empty(graph_sdp.shape)
    t = (-1)/(k-1) + 0.001
    for m in range(0, len(graph_sdp)):
        for n in range(0, len(graph_sdp)):
            if graph_sdp[m][n] < t:
                angle_matrix[m][n] = -100
            elif m != n:
                angle_matrix[m][n] = graph_sdp[m][n]
            else:
                angle_matrix[m][n] = 0
    return angle_matrix


def check_valid_colouring(G, colouring):

    for edge in G.edges():
        if (colouring[int(edge[0])] == colouring[int(edge[1])]).all():
            #print(f"{edge[0]} {colouring[int(edge[0])]} {edge[1]} {colouring[int(edge[1])]} ")
            return False
    return True


def random_rounding(values):
    return [1 if x > 0 else -1 for x in values]


def process_graph_sdp_clique_model(graph_sdp, g, graph_sdp_values, graph_name):

    edge_features = []
    sdpSize = len(graph_sdp)
    for n in range(0, sdpSize):
        for m in range(0, sdpSize):

            current_edge_features = [graph_name, n, m, graph_sdp_values[n][m],
                                     graph_sdp_values[n][m] * graph_sdp_values[n][m],
                                     graph_sdp_values[n][m] * graph_sdp_values[n][m] * graph_sdp_values[n][m]]

            if m == 0 or n == 0:
                # Is dummy node edge
                current_edge_features.append(1)
                # Not in original graph
                current_edge_features.append(0)
            else:
                # Is not dummy node edge
                current_edge_features.append(0)
                # Increment down to match with edge labels and already handling if m/n is 0
                if g.has_edge(n-1, m-1):
                    # In original graph
                    current_edge_features.append(1)
                else:
                    # Not in original graph
                    current_edge_features.append(0)

            #current_edge_features.append(sdpSize-1)
            edge_features.append(current_edge_features)

    return edge_features

def process_graph_sdp_clique_model_train(G, model, currentGraphDataframe):

    node_count = G.number_of_nodes()
    angle_matrix = np.empty((node_count, node_count))
    dummy_vector = np.empty(node_count)
    dummy_vector_tensors = []
    angle_matrix_tensors = []
    scaler = np.sqrt(node_count)
    for m in range(1, node_count+1):
        row_tensors = []
        for n in range(1, node_count+1):
            if m != n:
                current_edge = currentGraphDataframe.loc[(currentGraphDataframe['Edge 1'] == m) & (currentGraphDataframe['Edge 2'] == n)]
                current_edge = torch.tensor(current_edge[['Xi', 'Xi^2', 'Xi^3', 'Di', 'Ii']].values, dtype=torch.float32, requires_grad=True)
                value = model(current_edge)
                angle_matrix[m-1][n-1] = value
                row_tensors.append(value)
            else:
                angle_matrix[m-1][n-1] = 0
                current_edge = torch.tensor(0.0)
                row_tensors.append(current_edge)
        angle_matrix_tensors.append(row_tensors)

    for m in range(0, node_count):
        current_edge = currentGraphDataframe.loc[(currentGraphDataframe['Edge 1'] == 0) & (currentGraphDataframe['Edge 2'] == m)]
        current_edge = torch.tensor(current_edge[['Xi', 'Xi^2', 'Xi^3', 'Di', 'Ii']].values, dtype=torch.float32, requires_grad=True)
        value = model(current_edge)
        dummy_vector_tensors.append(value)
        dummy_vector[m] = value

    return angle_matrix, dummy_vector, angle_matrix_tensors, dummy_vector_tensors


def process_graph_sdp_cut_model(graph_sdp, g, graph_sdp_values, graph_name):

    edge_features = []
    for n in range(0, len(graph_sdp)-1):
        for m in range(0, len(graph_sdp)-1):

            current_edge_features = [graph_name, n, m, graph_sdp_values[m][n],
                                     graph_sdp_values[m][n] * graph_sdp_values[m][n],
                                     graph_sdp_values[m][n] * graph_sdp_values[m][n] * graph_sdp_values[m][n]]

            if g.has_edge(n, m):
                # In original graph
                current_edge_features.append(1)
            else:
                # Not in original graph
                current_edge_features.append(0)

            edge_features.append(current_edge_features)

    return edge_features


def process_graph_sdp_cut_model_train(G, model, current_graph_dataframe):

    node_count = G.number_of_nodes()
    angle_matrix = np.empty((node_count, node_count))
    angle_matrix_tensors = []

    for m in range(0, node_count):

        row_tensors = []

        for n in range(0, node_count):

            if m != n:

                current_edge = current_graph_dataframe.loc[(current_graph_dataframe['Edge 1'] == m) & (current_graph_dataframe['Edge 2'] == n)]
                current_edge = torch.tensor(current_edge[['Xi', 'Xi^2', 'Xi^3']].values, dtype=torch.float32, requires_grad=True)
                value = model(current_edge)
                angle_matrix[m][n] = value
                row_tensors.append(value)
            else:
                angle_matrix[m][n] = 0
                current_edge = torch.tensor(0.0)
                row_tensors.append(current_edge)

        angle_matrix_tensors.append(row_tensors)

    return angle_matrix, angle_matrix_tensors


def solve_max_cut(G):

    model = gp.Model("max_cut")
    model.setParam('TimeLimit', 1.0)
    model.setParam('OutputFlag', 0)
    x = model.addVars(G.nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[i] + x[j] - 2 * x[i] * x[j]for i, j in G.edges), GRB.MAXIMIZE)

    model.optimize()
    cut = [i for i in G.nodes if x[i].X > 0.5]

    return cut, model.objVal


