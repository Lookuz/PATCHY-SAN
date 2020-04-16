import os
import sys
import math
import networkx as nx

# Computes betweenness centrality measure for given graph
# Takes networkx graph as input
def betweenness_centrality(graph):
    centrality_scores = nx.betweenness_centrality(graph, k=None)
    labelled_nodes = list(centrality_scores.items())
    
    return labelled_nodes

# TODO: Implement 1-WL normalization

# Computes distance between nodes in a graph using djikstra's algorithm
def compute_distance(graph, vertex):
    distances = []
    shortest_path_lengths = nx.single_source_dijkstra_path_length(graph, vertex)
    distances = list(shortest_path_lengths.items())
    distances = sorted(distances, key=lambda x: x[1])
    
    return distances

# Computes average number of nodes (rounded down)
# for a set of given graphs
def compute_average_node_count(graphs):
    total = 0
    for graph in graphs:
        total += len(graph.nodes())
    average_nodes = total/len(graphs)
    
    return round(average_nodes)