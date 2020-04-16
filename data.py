import os
import sys
import networkx as nx
from collections import defaultdict

data_path = './data/'

# Mapping of graph datasets to folder names
dataset_dict = {
    'mutag' : 'MUTAG'
}

# All possible file extensions for data files
file_extensions = {
    'compulsory' : [
        'A', # Adjacency list
        'graph_labels',
        'node_labels'
    ],
    'optional' : [
        'edge_labels' # Edge attributes
    ]
}

# Loads graph dataset in compact format and processes them into nx graph objects
def load_compact_data(file_name):
    graph_data = process_compact_files(data_path + file_name + '/', file_name)

    class_labels = graph_data['class_labels']
    adjacency_list = graph_data['adjacency_list']
    graph_indicator = graph_data['graph_indicator']
    # Node attributes
    try:
        node_labels = graph_data['node_labels']
    except KeyError:
        pass
    
    try:
        node_attributes = graph_data['node_attributes']
    except KeyError:
        pass
    
    # TODO: Add edge attributes
    data = []

    N = len(class_labels)
    for i in range(N):
        x = nx.Graph()

        # Loop through adjacency list for current graph
        for node in graph_indicator[i+1]:
            if node not in dict(x.nodes()):
                x.add_node(node)

            # Register attributes and labels
            try:
                node_label = node_labels[node]
                x.add_node(node, node_label=node_label)
            except NameError:
                pass
            
            try:
                node_attr = node_attributes[node]
                x.add_node(node, node_attributes=node_attr)
            except NameError:
                pass

            # Add edges
            for neighbour in adjacency_list[node]:
                x.add_edge(node, neighbour)

        data.append((x, class_labels[i])) 

    return data

# Reads the data from graph datasets in compact format
# and returns the respective read data in the form of a dictionary
# that maps data_type -> data
def process_compact_files(data_path, file_name):
    # Compulsory files
    class_labels = read_graph_labels(data_path, file_name + '_graph_labels.txt')
    adjacency_list = read_adjacency_list(data_path, file_name + '_A.txt')
    graph_indicator = read_graph_indicator(data_path, file_name + '_graph_indicator.txt')

    graph_data = {
        'class_labels' : class_labels,
        'adjacency_list' : adjacency_list,
        'graph_indicator' : graph_indicator
    }

    # Optional files
    # Node labels
    try:
        node_labels = read_node_labels(data_path, file_name + '_node_labels.txt')
        graph_data['node_labels'] = node_labels
    except FileNotFoundError as e:
        print(e)
        
    # Node attributes
    try:
        node_attributes = read_node_attributes(data_path, file_name + '_node_attributes.txt')
        graph_data['node_attributes'] = node_attributes
    except FileNotFoundError as e:
        print(e)

    return graph_data
    

#################################
#   Driver code for file I/O    #
#################################

# Partitions lines based into a list of lines using 
# the given delimiter, else whitespace
def partition(lines, delimiter=lambda x : x.isspace()):
    section = []
    for line in lines:
        if delimiter(line):
            # Segment current section if non-empty
            if section:
                yield section
                section = []
        else:
            section.append(line.strip())

    if section:
        yield section

def read_graph_labels(data_path, file_name):
    labels = []

    with open(data_path + file_name, 'r') as f:
        y = list(partition(f))

        labels = [int(label) for y_0 in y for label in y_0]

    return labels

def read_adjacency_list(data_path, file_name):
    # Adjacency list in the form of dictionary
    adjacency_list = defaultdict(list)

    with open(data_path + file_name, 'r') as f:
        adjacency_lines = list(partition(f))

        for line in adjacency_lines[0]:
            u, v =  line.split(',')
            adjacency_list[int(u)].append(int(v))
        
    return adjacency_list

def read_graph_indicator(data_path, file_name):
    graph_to_nodes = defaultdict(list)

    with open(data_path + file_name, 'r') as f:
        sections = list(partition(f))

        for node_id, graph_id in enumerate(sections[0]):
            graph_to_nodes[int(graph_id)].append(int(node_id) + 1)

    return graph_to_nodes

def read_node_attributes(data_path, file_name):
    node_attributes = {}

    with open(data_path + file_name, 'r') as f:
        lines = list(partition(f))

        for node_id, attributes in enumerate(lines[0]):
            node_attributes[node_id + 1] = [float(attribute) for attribute in attributes.split(',')]

    return node_attributes

def read_node_labels(data_path, file_name):
    node_labels = {}

    with open(data_path + file_name, 'r') as f:
        labels = list(partition(f))
        
        for node_id, label in enumerate(labels[0]):
            node_labels[node_id + 1] = int(label)

    return node_labels

# TODO: Add reading of edge attributes
