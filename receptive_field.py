import os
import sys
import networkx as nx
from util import betweenness_centrality, compute_distance
from pynauty.graph import Graph, canonical_labeling

labelling_procedures = {
    'betweenness' : betweenness_centrality
}

# Class that generates a receptive field for a graph
class ReceptiveField():
    # graph: Input graph to generate receptive field on
    # w: Number of neighbourhoods. Automatically calculated if None
    # s: Stride of sequence selection
    # k: Neighbourhood size
    def __init__(self, graph, w, s=1, k=10, l='betweenness', default=None, attribute_name='node_attributes', rank_name='node_rank', num_attr=None):
        self.graph = graph
        self.w = w
        self.s = s
        self.k = k
        self.l = l
        self.node_labels = {}
        # Attribute names(Optional)
        self.attribute_name = attribute_name
        self.rank_name = rank_name
        # Default attribute initialization
        self.default_attributes = {}
        if not default: # Automatically fixes zero values
            # Handle empty graphs
            if not list(self.graph.nodes()) and num_attr:
                self.default_attributes[attribute_name] = 0.0 if num_attr == 1 else [0.0 for i in range(num_attr)]
            else:
                first_node = list(self.graph.nodes)[0]
                attributes = self.graph.nodes(data=True)[first_node]
                for k, v in attributes.items():
                    self.default_attributes[k] = 0.0 if type(v) != list else [0.0 for i in range(len(v))]
        else:
            self.default_attributes = default
        
    # Computes the ordering for the nodes given the set labelling procedure
    # Returns sorted ordered list of tuple of (node, score)
    def compute_labelling(self, graph):
        labelling_procedure = labelling_procedures[self.l]
        labelled_nodes = labelling_procedure(graph)
        labelled_nodes = sorted(labelled_nodes, key=lambda x: x[1], reverse=True)
        
        return labelled_nodes
    
    # Creates all receptive fields for the current graph
    # with respect to w,s,k and l
    def make_all_receptive_fields(self):
        top_w = self.select_node_sequence()
        ordered_nodes = [x[0] for x in top_w]
        f = []
        i,j = 0, 0
        while j < self.w:
            if i < len(ordered_nodes):
                f.append(self.make_receptive_field(ordered_nodes[i], self.node_labels))
            else:
                f.append(self.make_zero_receptive_field()) # Stride s may be too large
            i += self.s
            j += 1
            
        return f

    # Selects the top w elements according the labelling l
    def select_node_sequence(self):
        node_labels = self.compute_labelling(self.graph)
        self.node_labels = dict(node_labels) # Internal memory of node labelling
        node_labels = node_labels[:self.w] # Top w elements
        
        return node_labels

    # Creates the receptive field for the input graph
    # by calling neighbourhood assembly and graph normalization methods
    def make_receptive_field(self, v, node_labels):
        # Assembles the neighbourhood
        local_neighbourhood = self.assemble_neighbourhoods(v)
        # Normalizes the graph
        normalized_neighbourhood = self.normalize_graph(local_neighbourhood, v, dict(node_labels))
        
        # Reshaping and relabelling
        nb_tensor = []
        for i, attributes in list(normalized_neighbourhood.nodes(data=True)):
            # Sort by computed ranking, and 
            nb_tensor.append((attributes[self.rank_name], attributes[self.attribute_name]))
        nb_tensor = [x[1] for x in sorted(nb_tensor)]
        
        return nb_tensor
    
    # Creates receptive field with default values
    def make_zero_receptive_field(self):
        zero_padded_graph = nx.star_graph(self.k - 1)
        # Initialize default values for all nodes
        for k, v in self.default_attributes.items():
            nx.set_node_attributes(zero_padded_graph, v, k)
            
        default_rank = {i:{self.rank_name : i+1} for i in range(self.k)}
        nx.set_node_attributes(zero_padded_graph, default_rank)
        
        # Reshaping and relabelling
        zero_rf_tensor = []
        for i, attributes in list(zero_padded_graph.nodes(data=True)):
            # Sort by computed ranking, and 
            zero_rf_tensor.append((attributes[self.rank_name], attributes[self.attribute_name]))
        zero_rf_tensor = [x[1] for x in sorted(zero_rf_tensor)]
        
        return zero_rf_tensor
        
    # Assemble neighbourhood around vertex v
    def assemble_neighbourhoods(self, v):
        N = set() # Current set of vertices in neighourhood
        L = set() # Neighbours of N
        N.add(v)
        L.add(v)
        while len(N) < self.k and len(L) > 0:
            for v in L:
                L = L.union(set(self.graph.neighbors(v)))
            L = L - N # Get only new neighbours
            N = N.union(L)
            
        return self.graph.subgraph(N)
    
    # Impose order using graph normalization subject to labelling
    def normalize_graph(self, neighbourhood, v, node_labels):
        # Compute ranking:
        # d(u, v) < d(w, v) -> r(u) < r(v)
        ranking = self.compute_ranking(neighbourhood, v, node_labels, canonical=True)
        
        # |U| > k
        if len(ranking) > self.k:
            top_k = [x for x,_ in ranking[:self.k]]
            subgraph = nx.Graph(neighbourhood.subgraph(top_k))
            
            # Recompute sugraph ranking
            subgraph_labels = dict(self.compute_labelling(subgraph))
            subgraph_ranking = self.compute_ranking(subgraph, v, subgraph_labels, canonical=True)
        # |U| < k
        elif len(ranking) < self.k:
            subgraph = nx.Graph(neighbourhood)
            subgraph, subgraph_ranking = self.pad_graph(subgraph, ranking, self.k)
        # |U| = k
        else:
            subgraph = nx.Graph(neighbourhood)
            subgraph_ranking = ranking
            
        nx.set_node_attributes(subgraph, dict(subgraph_ranking), self.rank_name)
        # return self.canonicalize(subgraph)
        return subgraph
    
    # Computes the ranking of the graph with respect to vertex v
    # Node labels should be dictionary form
    def compute_ranking(self, graph, v, node_labels, canonical=False):
        distance = compute_distance(graph, v)
        
        # Compute ranking wrt to distance then node_labels
        distinct_distances = set([y for x,y in distance])
        partitioned_list = []
        for d in distinct_distances:
            current_nodes = [x for x,y in distance if y == d]
            # Sort by labelling first, then break ties with canonical labelling
            if canonical:
                canonical_labels = self.canonicalize(graph)
                
                current_nodes = sorted(current_nodes, key=lambda x: (-node_labels[x], -canonical_labels[x]))
            else:
                current_nodes = sorted(current_nodes, key=lambda x: -node_labels[x])
            partitioned_list.append(current_nodes)
        
        ranking = [x for sub in partitioned_list for x in sub]
        ranking = [(x,i + 1) for i,x in enumerate(ranking)]
        
        return ranking
    
    # Pads the graph with additional dummy nodes
    # Until total number of nodes is N
    def pad_graph(self, graph, ranking, N):
        padded_graph = nx.Graph(graph)
        node_next = max([x for x,_ in ranking])
        rank_next = max([y for _,y in ranking])
        step = 1
        while len(padded_graph.nodes()) < N:
            padded_graph.add_node(node_next + step, **self.default_attributes)
            ranking.append((node_next + step, rank_next + step))
            step += 1
            
        return padded_graph, ranking
    
    # Canonicalize the graph to find isomorphism
    # Returns the canonical labelling to the graph
    def canonicalize(self, graph):
        # Relabel nodes for canonicalization
        original_labels = list(graph.nodes())
        original_to_relabeled = {x:i for i,x in enumerate(original_labels)}
        relabeled_to_original = {i:x for i,x in enumerate(original_labels)}
        relabeled_graph = nx.relabel_nodes(graph, original_to_relabeled)
        
        nauty_graph = Graph(len(relabeled_graph), directed=False)
        # Adjacency dictionary for relabeled graph
        adjacency_dict = {n: list(nbrs) for n, nbrs in relabeled_graph.adjacency()}
        nauty_graph.set_adjacency_dict(adjacency_dict)
        canonical_labels = canonical_labeling(nauty_graph)
        
        # Break ties with nauty
        canonical_labels = {k:canonical_labels[k] for k in range(len(relabeled_graph))}
        
        # Switch back to original labels
        canonical_labels = {relabeled_to_original[i]:canonical_labels[i] for i in range(len(relabeled_graph))}
        
        return canonical_labels
                  
              