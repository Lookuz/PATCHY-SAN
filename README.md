# PATCHY-SAN

Implementation of Learning Convolutional Neural Networks for Graphs by Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov, ICML 2014-2023. https://scholar.google.com.sg/scholar?oi=bibs&cluster=9917957179670149192&btnI=1&hl=en

Required Modules:
- `networkx`
- `pynauty` (See https://web.cs.dal.ca/~peter/software/pynauty/html/install.html for installation)
- `tensorflow`, `keras`
- `numpy`

Python 3 Implementation that uses the `networkx` library to construct input graphs, and transform them into receptive fields, which are tensors that form the inputs into a Convolutional Neural Network(CNN) for learning of subgraph properties and approximate isomoprhisms, based on the choice of node labelling procedures(e.g Between-ness Centrality). In particular, the algorithm does the following:

- Select node sequence according to the labelling procedure
- Assemble local neighbourhoods from subgraphs centered on the nodes selected the node sequence
- Perform graph normalization on neighbourhoods to form receptive fields
- Learn subgraph properties using receptive fields as inputs

The `Conv1D`, `Dense`, `Dropout` and `softmax` layers from `keras` are used to construct the deep CNN that performs classification of graphs according to their respective subgraph receptive fields. This is represented in the `pscn.py` file that contains the module using the Patchy-San algorithm with a CNN.

Various well known graph datasets for measuring performance on graph kernels were used in evaluating the PSCN algorithm. https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
