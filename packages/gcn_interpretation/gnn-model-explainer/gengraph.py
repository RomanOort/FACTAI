"""gengraph.py

   Generating and manipulaton the synthetic graphs needed for the paper's experiments.
"""

import os

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Set matplotlib backend to file writing
plt.switch_backend("agg")

import networkx as nx

import numpy as np

from tensorboardX import SummaryWriter

from utils import synthetic_structsim
from utils import featgen
import utils.io_utils as io_utils


####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        print("orige: ", G.number_of_edges(), "orign: ", G.number_of_nodes())
        # exit()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
    """ Join two graphs along matching nodes, then perturb the resulting graph.
    Args:
        G1, G2: Networkx graphs to be joined.
        n_pert_edges: number of perturbed edges.
    Returns:
        A new graph, result of merging and perturbing G1 and G2.
    """
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F


def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    adj = np.array(nx.to_numpy_matrix(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}


####################################
#
# Generating synthetic graphs
#
###################################
def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5, f_val=1.0):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    basis_type = "ba"

    plt.figure(figsize=(8, 6), dpi=300)

    # if f_val == 1:
    list_shapes = [["house"]] * nb_shapes

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    # else:
    #     list_shapes = [["house_mod"]] * nb_shapes
    #
    #     G, role_id, _ = synthetic_structsim.build_graph(
    #         width_basis, basis_type, list_shapes, start=0, m=5
    #     )

    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)
    # feat_dict = {i: {'feat': f_val*np.ones(10, dtype=np.float32)} for i in G.nodes()}
    # nx.set_node_attributes(G, feat_dict)


    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn2(nb_shapes=100, width_basis=350):
    """ Synthetic Graph #2:

    Start with Barabasi-Albert graph and add node features indicative of a community label.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).

    Returns:
        G                 :  A networkx graph
        label             :  Label of the nodes (determined by role_id and community)
        name              :  A graph identifier
    """
    basis_type = "ba"

    random_mu = [0.0] * 8
    random_sigma = [1.0] * 8

    # Create two grids
    mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
    feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)

    G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4, f_val=1.0)
    G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4, f_val=-1.0)
    G1_size = G1.number_of_nodes()
    num_roles = max(role_id1) + 1
    role_id2 = [r + num_roles for r in role_id2]
    label = role_id1 + role_id2

    # Edit node ids to avoid collisions on join
    g1_map = {n: i for i, n in enumerate(G1.nodes())}
    G1 = nx.relabel_nodes(G1, g1_map)

    g2_map = {n: i + G1_size for i, n in enumerate(G2.nodes())}
    G2 = nx.relabel_nodes(G2, g2_map)

    # Join
    n_pert_edges = width_basis
    G = join_graph(G1, G2, n_pert_edges)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes) + "_2comm"

    return G, label, name


def gen_syn3(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
    """ Synthetic Graph #3:

    Start with Barabasi-Albert graph and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'grid') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["grid", 3]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    print("nodes: ", G.number_of_nodes(), "edges: ", G.number_of_edges())
    return G, role_id, name


def gen_syn4(nb_shapes=60, width_basis=8, feature_generator=None, m=4):
# def gen_syn4(nb_shapes=120, width_basis=10, feature_generator=None, m=4):

    """ Synthetic Graph #4:

    Start with a tree and attach cycle-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'Tree').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["cycle", 6]] * nb_shapes

    fig = plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.01)[0]
    # exit()
    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    # path = os.path.join("log/syn4_base_h20_o20")
    # writer = SummaryWriter(path)
    # io_utils.log_graph(writer, G, "graph/full")
    print("nodes: ", G.number_of_nodes(), "edges: ", G.number_of_edges())
    # exit()

    return G, role_id, name


def gen_syn5(nb_shapes=80, width_basis=8, feature_generator=None, m=3):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["grid", m]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.1)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    path = os.path.join("log/syn5_base_h20_o20")
    writer = SummaryWriter(path)

    return G, role_id, name

def gen_syn6(nb_shapes=120, width_basis=300, feature_generator=None, m=5):
    """ Synthetic Graph #6:

    Start with Barabasi-Albert graph and attach AA, AB, and BB cycles

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "ba"

    list_shapes = []
    for structure in ["cycleAA", "cycleAB", "cycleBB"]:
        list_shapes += [[structure]] * (nb_shapes // 3)
    list_shapes = list_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=role_id)
    nx.draw_networkx_edges(G, pos)
    plt.savefig("test.png")
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn7(nb_shapes=600, width_basis=0, feature_generator=None):
    """ Synthetic Graph #7:

    Add AA, AB, BB pairs 

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """

    # no basis graph with node pairs
    list_shapes = []
    for structure in ["dualPair"]:
        list_shapes += [[structure]] * (nb_shapes)
    list_shapes = list_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    label = []
    G = nx.Graph()
    feat_1 = [0, 1]
    feat_2 = [1, 0]
    feat_dict = {}
    label = []
    start = 0

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        G_size = G.number_of_nodes()

        Gi, roles_graph_i = synthetic_structsim.build_structure(args, shape_type) 
        n_s = nx.number_of_nodes(Gi)


        gi_map = {n: i + G_size for i, n in enumerate(Gi.nodes())}
        Gi = nx.relabel_nodes(Gi, gi_map)

        if shape_id % 2 == 0:
            for i, n in enumerate(Gi.nodes()):  
                if i % 2 == 0: 
                    feat_dict[n] = {'feat': np.array(feat_1, dtype=np.float32)}
                else: 
                    feat_dict[n] = {'feat': np.array(feat_1, dtype=np.float32)}
        else:
            for i, n in enumerate(Gi.nodes()):  
                if i % 2 == 0: 
                    feat_dict[n] = {'feat': np.array(feat_2, dtype=np.float32)}
                else:
                    feat_dict[n] = {'feat': np.array(feat_2, dtype=np.float32)}

        if G_size > 0:
            G = nx.compose(G, Gi)
            for n in Gi.nodes():
                node_1 = np.random.choice(G.nodes())
                while G.has_edge(node_1, n) or node_1 == n:
                    node_1 = np.random.choice(G.nodes())

                G.add_edge(node_1, n)
            # G = join_graph(G, Gi, 2)
        else:
            G = Gi
        start += n_s
    
    nx.set_node_attributes(G, feat_dict)

    G_size = nx.number_of_nodes(G)
    label = [0] * G_size
    # for i, n in enumerate(G.nodes()):
    #     label[i] = (1 if (feat_dict[n]['feat'] == feat_2).all() else 0)
    for i, n in enumerate(G.nodes()):
        label[i] = 1
        neih_feat = None
        for j, n_s in enumerate(G.neighbors(n)):
            # OPT a
            if neih_feat is not None and not (feat_dict[n_s]['feat'] == neih_feat).all():
                label[i] = 2
            neih_feat = feat_dict[n_s]['feat']

            # OPT b
            # if not (feat_dict[n_s]['feat'] == feat_dict[n]['feat']).all():
            #     label[i] = 2

    print(label.count(1))
    print(label.count(2))

    degrees = [n for i, n in G.degree()]
    print(max(degrees))
    print(min(degrees))
    color = [0 if (feat_dict[n]['feat'] == feat_1).all() else 1 for n in G.nodes()]
        
    # G, role_id, _ = synthetic_structsim.build_graph(
    #     width_basis, basis_type, list_shapes, start=0, m=5
    # )

    label_dict = {n: label[n] for n in G.nodes()}

    for i, n in enumerate(G.nodes()):
        if (feat_dict[n]['feat'] == feat_1).all():
            label_dict[n] = 1
        elif (feat_dict[n]['feat'] == feat_2).all():
            label_dict[n] = 2
        
        isFeat1 = True
        isFeat2 = True
        for j, n_s in enumerate(G.neighbors(n)):
            
            if (feat_dict[n_s]['feat'] == feat_1).all():
                isFeat2=False
            elif (feat_dict[n_s]['feat'] == feat_2).all():
                isFeat1=False

            # if not (feat_dict[n_s]['feat'] == feat_dict[n]['feat']).all() and (feat_dict[n]['feat'] == feat_1).all():
            #     label_dict[n] = 3
            # elif not (feat_dict[n_s]['feat'] == feat_dict[n]['feat']).all() and (feat_dict[n]['feat'] == feat_2).all():
            #     label_dict[n] = 4
        
        if not isFeat1 and not isFeat2:
            label_dict[n] = 3
        elif isFeat1:
            label_dict[n] = 2
        elif isFeat2:
            label_dict[n] = 1
        else:
            label_dict[n] = 0


    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=color)
    nx.draw_networkx_labels(G, pos, labels=label_dict)
    nx.draw_networkx_edges(G, pos)
    plt.savefig("syn7.png")

    G = perturb([G], 0.01)[0]

    # if feature_generator is None:
    #     feature_generator = featgen.ConstFeatureGen(1)
    # feature_generator.gen_node_features(G)

    name = "syn7" + "_" + str(nb_shapes)

    return G, label, name, label_dict

def gen_syn8(nb_shapes=120, width_basis=10, feature_generator=None, m=5):
    """ Synthetic Graph #8:

    Start with Barabasi-Albert graph and attach AB BC AC triplets (for cross-boundary classification)

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "ba"

    list_shapes = []
    for structure in ["triplet"]:
        list_shapes += [[structure]] * nb_shapes
    list_shapes = list_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )

    G_size = nx.number_of_nodes(G)
    label = [0] * G_size

    # basis
    feat_N = [1, 0, 0, 0, 0]
    # classifying node
    feat_P = [0, 1, 0, 0, 0]
    # relevant node features
    feat_A = [0, 0, 0, 0, 1]
    feat_B = [0, 0, 0, 1, 0]
    feat_C = [0, 0, 1, 0, 0]
    feat_dict = {}

    prev_label = 0
    prox_label = 0
    for i, n in enumerate(G.nodes()):
        # None
        if role_id[n] == 0:
            feat_dict[n] = {'feat': np.array(feat_N, dtype=np.float32)}
            label[n] = 0
        # classifier
        elif role_id[n] == 1:
            feat_dict[n] = {'feat': np.array(feat_P, dtype=np.float32)}
            prev_label = (prev_label + 1) % 3
            label[n] = prev_label + 1
            prox_label = 0
        # feat nodes
        elif role_id[n] == 2:
            this_label = (prev_label + prox_label) % 3
            if this_label == 0:
                feat_dict[n] = {'feat': np.array(feat_A, dtype=np.float32)}
            elif this_label == 1:
                feat_dict[n] = {'feat': np.array(feat_B, dtype=np.float32)}
            elif this_label == 2:
                feat_dict[n] = {'feat': np.array(feat_C, dtype=np.float32)}
            label[n] = 0
            prox_label += 1

    nx.set_node_attributes(G, feat_dict)
    role_id = label
    

    # plot graph
    color_lut = [
        (np.array(feat_A, dtype=np.float32), 1),
        (np.array(feat_B, dtype=np.float32), 2), 
        (np.array(feat_C, dtype=np.float32), 3), 
        (np.array(feat_P, dtype=np.float32), 4), 
        (np.array(feat_N, dtype=np.float32), 5)
    ]

    color_fn = lambda a: sum(v if (a==k).all() else 0 for k, v in color_lut)
    label_dict = {n: label[n] for n in G.nodes()}
    color_dict = [color_fn(feat_dict[n]['feat']) for n in G.nodes()]
    print(color_dict)
    pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_color=color)
    nx.draw_networkx_labels(G, pos, labels=label_dict)
    nx.draw_networkx_nodes(G, pos, node_color=color_dict)
    nx.draw_networkx_edges(G, pos)
    plt.savefig("syn8.png")

    G = perturb([G], 0.01)[0]

    # if feature_generator is None:
    #     feature_generator = featgen.ConstFeatureGen(1)
    # feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name
