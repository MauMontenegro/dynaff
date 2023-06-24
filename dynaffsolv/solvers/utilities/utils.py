import os
import json
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ete3 import Tree


class ExperimentLog:
    def __init__(self, path, file_name):
        self.path = path
        self.file = file_name + '.json'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.full_path = os.path.join(self.path, self.file)

    def log_save(self, stats):
        with open(self.full_path, 'w') as fp:
            json.dump(stats, fp)


# Performance
def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())


def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak / (1024 * 1024)
    print("Peak Size in MB - ", peak)
    return peak


def saveSolution(instance_path, instance, solution, saved, time, hash_calls, hash_size, solver_name):
    summary_name = solver_name + "_summary"
    output_path = instance_path / instance / summary_name
    with open(output_path, "w") as writer:
        writer.write("Solution: {}\n".format(solution))
        writer.write("Saved: {}\n".format(saved))
        writer.write("RunTime: {}\n".format(time))
        writer.write("Hash_Calls: {}\n".format(hash_calls))
        writer.write("Hash Size: {}\n".format(hash_size))


def Statistics(path, total_saved, total_times):
    """
    :param path: Path to save statistics for experiment
    :param total_saved: Total saved nodes array for each instance
    :param total_times: Total execution time per instance
    :return: Nothing
    """
    time_mean = []
    time_std_dv = []
    saved_mean = []
    saved_std_dv = []

    # Statistics for run time
    for node_size in total_times:
        m = np.mean(node_size)
        std = np.std(node_size)
        time_mean.append(m)
        time_std_dv.append(std)
    time_std_dv = np.asarray(time_std_dv)
    time_mean = np.asarray(time_mean)

    # Statistics for saved vertices
    for node_size in total_saved:
        m = np.mean(node_size)
        std = np.std(node_size)
        saved_mean.append(m)
        saved_std_dv.append(std)
    saved_std_dv = np.asarray(saved_std_dv)
    saved_mean = np.asarray(saved_mean)

    print(saved_mean)
    print(saved_std_dv)

    np.save(path / "Statistics_DP", np.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
    y = np.arange(0, len(time_mean), 1, dtype=int)
    fig, ax = plt.subplots(1)
    ax.plot(y, saved_mean, label="Mean saved Vertices", color="blue")
    ax.fill_between(y, saved_mean + saved_std_dv / 2, saved_mean - saved_std_dv / 2, facecolor="blue", alpha=0.5)
    plt.savefig(path / 'DP_Saved.png')

    fig, ax = plt.subplots(1)
    ax.plot(y, time_mean, label="Mean Time Vertices", color="red")
    ax.fill_between(y, time_mean + time_std_dv / 2, time_mean - time_std_dv / 2, facecolor="red", alpha=0.5)
    plt.savefig(path / 'DP_Time.png')


def loadmfpt(path):
    """Load a Moving Firefighter Instance with Tree structure

    :param path: Experiment Instance path
    """
    # Load Tree
    T = nx.read_adjlist(path / "MFF_Tree.adjlist")

    # Relabeling Nodes
    mapping = {}
    for node in T.nodes:
        mapping[node] = int(node)
    T = nx.relabel_nodes(T, mapping)

    # Load Distance Matrix and Weights
    arrays = np.load(path /  "FDM_MFFP.npz")
    DistanceMatrix = arrays['arr_0']
    weights = arrays['arr_1']

    # Load Position Layout
    lay = open(path /  "layout_MFF.json")
    pos = {}
    pos_ = json.load(lay)  # layout of Tree
    for position in pos_:
        pos[int(position)] = pos_[position]

    # Load General Instance Parameters
    p = open(path / "instance_info.json")
    parameters = json.load(p)
    N = parameters["N"]                         # Instances per Tree size
    seed = parameters["seed"]                   # Tree seed
    scale = parameters["scale"]                 # Scale of embedded space
    starting_fire = parameters["start_fire"]    # Ignition Node or root
    tree_height = parameters["tree_height"]     # Tree Height from root

    T = nx.bfs_tree(T, starting_fire)           # Induced Tree with BFS
    T.add_node(N)                               # Agent

    # Degree of each node
    degrees = T.degree()
    max_degree = max(j for (i, j) in degrees)  # Tree max degree
    root_degree = T.degree[starting_fire]  # Tree Root degree

    return T, N, starting_fire, DistanceMatrix, seed, scale, N, max_degree, root_degree, tree_height


def ComputeTime(a_pos, node_pos, dist_matrix):
    Adj = dist_matrix
    delta_time = Adj[int(node_pos)][int(a_pos)]
    return delta_time


def SavedNodes(t, cutting_node, Valid_):
    # First we get the corresponding branch of T
    saved = 1
    father = t.search_nodes(name=str(cutting_node))[0]
    if Valid_:
        for node in father.iter_descendants("postorder"):
            if int(node.name) in Valid_:
                Valid_.pop(int(node.name))
            saved += 1
    # print("Saved:{}".format(saved))
    # Now, we detach this sub_tree from original
    father.detach()
    return saved


def Find_DP_Solution(Forest, Time, a_pos, Hash, dist_matrix):
    total_budget = Time
    elapsed = 0
    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos)
    Solution = []
    while Hash[key]['value'] != 0:
        node = Hash[key]['max_node']
        Time = Time - ComputeTime(a_pos, node, dist_matrix)
        elapsed += (total_budget - Time)
        father = Forest.search_nodes(name=node)[0]
        father.detach()
        a_pos = node
        Solution.append(node)
        total_budget = Time
        key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos)
    return Solution


def computeChildren(all_nodes, T):
    """
    Compute subtree length of each protected node
    :param all_nodes: List of all nodes in T
    :param T: Networkx Tree structure
    :return:
    """
    children_ = {}
    for node in all_nodes:
        saved = 1 + len(list(nx.descendants(T, int(node))))
        children_[node] = saved
    nx.set_node_attributes(T, children_, "saved")


def newicktree(T, sf):
    t = Tree()
    father = t.add_child(name=sf)
    buffer = []
    buffer.append(sf)

    while buffer:
        for child in T.successors(buffer.pop(0)):
            father.add_child(name=child)
            buffer.append(child)
        if buffer:
            father = t.search_nodes(name=buffer[0])[0]

    return t
