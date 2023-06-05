import os
import json
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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


def loadmfpt(path, directory):
    # Return [Tree,s_fire,Dist_Matrix,seed,scale,ax_pos,ay_pos]
    T = nx.read_adjlist(path / directory / "MFF_Tree.adjlist")  # Tree
    # Relabeling Nodes
    mapping = {}
    for node in T.nodes:
        mapping[node] = int(node)
    T = nx.relabel_nodes(T, mapping)
    arrays = np.load(path / directory / "FDM_MFFP.npz")
    T_Ad_Sym = arrays['arr_0']
    weights = arrays['arr_1']
    lay = open(path / directory / "layout_MFF.json")
    pos = {}
    pos_ = json.load(lay)                                       # layout of Tree
    for position in pos_:
        pos[int(position)] = pos_[position]

    # Get Instance Parameters
    p = open(path / directory / "instance_info.json")
    parameters = json.load(p)
    N = parameters["N"]                                         # Instances per Tree
    seed = parameters["seed"]                                   # Tree seed
    scale = parameters["scale"]                                 # Scale of space
    starting_fire = parameters["start_fire"]                    # Ignition Node
    tree_height = parameters["tree_height"]                     # Tree Height

    T = nx.bfs_tree(T, starting_fire)                           # Induced Tree with BFS
    T.add_node(N)                                               # Agent

    degrees = T.degree()
    max_degree = max(j for (i, j) in degrees)                   # Tree max degree
    root_degree = T.degree[starting_fire]                       # Tree Root degree

    return T, N, starting_fire, T_Ad_Sym, seed, scale, N, max_degree, root_degree, tree_height


def ComputeTime(a_pos, node_pos, dist_matrix):
    Adj = dist_matrix
    delta_time = Adj[int(node_pos)][int(a_pos)]
    return delta_time

def SavedNodes(t, cutting_node, Valid_):
    # First we get the corresponding branch of T
    saved = 1  # As we "defend nodes" detach node is also saved
    father = t.search_nodes(name=cutting_node)[0]

    for node in father.iter_descendants("postorder"):
        if node.name in Valid_:
            Valid_.pop(node.name)
        saved += 1
    # print("Saved:{}".format(saved))
    # Now, we detach this sub_tree from original
    father.detach()
    return saved


def Find_Solution(Forest, Time, a_pos, Hash, dist_matrix):
    # Construct the Key for Hash ( String: "Forest;time;pos_x,pos_y" )
    frame = 0
    total_budget = Time
    # spread_time = update
    # last_level_burnt = 0
    elapsed = 0

    key = Forest.write(format=8) + ';' + str(Time) + ';' + str(a_pos)
    # graphSolution(plotting,frame)
    Solution = []
    while Hash[key]['value'] != 0:
        frame += 1
        # Getting node of max value
        # print('Times to all nodes:')
        # print(config[2][plotting[4],:])
        node = Hash[key]['max_node']

        # Computes Remaining Time if agent travel to this node
        Time = Time - ComputeTime(a_pos, node, dist_matrix)
        elapsed += (total_budget - Time)

        # all_levels = plotting[6]
        # levels_to_burnt = int((total_budget - Time)/spread_time) + last_level_burnt
        # print('Levels to Burnt')
        # print(levels_to_burnt)
        # for level in range(last_level_burnt+1,(last_level_burnt + levels_to_burnt)+1):
        #     print('Level')
        #     print(level)
        #     keys = [k for k, v in all_levels.items() if v <= level]
        #     print('Keys')
        #     print(keys)
        #     # Assign to burning nodes and quit from remaining
        #     for element in keys:
        #         if element not in plotting[2]:
        #             plotting[2].append(element)
        #         if element in plotting[3]:
        #             plotting[3].remove(element)
        #     graphSolution(plotting, frame, 0, level, spread_time*level)
        #     frame += 1

        # Change pos of agent to next node in solution
        # pos = plotting[1]

        # pos[plotting[4]] = pos[int(node)]
        # plotting[1] = pos
        # Saved Trees by selecting this node
        saved = DetachNode(Forest, node)

        # last_level_burnt += levels_to_burnt
        # Add saved Node
        # plotting[7].append(int(node))

        # New agent position moves to node position
        a_pos = node
        Solution.append(node)
        # New Key
        # graphSolution(plotting, frame,total_budget-Time,levels_to_burnt,elapsed)
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

