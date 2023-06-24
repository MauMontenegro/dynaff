import argparse
import sys
from pathlib import Path
import time as tm
import warnings
from solvers.utilities.utils import *


def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        List of Available Inputs:
            -Cellular Automata Environment (caenv)
            -Random Tree (rndtree)
            ''',
        epilog='''python dynaffsolv.py -s dpsolver -i rndtree -e experiment_name'''
    )

    parser.add_argument(
        '--input', '-i', type=str,
        help="Type of Input for Solver.")
    parser.add_argument(
        '--solver', '-s', type=str,
        help="Type of Solver.")
    parser.add_argument(
        '--experiment', '-e', type=str,
        help="Config File.")

    return parser.parse_known_args(args)[0]


def saveSolution(instance_path, instance, solution, saved, time, hash_calls, hash_size, solver_name):
    summary_name = solver_name + "_summary"
    output_path = instance_path / instance / summary_name
    print(output_path)
    with open(output_path, "w") as writer:
        writer.write("Solution: {}\n".format(solution))
        writer.write("Saved: {}\n".format(saved))
        writer.write("RunTime: {}\n".format(time))
        writer.write("Hash_Calls: {}\n".format(hash_calls))
        writer.write("Hash Size: {}\n".format(hash_size))


def Statistics(path, total_saved, total_times,solver):
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for node_size in total_times:
            m = np.mean(node_size)
            std = np.std(node_size)
            time_mean.append(m)
            time_std_dv.append(std)
        time_std_dv = np.asarray(time_std_dv)
        time_mean = np.asarray(time_mean)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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
    np.save(path / solver, np.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
    y = np.arange(0, len(time_mean), 1, dtype=int)
    fig, ax = plt.subplots(1)
    ax.plot(y, saved_mean, label="Mean saved Vertices", color="blue")
    ax.fill_between(y, saved_mean + saved_std_dv / 2, saved_mean - saved_std_dv / 2, facecolor="blue", alpha=0.5)
    string = solver + '.png'
    plt.savefig(path / string )

    fig, ax = plt.subplots(1)
    ax.plot(y, time_mean, label="Mean Time Vertices", color="red")
    ax.fill_between(y, time_mean + time_std_dv / 2, time_mean - time_std_dv / 2, facecolor="red", alpha=0.5)
    plt.savefig(path / 'DP_Time.png')

def setupSolver(solver):
    import solvers.solvers as solvers
    target_class = solver
    if hasattr(solvers, target_class):
        solverClass = getattr(solvers, target_class)
    else:
        raise AssertionError('There is no Solver called {}'.format(target_class))
    return solverClass


if __name__ == '__main__':

    total_times = []
    total_saved = []
    size_dirs = []

    args = argParser(sys.argv[:])
    if args.solver is None:
        raise argparse.ArgumentTypeError('No solver selected')
    if args.experiment is None:
        raise argparse.ArgumentTypeError('No Experiment folder selected')
    if args.input is None:
        raise argparse.ArgumentTypeError('No input selected')

    experiment_path = Path.cwd() / "Experiments" / str(args.experiment)
    solver = setupSolver(args.solver)

    for d in next(os.walk(experiment_path)):
        size_dirs.append(d)
    size_dirs = sorted(size_dirs[1])

    # Traverse for each Tree Size experiments
    for dir in size_dirs:
        instance_path = experiment_path / str(dir)
        inst_dirs = []
        for i in next(os.walk(instance_path)):
            inst_dirs.append(i)
        inst_dirs = sorted(inst_dirs[1])
        saved_p_nodes = []
        t_p_nodes = []

        # Traverse for each Instance
        for inst in inst_dirs:
            print("\n\n>>>>>>Compute solution for Tree {i} of {n} <<<<<<<<".format(n=dir, i=inst))

            # Load Instance
            T, N, starting_fire, T_Ad_Sym, seed, scale, agent_pos, max_degree, root_degree, time = \
                loadmfpt(instance_path / str(inst))

            # Get all available nodes
            all_nodes = dict.fromkeys(T.nodes)
            all_nodes.pop(N)

            # Get node Levels
            levels_ = nx.single_source_shortest_path_length(T, starting_fire)
            nx.set_node_attributes(T, levels_, "levels")

            # Get node out degree
            degrees = list(T.out_degree)
            degrees_ = {}
            for element in degrees:
                degrees_[element[0]] = element[1]
            nx.set_node_attributes(T, degrees_, "degrees")

            # Marked Property for all nodes in T
            marked_list = [0] * T.number_of_nodes()
            nx.set_node_attributes(T, marked_list, "marked")

            # Create Newick Tree for Dynamic Programming Solver
            if args.solver == "dpsolver":
                ntree = newicktree(T, starting_fire)
            else:
                ntree = 0

            # CALL THE SOLVER
            # ----------------------------------------------------------------------------------------------------------
            tracing_start()
            start = tm.time()

            max_saved_trees, Sol = solver(agent_pos, all_nodes, time, time, 0, T_Ad_Sym, 0, T, ntree)

            end = tm.time()
            t = (end - start)
            print("time elapsed {} seconds".format(t))
            peak = tracing_mem()
            # -----------------------------------------------------------------------------------------------------------

            # Console Printing Results
            print("\nForest with:{n} nodes".format(n=len(all_nodes)))
            print("Max saved Trees:{t}".format(t=max_saved_trees))

            Hash_Calls = 0
            # Retrieve Solution Strategy
            if args.solver == "dpsolver":
                solution = Find_DP_Solution(ntree, time, agent_pos, Sol[1], T_Ad_Sym)
                print("\nHash calls:{n}".format(n=Sol[0]))
                print("\nHash size:{n}".format(n=len(Sol[1])))
                print("\nSolution Sequence:{n}".format(n=solution))
                Hash_Calls = Sol[0]
            else:
                print("\nSolution Sequence: {s}".format(s=Sol))

            # Saving stats for general parameters
            saveSolution(instance_path, inst, Sol, max_saved_trees, t, Hash_Calls, len(Sol), args.solver)
            
            # Saved nodes per Graph
            saved_p_nodes.append(max_saved_trees)
            t_p_nodes.append(t)
        total_times.append(t_p_nodes)
        total_saved.append(saved_p_nodes)
    Statistics(experiment_path, total_saved, total_times, args.solver)
