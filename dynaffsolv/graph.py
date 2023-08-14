import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        List of Available Experiments:
            - DP vs Hs vs rd
            - DP vs Time vs rd
            ''',
        epilog='''python graph.py -e experiment'''
    )
    parser.add_argument(
        '--experiment', '-e', type=int,
        help="Type of Experiment.")
    return parser.parse_known_args(args)[0]

args = argParser(sys.argv[:])
if args.experiment is None:
    raise argparse.ArgumentTypeError('No solver selected')


if args.experiment ==1 :
    bt_results = np.load('Experiments/Experiment_32_2-5_d5/backtrackSolver.npy')
    ms_results = np.load('Experiments/Experiment_32_2-5_d5/ms_heuristic.npy')
    hd_results = np.load('Experiments/Experiment_32_2-5_d5/hd_heuristic.npy')

    fig, ax = plt.subplots(1)
    y= np.arange(0, len(bt_results[0]), 1, dtype=int)
    fig.suptitle('Random Graphs')
    plt.xlabel('Tree Size')
    plt.ylabel('# Saved vertices')

    ax.plot(y, bt_results[0], label="Dynamic Programming", color="brown",marker="s")
    ax.plot(y, ms_results[0], label="MS Heuristic", color="green",marker="8")
    ax.plot(y, hd_results[0], label="HD Heuristic", color="blue",marker="D")

    labels = ['10',  '20', '30', '40']
    plt.xticks(np.arange(len(bt_results[0])), labels, rotation='vertical')
    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="lower left", ncol=1, bbox_to_anchor=(1, 0.5), labelspacing=1)

    #ax.fill_between(y, bt_results[0] + bt_results[1], bt_results[0] - bt_results[1], facecolor="brown", alpha=0.5)
    #ax.fill_between(y, ms_results[0] + ms_results[1], ms_results[0] - ms_results[1], facecolor="green", alpha=0.5)
    #ax.fill_between(y, hd_results[0] + hd_results[1], hd_results[0] - hd_results[1], facecolor="blue", alpha=0.5)

    plt.savefig("Experiments/Experiment_32_2-5_d5/BT_saved.png")


    # Plotting Degree vs Saved
    bt_results_d2 = np.load('Experiments/Experiment_25_2-5_d2/backtrackSolver.npy')
    ms_results_d2 = np.load('Experiments/Experiment_25_2-5_d2/ms_heuristic.npy')
    hd_results_d2 = np.load('Experiments/Experiment_25_2-5_d2/hd_heuristic.npy')

    bt_results_d3 = np.load('Experiments/Experiment_30_2-5_d3/backtrackSolver.npy')
    ms_results_d3 = np.load('Experiments/Experiment_30_2-5_d3/ms_heuristic.npy')
    hd_results_d3 = np.load('Experiments/Experiment_30_2-5_d3/hd_heuristic.npy')

    bt_results_d4 = np.load('Experiments/Experiment_31_2-5_d4/backtrackSolver.npy')
    ms_results_d4 = np.load('Experiments/Experiment_31_2-5_d4/ms_heuristic.npy')
    hd_results_d4 = np.load('Experiments/Experiment_31_2-5_d4/hd_heuristic.npy')

    bt_results_d5 = np.load('Experiments/Experiment_32_2-5_d5/backtrackSolver.npy')
    ms_results_d5 = np.load('Experiments/Experiment_32_2-5_d5/ms_heuristic.npy')
    hd_results_d5 = np.load('Experiments/Experiment_32_2-5_d5/hd_heuristic.npy')

    st_bt = [bt_results_d2[0][3], bt_results_d3[0][3], bt_results_d4[0][3], bt_results_d5[0][3]]
    st_ms = [ms_results_d2[0][3], ms_results_d3[0][3], ms_results_d4[0][3], ms_results_d5[0][3]]
    st_hd = [hd_results_d2[0][3], hd_results_d3[0][3], hd_results_d4[0][3], hd_results_d5[0][3]]


    fig2, ax = plt.subplots(1)
    y = np.arange(0, len(st_bt), 1, dtype=int)
    fig2.suptitle('Random Graphs with 40-vertices')
    plt.xlabel('Root Degree')
    plt.ylabel('Saved vertices')

    ax.plot(y, st_bt, label="BackTracking", color="brown",marker="s")
    ax.plot(y, st_ms, label="MS Heuristic", color="green",marker="8")
    ax.plot(y, st_hd, label="HD Heuristic", color="blue",marker="D")

    labels = ['2',  '3', '4', '5']
    plt.xticks(np.arange(len(st_bt)), labels, rotation='vertical')
    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="lower left", ncol=1, bbox_to_anchor=(1, 0.5), labelspacing=1)
    plt.savefig("Experiments/BT_saved_rdegree.png")

if args.experiment == 2:
    dp_results_rd2 = np.load('Experiments/Experiment_25_2-5_d2/backtrackSolver.npy')
    dp_results_rd3 = np.load('Experiments/Experiment_30_2-5_d3/backtrackSolver.npy')
    dp_results_rd4 = np.load('Experiments/Experiment_31_2-5_d4/backtrackSolver.npy')
    dp_results_rd5 = np.load('Experiments/Experiment_32_2-5_d5/backtrackSolver.npy')

    fig, ax = plt.subplots(1)
    y = np.arange(0, len(dp_results_rd4[0]), 1, dtype=int)
    fig.suptitle('Dynamic Programming Average Time')
    plt.xlabel('Tree Size')
    plt.ylabel('# Average Time (sec)')

    ax.plot(y, dp_results_rd2[2], label="rd = 2", color="brown", marker="s")
    ax.plot(y, dp_results_rd3[2], label="rd = 3", color="green", marker="8")
    ax.plot(y, dp_results_rd4[2], label="rd = 4", color="blue", marker="D")
    #ax.plot(y, dp_results_rd5[2], label="rd = 5", color="red", marker="x")

    labels = ['10', '20', '30', '40']
    plt.xticks(np.arange(len(dp_results_rd2[0])), labels, rotation='vertical')
    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="lower left", ncol=1, bbox_to_anchor=(1, 0.5), labelspacing=1)
    #plt.yscale("log")
    plt.savefig("Experiments/DP_Time_vsrd.png")