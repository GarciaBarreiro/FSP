from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse

def plot_graph (data, filename, log_base=10, y_log=False, t_host=True, t_device=True):
    fig,ax = plt.subplots()

    # plt.xlabel('Número de nodos')
    # ax.set_title('Número de threads por bloque')
    ax.set_ylabel('Tempo (ms)')
    ax.set_xscale('log', base=log_base)
    if y_log:
        ax.set_yscale('log')

    if t_host:
        ax.plot(data['x'], data['t_host'], label='Suma no host', marker='o')
    if t_device:
        ax.plot(data['x'], data['t_device'], label='Suma no device', marker='o')

    plt.xticks(data['x'])
    plt.legend()
    plt.savefig(filename)

def main():
    threads = {'x': [32,64,128,256,512,1024],
               't_host': [8206.040677,8192.542434,8180.001850,
                          8182.845302,8186.443082,8246.838962],
               't_device': [4897.503498,4800.365083,4781.550056,
                            4834.887684,4835.814052,3723.304673],
              }

    num_reps = {'x': [1,10,100,1000],
                't_host': [8207.947832,74208.212523,734703.949378,7340602.372579],
                't_device': [3797.599765,4196.888845,5632.132094,23058.602886],
               }

    mat_threads = {'x': [32,16,8,4,2,1],
                   't_host': [499212.680472,498039.980314,4366676.319740,
                              493697.268409,491184.255205,491294.250847],
                   't_device': [643.8480,547.4482,434.454810,
                                556.327524,1294.805963,738.772121],
                  }
 
    plot_graph(threads, 'threads.png', log_base=2)
    plot_graph(num_reps, 'n_reps.png', y_log=True)
    plot_graph(mat_threads, 'mat_threads.png', log_base=2, t_host=False)
    # plot_graph(no_nodes, 'cpus', 'qual', 'qual.png')
    # plot_graph(no_nodes, 'cpus', 'err', 'err.png')

if __name__ == "__main__":
    main()
