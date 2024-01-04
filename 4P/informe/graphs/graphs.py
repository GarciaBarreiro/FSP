from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse

def plot_graph (data, filename, x_log=True, log_base=10, y_log=False, t_host=True, t_device=True, rotation=0):
    fig,ax = plt.subplots()

    # plt.xlabel('Número de nodos')
    # ax.set_title('Número de threads por bloque')
    ax.set_ylabel('Tempo (ms)')
    if x_log:
        ax.set_xscale('log', base=log_base)
    if y_log:
        ax.set_yscale('log')

    if t_host:
        ax.plot(data['x'], data['t_host'], label='Suma no host', marker='o')
    if t_device:
        ax.plot(data['x'], data['t_device'], label='Suma no device', marker='o')

    plt.xticks(data['x'], rotation=rotation)
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
    cuBLAS_t = {'x': [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,
                      13000,14000,15000,16000,17000,18000,19000,20000,],
                't_device': [654.804711,663.361515,655.316097,693.574489,737.583535,
                             770.401459,810.624650,859.800075,946.202668,1042.572156,
                             1093.555950,1180.640421,1255.138030,1409.672628,1478.098637,
                             1657.943146,1785.546241,1880.163827,2046.650499,2206.970698,],

               }
 
    plot_graph(threads, 'threads.png', log_base=2)
    plot_graph(num_reps, 'n_reps.png', y_log=True)
    plot_graph(mat_threads, 'mat_threads.png', log_base=2, t_host=False)
    plot_graph(cuBLAS_t, 'cublas.png', x_log=False, t_host=False, rotation=35)
    # plot_graph(no_nodes, 'cpus', 'qual', 'qual.png')
    # plot_graph(no_nodes, 'cpus', 'err', 'err.png')

if __name__ == "__main__":
    main()
