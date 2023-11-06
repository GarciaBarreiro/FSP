from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse

# read csv
# avg per node number
# plot different things

def read_csv(filename):   # returns dictionary of lists, good for working with graphs
    data = {'nodes': [], 'cpus': [], 'time': [], 'err': [], 'qual': []}
    with open(filename, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        try:
            row = rdr.__next__()
        except StopIteration as si:
            return None

        prev_cpu = int(row[1])
        temp_list = {'nodes': [], 'cpus': [], 'time': [], 'err': [], 'qual': []}
        continue_while = True
        while continue_while:
            if prev_cpu == int(row[1]):
                temp_list['nodes'].append(int(row[0]))
                temp_list['cpus'].append(int(row[1]))
                temp_list['time'].append(float(row[2]))
                temp_list['err'].append(float(row[3]))
                temp_list['qual'].append(float(row[4]))
            else:
                data['nodes'].append(temp_list['nodes'][0])
                data['cpus'].append(temp_list['cpus'][0])
                data['time'].append(np.mean(temp_list['time']))
                data['err'].append(np.mean(temp_list['err']))
                data['qual'].append(np.mean(temp_list['qual']))

                temp_list = {'nodes': [], 'cpus': [], 'time': [], 'err': [], 'qual': []}

            prev_cpu = int(row[1])

            try:
                row = rdr.__next__()
            except StopIteration as si:
                continue_while = False

        # last iteration
        data['nodes'].append(temp_list['nodes'][0])
        data['cpus'].append(temp_list['cpus'][0])
        data['time'].append(np.mean(temp_list['time']))
        data['err'].append(np.mean(temp_list['err']))
        data['qual'].append(np.mean(temp_list['qual']))

        return data

def remove_nodes(data): # merge where # of cpus is the same
    no_nodes = {'cpus': [], 'time': [], 'err': [], 'qual': []}
    cpu = min(data['cpus'])
    while cpu <= max(data['cpus']):
        temp_dict = {'time': [], 'err': [], 'qual': []}
        for i in range(len(data['cpus'])):
            if data['cpus'][i] == cpu:
                temp_dict['time'].append(data['time'][i])
                temp_dict['err'].append(data['err'][i])
                temp_dict['qual'].append(data['qual'][i])
        no_nodes['cpus'].append(cpu)
        no_nodes['time'].append(np.mean(temp_dict['time']))
        no_nodes['err'].append(np.mean(temp_dict['err']))
        no_nodes['qual'].append(np.mean(temp_dict['qual']))

        cpu *= 2        # change
    return no_nodes

def plot_graph (data, x, y, filename, y2=None):
    fig,ax = plt.subplots()

    plt.xlabel(x)
    ax.set_ylabel(y)

    ax.plot(data[x], data[y], marker='o', label='number of '+x+' over '+y)

    if y2:
        ax2 = ax.twinx()
        ax2.set_ylabel(y2)
        ax2.plot(data[x], data[y2], marker='s', color='orange', label='number of '+x+' over '+y2)

    plt.legend()    # label of ax2 doesn't appear
    plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    args = parser.parse_args()
    
    data = read_csv(args.input)

    # print(data)
    no_nodes = remove_nodes(data)
    # print(no_nodes)

    plot_graph(no_nodes, 'cpus', 'time', 'time.png')
    plot_graph(no_nodes, 'cpus', 'qual', 'qual.png')
    plot_graph(no_nodes, 'cpus', 'err', 'err.png')

if __name__ == "__main__":
    main()
