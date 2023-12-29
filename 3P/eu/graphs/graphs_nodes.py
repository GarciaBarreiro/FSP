from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse

# read csv
# avg per node number
# plot different things

def read_csv(filename):   # returns dictionary of lists, good for working with graphs
    data = {'nodes': [], 'dir': [], 'mat_m': [], 'mat_n': [], 'vec_l': [], 'time': []}
    with open(filename, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        try:
            row = rdr.__next__()
        except StopIteration as si:
            return None

        prev_dir = int(row[1])
        temp_list = {'nodes': [], 'dir': [], 'mat_m': [], 'mat_n': [], 'vec_l': [], 'time': []}
        continue_while = True
        while continue_while:
            if prev_dir == int(row[1]):
                temp_list['nodes'].append(int(row[0]))
                temp_list['dir'].append(int(row[1]))
                temp_list['mat_m'].append(int(row[2]))
                temp_list['mat_n'].append(int(row[3]))
                temp_list['vec_l'].append(int(row[4]))
                temp_list['time'].append(float(row[5]))
            else:
                data['nodes'].append(temp_list['nodes'][0])
                data['dir'].append(temp_list['dir'][0])
                data['mat_m'].append(temp_list['mat_m'][0])
                data['mat_n'].append(temp_list['mat_n'][0])
                data['vec_l'].append(temp_list['vec_l'][0])
                data['time'].append(np.mean(temp_list['time']))

                temp_list = {'nodes': [], 'dir': [], 'mat_m': [], 'mat_n': [], 'vec_l': [], 'time': []}

            prev_dir = int(row[1])

            try:
                row = rdr.__next__()
            except StopIteration as si:
                continue_while = False

        # last iteration
        data['nodes'].append(temp_list['nodes'][0])
        data['dir'].append(temp_list['dir'][0])
        data['mat_m'].append(temp_list['mat_m'][0])
        data['mat_n'].append(temp_list['mat_n'][0])
        data['vec_l'].append(temp_list['vec_l'][0])
        data['time'].append(np.mean(temp_list['time']))

        return data

def merge_everything(data):
    ret = {'nodes': [], 'dir': [], 'mat_m': [], 'mat_n': [], 'vec_l': [], 'time': []}
    nodes = min(data['nodes'])
    while nodes <= max(data['nodes']):
        for dir in range(2):
            temp_dict  = {'mat_m': [], 'mat_n': [], 'vec_l': [], 'time': []}
            for i in range(len(data['nodes'])):
                if data['nodes'][i] == nodes and data['dir'][i] == dir:
                    temp_dict['mat_m'].append(data['mat_m'][i])
                    temp_dict['mat_n'].append(data['mat_n'][i])
                    temp_dict['vec_l'].append(data['vec_l'][i])
                    temp_dict['time'].append(data['time'][i])
            ret['nodes'].append(nodes)
            ret['dir'].append(dir)
            ret['mat_m'].append(temp_dict['mat_m'][0])
            ret['mat_n'].append(temp_dict['mat_n'][0])
            ret['vec_l'].append(temp_dict['vec_l'][0])
            ret['time'].append(np.mean(temp_dict['time']))

        nodes += 1
    return ret

def split_dirs(data):
    dir0 = {'nodes': [], 'time': []}
    dir1 = {'nodes': [], 'time': []}

    for i in range(len(data['nodes'])):
        if data['dir'][i] == 0:
            dir0['nodes'].append(data['nodes'][i])
            dir0['time'].append(data['time'][i])
        else:
            dir1['nodes'].append(data['nodes'][i])
            dir1['time'].append(data['time'][i])
    return dir0,dir1

def plot_graph (data, filename):
    fig,ax = plt.subplots()

    plt.xlabel('Número de nodos')
    ax.set_ylabel('Tempo (s)')

    d0,d1 = split_dirs(data)

    ax.plot(d0['nodes'], d0['time'], marker='o', label='matriz x vector')
    ax.plot(d1['nodes'], d1['time'], marker='o', label='vector x matriz')

    plt.xticks(d0['nodes'])
    plt.legend()    # label of ax2 doesn't appear
    plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    args = parser.parse_args()
    
    data = read_csv(args.input)

    print(data)
    merged = merge_everything(data)
    print(merged)

    plot_graph(merged, 'time.png')
    # plot_graph(no_nodes, 'cpus', 'qual', 'qual.png')
    # plot_graph(no_nodes, 'cpus', 'err', 'err.png')

if __name__ == "__main__":
    main()
