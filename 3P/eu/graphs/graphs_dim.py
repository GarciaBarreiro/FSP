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
    mat_m = min(data['mat_m'])
    while mat_m <= max(data['mat_m']):
        for dir in range(2):
            temp_dict  = {'nodes': [], 'time': []}
            for i in range(len(data['mat_m'])):
                if data['mat_m'][i] == mat_m and data['dir'][i] == dir:
                    temp_dict['nodes'].append(data['nodes'][i])
                    temp_dict['time'].append(data['time'][i])
            ret['nodes'].append(temp_dict['nodes'][0])
            ret['dir'].append(dir)
            ret['mat_m'].append(mat_m)
            ret['mat_n'].append(mat_m)
            ret['vec_l'].append(mat_m)
            ret['time'].append(np.mean(temp_dict['time']))

        mat_m += 100
    return ret

def split_dirs(data):
    dir0 = {'dim': [], 'time': []}
    dir1 = {'dim': [], 'time': []}

    for i in range(len(data['mat_m'])):
        if data['dir'][i] == 0:
            dir0['dim'].append(data['mat_m'][i])
            dir0['time'].append(data['time'][i])
        else:
            dir1['dim'].append(data['mat_m'][i])
            dir1['time'].append(data['time'][i])
    return dir0,dir1

def plot_graph (data, filename):
    fig,ax = plt.subplots()

    # plt.xlabel('Número de nodos')
    ax.set_title('Tamaño dos elementos da multiplicación')
    ax.set_ylabel('Tempo (s)')

    d0,d1 = split_dirs(data)

    ax.plot(d0['dim'], d0['time'], label='matriz x vector')
    ax.plot(d1['dim'], d1['time'], label='vector x matriz')

    # plt.xticks(d0['dim'])
    plt.xticks(range(min(d0['dim']), max(d0['dim'])+1, 1500), rotation=35)
    plt.legend()    # label of ax2 doesn't appear
    plt.savefig(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    args = parser.parse_args()
    
    data = read_csv(args.input)

    print("data:")
    print(data)
    merged = merge_everything(data)
    print("data:")
    print(merged)

    plot_graph(merged, 'sizes.png')
    # plot_graph(no_nodes, 'cpus', 'qual', 'qual.png')
    # plot_graph(no_nodes, 'cpus', 'err', 'err.png')

if __name__ == "__main__":
    main()
