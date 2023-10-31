from matplotlib import pyplot as plt
import numpy as np
import csv

# read csv
# avg per node number
# plot different things

def read_csv_old(filename): # returns list of dictionaries, not good for working with graphs
    data = []
    with open(filename, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=';')
        while True:
            temp_list = {'cpus': [], 'time': [], 'err': [], 'qual': []}
            for i in range(5):
                try:
                    row = rdr.__next__() # check this
                except StopIteration as si:
                    return data
                
                temp_list['cpus'].append(row[0])
                temp_list['time'].append(float(str(row[1]).replace(',','.')))
                temp_list['err'].append(float(str(row[2]).replace(',','.')))
                temp_list['qual'].append(float(str(row[3]).replace(',','.')))
            
            temp_dict = {
                    'nodes': 1,
                    'cpus': int(temp_list['cpus'][0]),
                    'time': np.mean(temp_list['time']),
                    'err': np.mean(temp_list['err']),
                    'qual': np.mean(temp_list['qual'])
                    }

            if data:
                if data[-1]['cpus'] < temp_dict['cpus']:
                    temp_dict['nodes'] = data[-1]['nodes']
                else:
                    temp_dict['nodes'] = data[-1]['nodes'] * 2
            data.append(temp_dict)

def read_csv(filename):   # returns dictionary of lists, good for working with graphs
    data = {'nodes': [], 'cpus': [], 'time': [], 'err': [], 'qual': []}
    with open(filename, newline='') as csvfile:
        rdr = csv.reader(csvfile, delimiter=';')
        while True:
            temp_list = {'cpus': [], 'time': [], 'err': [], 'qual': []}
            for i in range(5):
                try:
                    row = rdr.__next__() # check this
                except StopIteration as si:
                    return data
                
                temp_list['cpus'].append(row[0])
                temp_list['time'].append(float(str(row[1]).replace(',','.')))
                temp_list['err'].append(float(str(row[2]).replace(',','.')))
                temp_list['qual'].append(float(str(row[3]).replace(',','.')))
            
            data['cpus'].append(int(temp_list['cpus'][0]))
            data['time'].append(np.mean(temp_list['time']))
            data['err'].append(np.mean(temp_list['err']))
            data['qual'].append(np.mean(temp_list['qual']))
            if not data['nodes']:
                data['nodes'].append(1)
            else:
                if data['cpus'][-2] < data['cpus'][-1]:
                    data['nodes'].append(data['nodes'][-1])
                else:
                    data['nodes'].append(data['nodes'][-1] * 2)

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

        cpu *= 2
    return no_nodes

def plot_graph (data, x, y, filename, y2=None):
    fig,ax = plt.subplots()

    plt.xlabel(x)
    ax.set_ylabel(y)

    ax.plot(data[x], data[y], marker='o')

    if y2:
        ax2 = ax.twinx()
        ax2.set_ylabel(y2)
        ax2.plot(data[x], data[y2], marker='s', color='orange')

    plt.legend()    # prints error
    plt.savefig(filename)

data = read_csv('../1P/results.csv')

# print(data)
no_nodes = remove_nodes(data)
# print(no_nodes)

plot_graph(no_nodes, 'cpus', 'time', 'proba.png', y2='err')
