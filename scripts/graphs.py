from matplotlib import pyplot as plt
import numpy as np      # not needed i think
import csv

# read csv
# avg per node number
# plot different things

def plot_graph (graphname, filename):
    fig = plt.subplots()

    plt.xlabel("")
    plt.ylabel("")

    plt.legent()
    plt.savefig(file)

def read_csv(filename):
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

data = read_csv('../1P/results.csv')

for dic in data:
    print(dic)
