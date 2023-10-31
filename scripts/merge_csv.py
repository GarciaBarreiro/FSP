import csv
import argparse
from pathlib import Path

# HOW TO:
# 1) open input[n]                                                                                      DONE
# 2) check cpus first line => number of nodes                                                           DONE
# 3) when cpus < cpus[-1] => number of nodes = cpus                                                     DONE
# 4) output all that to temp.csv                                                                        TODO
# 5) find a way to read temp.csv and order it by number of nodes, then cpus, thus creating output       TODO

# reads csv, generates another with better formating and number of nodes
def csv_rdr_wrtr(input, output):
    with open(input, newline='') as incsv, open(output, 'w', newline='') as outcsv:
        rdr = csv.reader(incsv, delimiter=';')
        wrtr = csv.writer(outcsv, delimiter=',')
        prev_cpus = 9999
        nodes = 0
        for row in rdr:
            if int(row[0]) < prev_cpus:
                nodes = row[0]
            prev_cpus = int(row[0])
            wrtr.writerow([nodes, row[0], row[1].replace(',','.'),
                           row[2].replace(',','.'), row[3].replace(',','.').strip()])

# merges various csvs into one
def merge_csv(inputs, output):
    # 1) open all files
    # 2) read first line of each file => check smaller
    # 3) write until cpu changes => check other file
    # 4) repeat 2) & 3) until end

    # OR

    # 1) open one file, dump all data to an array in memory
    # 2) do 1) with the rest of files
    # 3) order (check how to order by node and cpu)

    data = []
    for inp in inputs:
        with open(inp, newline='') as incsv:
            rdr = csv.reader(incsv, delimiter=',')
            for row in rdr:
                data.append({
                    'nodes': int(row[0]),
                    'cpus': int(row[1]),
                    'time': row[2],
                    'err': row[3],
                    'qual': row[4]
                    })
        Path.unlink(inp)
    ordered = sorted(data, key=lambda d: (d['nodes'], d['cpus']))
    with open(output, 'w', newline='') as outcsv:
        wrtr = csv.writer(outcsv, delimiter=',')
        for item in ordered:
            wrtr.writerow(item.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', help='input files, separated by commas')
    parser.add_argument('-o', '--output', help='output file, defaults to ./output.csv')
    args = parser.parse_args()
    print(args.input_files, args.output)
    input = args.input_files.split(',')
    if args.output:
        output = args.output
    else:
        output = './output.csv'
    temp_out = []
    print(input, output)
    i = 0
    for inp in input:
        temp_out.append('./temp_'+str(i)+'.csv')
        csv_rdr_wrtr(inp, temp_out[i])
        i+=1
    merge_csv(temp_out, output)

if __name__ == "__main__":
    main()
