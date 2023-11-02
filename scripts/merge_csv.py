import csv
import argparse
from pathlib import Path

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
