import csv
import functions as fn
import pandas as pd

files = list(csv.reader(open('Data/users.csv')))[0]

for file in files:
	print(file)
	fileLoc = 'Data/Raw/' + file + '.csv'
	csvFile = pd.read_csv(fileLoc, sep=',', names = ['device','time','action','button','x','y'], parse_dates=['time'])
	readable = fn.rawToReadable(csvFile)
	readable[0].to_csv(('Data/Training/M'+file+'.csv'),index=False)
	readable[1].to_csv(('Data/Training/K'+file+'.csv'),index=False)