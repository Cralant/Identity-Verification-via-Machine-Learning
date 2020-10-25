from pynput import mouse, keyboard
from datetime import datetime
import functions as fn
import numpy as np
import csv
import pandas as pd

log = []
csvread = csv.reader(open("Data/test/user1 test.csv"))
for row in csvread:
	log.append([row[0],datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f'),row[2],row[3],row[4],row[5]])

readable = fn.rawToReadable(pd.DataFrame(log))
readable[0]['user'] = 0
readable[0] = readable[0][['user', 'time', 'button', 'action', 'x', 'y']]
readable[0]['x'] = readable[0]['x'].astype(int)
readable[0]['y'] = readable[0]['y'].astype(int)
readable[1]['user'] = 0
readable[1] = readable[1][['user', 'key', 'press', 'release']]
proccessedMouse = fn.processMouseFile(readable[0])
processedKey = fn.processKeyboardFile(readable[1])
users = list(csv.reader(open('Data/users.csv')))[0]
results = [[],[]]
bothResult = []
finalResults = []
for i,u in enumerate(users):
	mModelName = 'Data/Models/trainingMouseModel-' + u + '.pkl'
	kModelName = 'Data/Models/trainingKeyModel-' + u + '.pkl'
	results[0].append([i,np.mean(fn.testMouseAgainst(proccessedMouse,mModelName))])
	results[1].append([i,np.mean(fn.testKeyAgainst(processedKey,kModelName))])
	bothResult.append([i, [results[0][-1][1], results[1][-1][1]]])
sortedResults = [sorted(results[0], key=lambda t: t[:][1], reverse=True), sorted(results[1], key=lambda t: t[:][1], reverse=True),]
mscale = np.sum([i[1] for i in results[0]])
kscale = np.sum([i[1] for i in results[1]])
scaledBothResults = [[i[1][0]/mscale for i in bothResult], [i[1][1]/kscale for i in bothResult]]
meanScaledResults = (np.mean( np.array([ scaledBothResults[0],scaledBothResults[1] ]), axis=0 ))
for i in range(0, len(meanScaledResults)):
	finalResults.append([i,meanScaledResults[i]])
finalResults = sorted(finalResults, key=lambda t: t[:][1], reverse=True)
print('Combined and scaled results:')
for result in finalResults:
	print(f'	{users[result[0]]} - {round(result[1]*100,2)}% - [{round(results[0][result[0]][1]*100,2)}%, {round(results[1][result[0]][1]*100,2)}%]')
	print(f'	{np.mean([results[0][result[0]][1]*100, results[1][result[0]][1]*100])}')
#print(sortedResults)
#print(f'Best (scaled) match found for mouse is {users[sortedResults[0][0][0]]}-{round((sortedResults[0][0][1]/mscale)*100,2)}%, and keyboard is {users[sortedResults[1][0][0]]}-{round((sortedResults[1][0][1]/kscale)*100,2)}%')