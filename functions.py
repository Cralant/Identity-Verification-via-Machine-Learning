import math
import csv
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.options.mode.chained_assignment = None

fingerPositions = {	'F1':['key.esc','key.1','key.q','key.a','key.z','key.ctrl_l'],
					'F2':['key.2','key.w','key.s','key.x'],
					'F3':['key.3','key.e','key.d','key.c',],
					'F4':['key.4','key.5','key.r','key.t','key.f','key.g','key.v','key.b'],
					'F5':['key.space','key.alt_l','key.alt_r'],
					'F6':['key.6','key.7','key.y','key.u','key.h','key.j','key.n','key.m'],
					'F7':['key.8','key.i','key.k','key.comma'],
					'F8':['key.9','key.o','key.l','key.period'],
					'F9':['key.0','key.p','key.;','key./']}
letters = ['key.a','key.b','key.c','key.d','key.e','key.f','key.g','key.h','key.i','key.j','key.k','key.l','key.m','key.n','key.o','key.p','key.q','key.r','key.s','key.t','key.u','key.v','key.w','key.x','key.y','key.z',]

def readMouseLog(fileName, user):
	log = pd.read_csv(fileName, sep=',')#, dtype={'time':np.float64, 'button':np.str_, 'action':np.str_, 'x':np.int16,'y':np.int16})
	#log = log.replace('\n',' ', regex=True)
	#log.drop(['name'], axis = 1, inplace = True)
	#log.reset_index(drop=True, inplace=True)
	log['user'] = user
	log = log[['user', 'time', 'button', 'action', 'x', 'y']]
	#log.fillna(0, inplace = True)
	#log['x'] = log['x'].abs()
	log['x'] = log['x'].astype(int)
	#log['y'] = log['y'].abs()
	log['y'] = log['y'].astype(int)
	return log


def readKeyboardLog(fileName, user):
	log = pd.read_csv(fileName, sep=',')
	#log.drop(['name'], axis = 1, inplace = True)
	#log.reset_index(drop=True, inplace=True)
	log['user'] = user
	log = log[['user', 'key', 'press', 'release']]
	#log['x'] = log['x'].astype(int)
	#log['y'] = log['y'].astype(int)
	log.fillna(0, inplace = True)
	return log


def findDeviation(actions, actionType, threshold):
	result = {}
	n = len(actions)
	if n < threshold:
		return None
	vx, vy, v = [0], [0], [0]
	angles, path = [0], [0]
	sumA = 0
	actDist = 0
	o = [0]
	a = [0]
	startAccel = 0
	accel = True
	j = [0]
	c = [0]
	for i in range(1, n):
		dx = abs(actions[i]['x'] - actions[i-1]['x'])
		dy = abs(actions[i]['y'] - actions[i-1]['y'])
		dt = actions[i]['t']-actions[i-1]['t'] if actions[i]['t']-actions[i-1]['t'] != 0 else 0.01
		vx_ = dx/dt
		vy_ = dy/dt
		vx.append(vx_)
		vy.append(vy_)
		v.append(math.sqrt(vx_**2 + vy_**2))
		dist = math.sqrt(dx**2 + dy**2)
		actDist += dist
		path.append(actDist)
		ang = math.atan2(dy,dx)
		angles.append(ang)
		if i < n-1:
			dtheta = angles[i]-angles[i-1]
			o.append(dtheta/dt)
			dv = v[i] - v[i-1]
			if dv > 0 and accel:
				startAccel +=np.abs(dt)
			else:
				accel = False
			a.append(dv/dt)
			da = a[i] - a[i-1]
			j.append(da/dt)
			if i > 1:
				dp = path[i] - path[i-1]
				if dp == 0: continue
				dangle = angles[i] - angles[i-1]
				curv = dangle/dp
				c.append(curv)
	result['user'] = actions[0]['user']
	#result['time'] = actions[-1]['t'] - actions[0]['t']
	#result['theta'] = math.atan2(actions[-1]['y'] - actions[0]['y'], actions[-1]['x'] - actions[0]['x'])
	#result['direction'] = round((180+math.degrees(result['theta']))/45)-1
	#result['actDist'] = actDist
	#result['optDist'] = math.sqrt((actions[-1]['x'] - actions[0]['x'])**2 + (actions[-1]['y'] - actions[0]['y'])**2)
	result['straightness'] = 0 if actDist == 0 else math.sqrt((actions[-1]['x'] - actions[0]['x'])**2 + (actions[-1]['y'] - actions[0]['y'])**2)/actDist
	result['startAccel'] = startAccel
	metrics = [[vx,vy,v,o,a,j,c],['vx','vy','v','o','a','j','c']]
	for n, i in enumerate(metrics[0]):
		result['mean_'+metrics[1][n]] = np.mean(i)
		result['std_'+metrics[1][n]] = np.std(i)
		result['min_'+metrics[1][n]] = np.nanmin(i)
		result['max_'+metrics[1][n]] = np.max(i)
	return(result)


def processMouseFile(logFile):
	output = pd.DataFrame()
	prevEntry = None
	actionData = []
	for row in logFile.itertuples():
		entry = {
			'i': row[0],
			'user': row[1],
			't': row[2],
			'button': row[3],
			'action': row[4],
			'x': row[5],
			'y': row[6]}
		if row[4] == 'scroll':
			#Ignore scrolling
			if prevEntry != None:
				entry = prevEntry
		actionData.append(entry)
		if entry['t'] - actionData[0]['t'] > 3:
			output = output.append(findDeviation(actionData,3,5), ignore_index=True)
			actionData = []
			actionData.append(entry)
		if entry['button'] == 'left':
			if entry['action'] == 'press':
				if prevEntry['action'] == 'move':
					output = output.append(findDeviation(actionData,1,5), ignore_index=True)
					actionData = []
					actionData.append(entry)
			if entry['action'] == 'release':
				if len(actionData) > 3:
					if prevEntry['action'] == 'drag':
						output = output.append(findDeviation(actionData,2,5), ignore_index=True)
						actionData = []
						actionData.append(entry)
				if prevEntry['action'] == 'press':
					#click
					actionData = []
		if entry != prevEntry:
			prevEntry = entry
	return output


def processKeyboardFile(logFile):
	logFile['H'] = None
	logFile['PP'] = None
	logFile['PR'] = None
	logFile['RP'] = None
	logFile['RR'] = None
	logFile['F1'] = 0
	logFile['F2'] = 0
	logFile['F3'] = 0
	logFile['F4'] = 0
	logFile['F5'] = 0
	logFile['F6'] = 0
	logFile['F7'] = 0
	logFile['F8'] = 0
	logFile['F9'] = 0
	logFile['F0'] = 0
	logFile['KPS'] = 0
	logFile['RAT20K'] = 0
	for i in range(0,len(logFile['user'])):
		logFile['H'][i] = logFile['release'][i] - logFile['press'][i]
		if logFile['key'][i] in fingerPositions['F1']: logFile['F1'][i] = 1
		elif logFile['key'][i] in fingerPositions['F2']: logFile['F2'][i] = 1
		elif logFile['key'][i] in fingerPositions['F3']: logFile['F3'][i] = 1
		elif logFile['key'][i] in fingerPositions['F4']: logFile['F4'][i] = 1
		elif logFile['key'][i] in fingerPositions['F5']: logFile['F5'][i] = 1
		elif logFile['key'][i] in fingerPositions['F6']: logFile['F6'][i] = 1
		elif logFile['key'][i] in fingerPositions['F7']: logFile['F7'][i] = 1
		elif logFile['key'][i] in fingerPositions['F8']: logFile['F8'][i] = 1
		elif logFile['key'][i] in fingerPositions['F9']: logFile['F9'][i] = 1
		else: logFile['F0'][i] = 1
		if i > 0:
			logFile['PR'][i] = logFile['release'][i] - logFile['press'][i-1] if logFile['release'][i] - logFile['press'][i-1] < 3 else logFile['PR'].mean()
			logFile['PP'][i] = logFile['press'][i] - logFile['press'][i-1] if logFile['press'][i] - logFile['press'][i-1] < 3 else logFile['PP'].mean()
			logFile['RR'][i] = logFile['release'][i] - logFile['release'][i-1] if logFile['release'][i] - logFile['release'][i-1] < 3 else logFile['RR'].mean()
			logFile['RP'][i] = logFile['press'][i] - logFile['release'][i-1] if logFile['press'][i] - logFile['release'][i-1] < 3 else logFile['RP'].mean()
		if i > 15 and i < len(logFile['user'])-15:
			logFile['KPS'][i] = (len([n for n in range(1,16) if logFile['press'][i]-logFile['press'][i-n] < 1 and logFile['key'][i-n] in letters]+[k for k in range(1,16) if logFile['press'][i+k]-logFile['press'][i] < 1 and logFile['key'][i+k] in letters]))
		if i < len(logFile['user'])-10 and i >= 10:
			logFile['RAT20K'] = np.sum([logFile['press'][10+i-n]-logFile['press'][i-10] for n in range(0,20)])
	logFile.drop(['key','press','release'], axis = 1, inplace = True)
	logFile.drop([0], axis = 0, inplace = True)
	logFile.reset_index(drop=True, inplace=True)
	logFile.fillna(0, inplace = True)
	return(logFile.iloc[0:,0:-1])
			

def buildMouseTraining(desired, undesired):
	allLogs = pd.DataFrame()
	file = 'Data/Training/M' + str(desired) + '.csv'
	allLogs = allLogs.append(processMouseFile(readMouseLog(file,1)).iloc[0:,0:], ignore_index = True)
	for u in undesired:
		try:
			file2 = 'Data/Training/M' + str(u) + '.csv'
			allLogs = allLogs.append(processMouseFile(readMouseLog(file2,0)).iloc[0:,0:], ignore_index = True)
		except:
			print('Error, cannot find file: '+file2)
			continue
	data = allLogs
	X = data.values[:,:data.shape[1]-1]
	y = data.values[:,data.shape[1]-1]
	#Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3)
	model = RandomForestClassifier(n_estimators=100)
	#model.fit(Xt, yt)
	#predictions = model.predict(Xv)
	#print(classification_report(yv, predictions))
	model.fit(X,y)
	with open('Data/Models/trainingMouseModel-'+str(desired)+'.pkl', 'wb') as file:
	    pickle.dump(model, file)


def buildKeyTraining(desired, undesired):
	allLogs = pd.DataFrame()
	file = 'Data/Training/K' + str(desired) + '.csv'
	allLogs = allLogs.append(processKeyboardFile(readKeyboardLog(file,1)).iloc[0:,0:], ignore_index = True)
	for u in undesired:
		try:
			file2 = 'Data/Training/K' + str(u) + '.csv'
			allLogs = allLogs.append(processKeyboardFile(readKeyboardLog(file2,0)).iloc[0:,0:], ignore_index = True)
		except:
			print('Error, cannot find file: '+file2)
			continue
	data = allLogs
	X = data.values[:,1:]
	y = data.values[:,0]
	y = y.astype('int')
	#Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3)
	model = RandomForestClassifier(n_estimators=100)
	#model.fit(Xt, yt)
	#predictions = model.predict(Xv)
	#print(classification_report(yv, predictions))
	model.fit(X,y)
	with open('Data/Models/trainingKeyModel-'+str(desired)+'.pkl', 'wb') as file:
		pickle.dump(model, file)


def testMouseAgainst(data, modelName):
	with open(modelName, 'rb') as file:
		model = pickle.load(file)
	X = data.values[:,:data.shape[1]-1]
	y = model.predict(X)
	return y


def testKeyAgainst(data, modelName):
	with open(modelName, 'rb') as file:
		model = pickle.load(file)
	X = data.values[:,1:]
	y = model.predict(X)
	return y


def rawToReadable(dataframe):
	keys = {}
	kp = []
	dataframe.columns = ['device','time','action','button','x','y']
	mouseEvents = dataframe.copy()
	mouseEvents = mouseEvents[~mouseEvents.device.str.contains('keyboard')]
	mouseEvents.drop(['device'], axis = 1, inplace = True)
	mouseEvents.reset_index(drop=True, inplace=True)
	mouseEvents['time'] = (mouseEvents['time'] - mouseEvents['time'][0]).dt.total_seconds()
	mouseEvents = mouseEvents[['time', 'button', 'action', 'x', 'y']]
	held = False
	for i,y in enumerate(mouseEvents['action']):
		if y == 'move':
			if held:
				mouseEvents['action'][i] = 'drag'
		if y == 'press':
			held = True
		if y == 'release':
			held = False
	keyEvents = dataframe.copy()
	keyEvents = keyEvents[~keyEvents.device.str.contains('mouse')]
	keyEvents.drop(['device', 'x', 'y'], axis = 1, inplace = True)
	#keyEvents.drop([0], axis = 0, inplace = True)
	keyEvents.reset_index(drop=True, inplace=True)
	keyEvents['time'] = (keyEvents['time'] - keyEvents['time'][0]).dt.total_seconds()
	keyEvents = keyEvents[['time', 'button', 'action']]
	for i, entry in enumerate(keyEvents['action']):

		if entry == 'press':
			if keys.get(keyEvents['button'][i].lower()) == None:
				keys[keyEvents['button'][i].lower()] = len(kp)
				kp.append([keyEvents['button'][i].lower(),keyEvents['time'][i],None])
		elif entry == 'release':
			try:
				kp[keys[keyEvents['button'][i].lower()]][2] = keyEvents['time'][i]
				keys[keyEvents['button'][i].lower()] = None
			except:
				pass
	keyEvents = pd.DataFrame(kp)
	keyEvents = keyEvents.rename(columns={0:'key',1:'press',2:'release'})
	#keyEvents.columns=['key','release','press']
	return([mouseEvents,keyEvents])