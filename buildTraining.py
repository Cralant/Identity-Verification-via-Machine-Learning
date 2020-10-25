import csv
import functions as fn

users = list(csv.reader(open('Data/users.csv')))[0]
for n,user in enumerate(users):
	print(user, users[:n]+users[n+1:])
	try:
		fn.buildMouseTraining(user,users[:n]+users[n+1:])
	except:
		print('Error, cannot build mouse model')
	try:
		fn.buildKeyTraining(user,users[:n]+users[n+1:])
	except:
		print('Error, cannot build key model')