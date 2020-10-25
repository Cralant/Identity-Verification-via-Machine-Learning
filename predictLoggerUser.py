import os
from pynput import mouse, keyboard
from datetime import datetime
import functions as fn
import numpy as np
import glob
import csv
import pandas as pd
import time

log = []
print("Recording user input now, press 'esc' to end logging software and predict results.")

#Get time
def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

#Write to file
def log_write(input, timestamp, value, action, x, y):
    log.append([input, datetime.now(), action, value, x, y])

# When the mouse is moved
def on_move(x, y):
    log_write('mouse',get_time(),None,'move',x,y)

# When the mouse is clicked
def on_click(x, y, button, pressed):
    if pressed:
        log_write('mouse',get_time(),button.name,'press',x,y)
    else:
        log_write('mouse',get_time(),button.name,'release',x,y)

# When the mouse wheel is scrolled
def on_scroll(x, y, dx, dy):
    log_write('mouse',get_time(),None,'scroll',dx,dy)

# On key press
def on_press(key):
    # Stop listener on f1 key
    if key == keyboard.Key.esc:
        listener.stop()
    try:
        if str(key) == '\',\'':
            log_write('keyboard',get_time(),'key.comma','press',None,None)
        elif str(key) == '\'.\'':
            log_write('keyboard',get_time(),'key.period','press',None,None)
        elif str(key) == '\' \'':
            log_write('keyboard',get_time(),'key.space','press',None,None)
        else:
            log_write('keyboard',get_time(),'key.'+str(key.char).lower(),'press',None,None)
    except:
        log_write('keyboard',get_time(),str(key).lower(),'press',None,None)


# On key release
def on_release(key):
    try:
        if str(key) == '\',\'':
            log_write('keyboard',get_time(),'key.comma','release',None,None)
        elif str(key) == '\'.\'':
            log_write('keyboard',get_time(),'key.period','release',None,None)
        elif str(key) == '\' \'':
            log_write('keyboard',get_time(),'key.space','release',None,None)
        else:
            log_write('keyboard',get_time(),'key.'+str(key.char).lower(),'release',None,None)
    except:
        log_write('keyboard',get_time(),str(key).lower(),'release',None,None)


# Start listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    with mouse.Listener(on_click=on_click, on_scroll=on_scroll, on_move=on_move) as listener: 
        listener.join()
    
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
for i,u in enumerate(users):
    mModelName = 'Data/Models/trainingMouseModel-' + u + '.pkl'
    kModelName = 'Data/Models/trainingKeyModel-' + u + '.pkl'
    results[0].append([i,np.mean(fn.testMouseAgainst(proccessedMouse,mModelName))])
    results[1].append([i,np.mean(fn.testKeyAgainst(processedKey,kModelName))])
    sortedResults = [sorted(results[0], key=lambda t: t[:][1], reverse=True), sorted(results[1], key=lambda t: t[:][1], reverse=True),]
print(f'Best match found for mouse is {users[sortedResults[0][0][0]]}-{round(sortedResults[0][0][1]*100,2)}%, and keyboard is {users[sortedResults[1][0][0]]}-{round(sortedResults[1][0][1]*100,2)}% - All:{[p[1] for p in results[0]]},{[p[1] for p in results[1]]}')
time.sleep(30)