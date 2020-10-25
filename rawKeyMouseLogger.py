# Import mouse reading
import os
from pynput import mouse, keyboard
from datetime import datetime

file = input("Please enter your name: ")
log = open(file+'.csv',"w+")

#Get time
def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

#Write to file
def log_write(input, timestamp, value, action, x, y):
    log.write(f'{input},{timestamp},{action},{value},{x},{y}\n')

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
        log.close()
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