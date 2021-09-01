#imports 
from pymouse import PyMouse
from pykeyboard import PyKeyboard

#set mouse and keyboard as variables
m = PyMouse()
k = PyKeyboard()

#set mouse coordinates according to the screen_size
x_dim, y_dim = m.screen_size()

#click mouse at the selected position
m.click(x_dim/2, y_dim/2, 1)

#enter text using keyboard variable
k.type_string('#This is a Test Message!')


