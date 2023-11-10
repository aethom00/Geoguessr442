from gui import GUI

import tensorflow as tf
import numpy as np

g = GUI(15, 10)
g.init()

correct = tuple((83.0458, 42.3314))
guess = tuple((85.6206, 44.7631))

g.generate_random_output()
g.show(correct, guess)



# need to calculate correct and guess and plot them

# g.clear_output()
# g.show()

# Cordinates for Detriot, MI: 83.0458 W, 42.3314 N
# Cordintaes for Traverse City, MI: 85.6206 W, 44.7631 N


