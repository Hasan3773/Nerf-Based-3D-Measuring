import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Extract points from the countour map of the top down view
# figure out which curve fits best, prob polynomial, and fit curves
# what do I do with these new curves
# I could take splices in all 4 directions, then fit lines and curves to each shape, then output measurements on an orthographic view 
# I wanna choose which measurements are important, I could technically train an Neural Net that takes in all the measurements of each view,
# then it would display the necassary measurements, like lengths, width, and hopefully radius's of curves 
# Although a traditional algo would probobly be better, and I'm not sure what I would train it on
# How do I display the orthographic measurments
# I could just show the contour maps arranged in an ortho view 
# or I could integrate with a 3d modelling and have it output an actual ortho file or something 