import matplotlib as plt
import numpy as np
import os
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator

file_path ='./exercise_data'
label_path = './Labels.json'




gen = ImageGenerator(file_path, label_path, 20, (32,32,3), False, False, False)

gen.next()
gen.show()