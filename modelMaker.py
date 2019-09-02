import json
import csv
from pyModelChecking.LTL import *
# from pyModelChecking.CTLS import *
from pyModelChecking import *
import tkinter as tk
from tkinter import Label,Tk
from PIL import Image, ImageTk
import PIL.Image
import json
from tkinter import messagebox

from tkinter import filedialog
from tkinter import *
import sys

segmentation = json.loads(open('data.json').read())
functions = {}
relations = []
with open('1700wxoffsetyoffsetxy.csv') as tracker:
    fixation = csv.reader(tracker, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    print(fixation)
    for index_fixation, row in enumerate(fixation):
        relations.append((index_fixation,index_fixation+1))
        for index_segment, segment in enumerate(segmentation):
            if (float(row[2]) >= float(segment['x0']) and float(row[2]) <= float(segment['x1'])):
                if (float(row[3]) >= float(segment['y0']) and float(row[3]) <= float(segment['y1'])):
                   #print ('fixation index', index_fixation, 'segment index', index_segment, 'match')
                   functions.update({index_fixation: [segment['aoi']]})
                   break
                else:
                    functions.update({index_fixation: ['undefined']})
            else:
                functions.update({index_fixation: ['undefined']})
    relations.append((len(relations),len(relations)))
print('relations', len(relations))
print('functions', len(functions))
print(relations)
print(functions)

K = Kripke(R=relations, L=functions)
f = input()
parser = Parser()
psi = parser(f)
print(psi)
print(modelcheck(K, psi))

