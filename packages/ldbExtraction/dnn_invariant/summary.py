import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/home/mohit/Mohit/model_interpretation/ai-adversarial-detection")

from dnn_invariant.models.models4invariant import *
from dnn_invariant.utilities.trainer import *
from dnn_invariant.utilities.datasets import *
from dnn_invariant.utilities.environ import *

gap = np.around(output_size / 6).astype(int)
box_edge = output_size + gap
box_edge = box_edge.astype(int)
right_margin = np.around(output_size / 2).astype(int)
top_margin = np.around(output_size / 2).astype(int)

def num_row(fnames):
    output = 0
    for fname in fnames:
        count = 0
        for i in range(len(fname)):
            if fname[i].isdigit():

                if not fname[i+1].isdigit():
                    count += 1
            if count == 2:
                char = int(fname[i])
                if char > output:
                    output = char
                break

    return output + 1

def num_col(fnames):
    output = 0
    for fname in fnames:
        for i in range(len(fname)):
            if fname[i]=='d' and fname[i+1]=='b' and fname[i+2]!='_':
                db = fname[i+2:-4]
                db = int(db)
                if db > output:
                    output = db

    return output + 1 + 1 + 2 + 1 # + 1 as Python starts with 0;  + 1 for original image; + 2 for 2 baseline methods; + 1 for average heatmap


def position(fname):
    row = 0
    count = 0
    col = 1 # Assume this is an original image
    for i in range(len(fname) - 1):
        if fname[i] == 'd' and fname[i + 1] == 'b' and fname[i+2]!='_':
            db = fname[i + 2:-4]
            db = int(db)
            col = db + 1 + 1 + 2 + 1 # + 1 as Python starts with 0;  + 1 for original image; + 2 for 2 baseline methods; + 1 for average heatmap
        elif fname[i] == 'g' and fname[i + 1] == 'r':
            col = 2 # baseline method: Grad-CAM
        elif fname[i] == 'l' and fname[i + 1] == 'i':
            col = 3 # baseline method: LIME
        elif fname[i] == 'd' and fname[i+1] == 'b' and fname[i+2] == '_':
            col = 4 # Average heatmap

        if fname[i].isdigit():
            if not fname[i+1].isdigit():
                count += 1
        if count == 2:
            row = int(fname[i]) + 1
            count = 1000

    return row, col

root = '/home/mohit/Mohit/model_interpretation/ai-adversarial-detection/dnn_invariant/logs'
experiments = os.listdir(root)

for experiment in experiments:
    path = os.path.join(root, experiment)
    rules = os.listdir(path)
    for rule in rules:
        images = os.path.join(path, rule)
        summary = np.ones((box_edge * num_row(os.listdir(images)) + right_margin, box_edge * num_col(os.listdir(images)) + top_margin, 3))
        for image in os.listdir(images):
            print(image)
            img = plt.imread(os.path.join(images, image))
            r, n = position(image)
            summary[(r - 1) * box_edge + right_margin:(r - 1) * box_edge + output_size + right_margin, (n - 1) * box_edge + top_margin:(n - 1) * box_edge + output_size + top_margin, :] = img / 255

        name = ''
        for i in range(len(class_list)):
            if class_list[i] in os.listdir(images)[0]:
                name = class_list[i]

        plt.imsave(os.path.join(root, experiment + '_' + name + '_' + rule + '_summary.jpg'), summary)

