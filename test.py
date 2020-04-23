# DQN vs. Heuristic Test
import environment as gym
import time
import numpy as np
from model import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # FIXME: is this needed for tensorflow model save?
import matplotlib.pyplot as pyplot

def organize_output(output, new_output):
    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True