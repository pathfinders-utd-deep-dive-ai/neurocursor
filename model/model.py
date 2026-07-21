import argparse
import json
import os
import re

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (BatchNormalization, Conv2D, Dense,
        Dropout, GRU, InputLayer, Masking, MaxPooling2D)
from sklearn.model_selection import train_test_split
