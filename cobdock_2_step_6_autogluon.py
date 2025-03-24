#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/09/2024
# Author: Sadettin Y. Ugurlu
import pandas as pd
import requests
import os
import os
import pickle
import random
import sys
import warnings
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer,StandardScaler,RobustScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import random
import numpy as np
from autogluon.tabular import TabularPredictor
from datetime import datetime

# Set column numbers
pd.set_option('display.max_columns', None)

# Read training.csv
training_data=pd.read_csv("training_290_ready.csv")
# Drop columns may causes data leak
training_data=training_data.drop(columns=['protein_pocket',"distance","mass_center_x","mass_center_y","mass_center_z",
                                          "fpx_x","fpx_y","fpx_z"])
# The feature should be change
# Autogluon need label in the training data
sample_features=['label','Score', 'Druggability Score', 'Number of Alpha Spheres', 'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume', 'Mean local hydrophobic density', 'Mean alpha sphere radius', 'Mean alp. sph. solvent access', 'Apolar alpha sphere proportion', 'Hydrophobicity score', 'Volume score', 'Polarity score', 'Charge score', 'Proportion of polar atoms', 'Alpha sphere density', 'Cent. of mass - Alpha Sphere max dist', 'Flexibility',]
# Keep only selected features
training_data = training_data.loc[:, sample_features]

# Convert into TabularDataset
train_data_fpocket = TabularDataset(training_data)

# Train the model
# Once auto_stacking is True, Autogluon automatically select validation data from training set.
TabularPredictor(label="label", eval_metric='average_precision').fit(
        train_data_fpocket, auto_stack=True, presets="good_quality",
        time_limit=60*10, num_stack_levels=0, num_bag_folds=8, num_bag_sets=20
        )

