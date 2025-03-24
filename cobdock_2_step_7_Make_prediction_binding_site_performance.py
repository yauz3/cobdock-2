#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/09/2024
# Author: Sadettin Y. Ugurlu

import requests
import os
import pickle
import random
import sys
import warnings
import xlsxwriter
import numpy as np
import pandas as pd
import statistics
import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import (
    average_precision_score, f1_score, accuracy_score, recall_score,
    precision_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from tabulate import tabulate

# ========== Set current working directory ==========
current_dir = os.path.dirname(os.path.abspath(__file__))

# ========== Function to load AutoGluon model ==========
def model_fn(model_dir):
    """Loads a trained AutoGluon model from the given directory"""
    model = TabularPredictor.load(model_dir)
    model.persist_models()  # Ensures models are stored for future access
    return model

# Show all columns in pandas DataFrames
pd.set_option('display.max_columns', None)

# ========== Save predicted probabilities to CSV ==========
def save_predictions(y_proba, protein, proba_number):
    """Saves prediction probabilities to CSV"""
    df = pd.DataFrame(y_proba)
    df.to_csv(f'y_proba_{proba_number}_{protein}.csv', index=False)

def save_predictions_updated(y_proba, protein, proba_number, test_data):
    """Saves prediction probabilities with additional metadata"""
    df_proba = pd.DataFrame(y_proba, columns=[f'prob_class_{i}' for i in range(y_proba.shape[1])])
    additional_columns = test_data[['protein_pocket', 'label', 'distance', 'mass_center_x',
                                    'mass_center_y', 'mass_center_z', 'fpx_x', 'fpx_y', 'fpx_z']]
    df_combined = pd.concat([additional_columns.reset_index(drop=True), df_proba], axis=1)
    df_combined.to_csv(f'y_proba_{proba_number}_{protein}.csv', index=False)

# ========== Initialize counters ==========
b = 0  # Number of correct top predictions (true label = 1)
c = 0  # Total number of predictions

# ========== Load 3 prediction models ==========
model_1 = model_fn(f"{current_dir}/AutogluonModels/MODEL_NAME_1/")
model_2 = model_fn(f"{current_dir}/AutogluonModels/MODEL_NAME_2/")
model_3 = model_fn(f"{current_dir}/AutogluonModels/MODEL_NAME_3/")


# Other test sets (currently unused)
dude_test=['ital', 'mk14', 'abl1', 'ppard', 'wee1', 'pparg', 'thrb', 'hivpr', 'cxcr4', 'mp2k1', 'tysy', 'akt1', 'hxk4', 'mapk2', 'fak1', 'pa2ga', 'csf1r', 'pnph', 'fnta', 'nos1', 'fkb1a', 'xiap', 'mk01', 'aofb', 'aces', 'gcr', 'aldr', 'dyr', 'inha', 'gria2', 'hdac2', 'kith', 'ampc', 'ace', 'plk1', 'urok', 'hivrt', 'hivint', 'ppara', 'pgh2', 'tgfr1', 'rxra', 'hmdh', 'cp2c9', 'sahh', 'thb', 'akt2', 'tryb1', 'aa2ar', 'def', 'lck', 'jak2', 'fabp4', 'casp3', 'pgh1', 'fa10', 'hdac8', 'mk10', 'parp1', 'esr2', 'mmp13', 'dpp4', 'adrb1', 'fa7', 'kif11', 'adrb2', 'ada', 'hs90a', 'reni', 'lkha4', 'kpcb', 'rock1', 'fgfr1', 'andr', 'esr1', 'grik1', 'pur2', 'cdk2', 'pygm', 'drd3', 'fpps', 'mcr', 'met', 'kit', 'igf1r', 'comt', 'bace1', 'pde5a', 'dhi1', 'nram', 'braf', 'glcm', 'prgr', 'cp3a4', 'try1', 'ptn1', 'cah2', 'src', 'egfr', 'vgfr2', 'pyrd', 'ada17']
mti_test=['3erd-A', '4asd-A', '1gs4-A', '1ki2-A', '1rv1-A', '1fm6-ABD', '1z95-A', '3h0a-ABD', '2p16-AL', '3mxf-A', '2hyy-A', '1m48-A', '3vn2-AC', '2gqg-A', '3clx-A', '2yek-A', '2ydo-A', '1fm9-ABD', '4ag8-A', '2yxj-A', '2w26-AB', '4mxo-A', '3dzy-ADE', '4agd-A', '1t4e-A', '4ey7-A', '4ey6-A']
ads_test=['1s19-A', '1hww-A', '1xoq-B', '1gpk-A', '1ke5-A', '1n46-A', '1l2s-B', '1k3u-A', '1jje-A', '1yv3-A', '1sg0-AB', '1kzk', '1yvf-A', '1vcj-A', '1unl-AD', '1hvy-D', '1owe-A', '1w1p-B', '1hnn-A', '1of6-A', '1x8x-A', '1w2g-B', '1z95-A', '1r58-A', '1s3v-A', '1sq5-A', '1v0p-A', '1hq2-A', '1t46-A', '1xm6-A', '1y6b-A', '1of1-B', '1oq5-A', '1mmv-B', '1v4s-A', '1n2j-A', '1oyt-HIL', '1n2v-A', '1q4g-B', '1xoz-A', '2br1-A', '1g9v-AB', '1uml-A', '1meh-A', '1tz8-AB', '1lpz-AB', '1q41-A', '1yqy-A', '1jla-A', '1pmn-A', '2bm2-B', '1ia1-B', '1ygc-HL', '1lrh-D', '1r1h-A', '1opk-A', '1u1c-F', '1sj0-A', '1t40-A', '1tt1-A', '2bsm-A', '1ywr-A', '1jd0-B', '1mzc-AB', '1p62-B', '1nav-A', '1u4d-A', '1ig3-AB', '1v48-A', '1t9b-A', '1tow-A', '1p2y-A', '1hwi-AB', '1j3j-A', '1l7f-A', '1hp0-A', '1r55-A', '1m2z-A', '1sqn-B', '1n1m-A', '1gkc-A', '1uou-A', '1q1g-F', '1r9o-A', '1gm8-AB']
casf_test = ['3up2', '3cj4', '1q8u', '2xj7', '2zb1', '4eo8', '4f3c', '3kgp', '2br1', '1lpg', '3rsx', '3zdg', '3ge7',
               '1yc1', '3gbb', '2qbp', '1gpk', '3e92', '3uew', '1y6r', '1sqa', '3ag9', '1ydt', '3uuo', '3u9q', '2qnq',
               '3d4z', '4qd6', '3b1m', '1nc1', '2w4x', '2hb1', '3zsx', '2xbv', '4ivd', '1ydr', '1e66', '3jvs', '4w9i',
               '5a7b', '4f2w', '3ao4', '3kwa', '4e5w', '3n76', '4llx', '3g0w', '4agq', '3gy4', '4ivc', '2c3i', '1h23',
               '2ymd', '1z6e', '2j78', '2fxs', '4ih7', '3n86', '2xii', '3u8k', '1bzc', '5aba', '3f3e', '2wn9', '2wca',
               '3e5a', '1k1i', '4dld', '3coz', '3b65', '3prs', '2zcr', '1a30', '4w9h', '4agn', '1o5b', '2iwx', '3wz8',
               '4dli', '4u4s', '2wvt', '2wnc', '4agp', '4e6q', '3arq', '2xys', '3bgz', '4crc', '4ddh', '2x00', '4kzq',
               '1gpn', '4ea2', '2v7a', '4cr9', '5c28', '4qac', '3ejr', '3dx2', '1nc3', '3acw', '4ivb', '2cet', '5dwr',
               '2cbv', '2wbg', '1s38', '4f09', '3coy', '2j7h', '2zcq', '3ui7', '3g31', '4jsz', '3oe4', '2v00', '3lka',
               '4mme', '3wtj', '4eky', '4j3l', '4gid', '3utu', '3k5v', '3ebp', '4f9w', '3fur', '4abg', '1g2k', '1syi',
               '1ps3', '3f3c', '3b27', '3pxf', '3e93', '3uex', '3dx1', '3p5o', '3l7b', '2yki', '4ty7', '4ciw', '3nq9',
               '1vso', '3gv9', '4j21', '1mq6', '2wtv', '3jvr', '1owh', '3o9i', '4j28', '4m0z', '4jia', '4jxs', '4de2',
               '3uri', '1p1n', '2xdl', '2qbr', '3ozt', '3nw9', '3qgy', '2brb', '3kr8', '1nvq', '1p1q', '2qbq', '3arv',
               '3tsk', '1q8t', '2y5h', '4ih5', '3f3d', '3oe5', '1r5y', '4gfm', '3ryj', '3arp', '4jfs', '4kz6', '3pyy',
               '4x6p', '4lzs', '4pcs', '1o3f', '1z95', '3ivg', '3g2z', '4wiv', '4gr0', '3zt2', '4k18', '3g2n', '3n7a',
               '1c5z', '2vkm', '2al5', '3qqs', '3ary', '3b68', '4w9c', '3dd0', '3uo4', '2p15', '3u8n', '4w9l', '3gnw',
               '4de1', '2vw5', '2yfe', '1uto', '2yge', '3fv1', '1h22', '4cra', '2zy1', '2wer', '3myg', '3ueu', '4bkt',
               '4m0y', '2zda', '3pww', '3bv9', '3b5r', '3f3a', '2vvn', '3fv2', '4de3', '4djv', '5c2h', '2xb8', '3nx7',
               '4rfm', '4cig', '1eby', '4kzu', '4ddk', '3uev', '3twp', '3zso', '4ogj', '1oyt', '3udh', '4gkm', '1qkt',
               '4twp', '3ozs', '2w66', '3gr2', '1bcu', '3gc5', '3r88', '3jya', '3rr4', '3ehy', '3rlr', '4k77', '2weg',
               '3syr', '4mgd', '3u5j', '3aru', '4owm', '4hge']
pdbbind_test = ['5ort', '4ch8', '5szc', '1bwn', '3twr', '6fzj', '4o3u', '4gj7', '4jmu', '5nme', '3avg',
                             '2fsa',
                             '2o65', '3ant', '2llq', '2g24', '5t35', '4jq7', '2v85', '4fhi', '5ick', '5wfc', '3m17',
                             '4urk',
                             '4pns', '4jpc', '5ct7', '3sou', '4ayv', '4xg3', '1qhr', '5he5', '5q1i', '4bd3', '6e83',
                             '3mrx',
                             '5j18', '2m3z', '5wik', '4ono', '4nb6', '1x78', '6f6u', '5em7', '3ds4', '5d6y', '3n4l',
                             '2c1n',
                             '6miq', '6di1', '4zjr', '4mxc', '3cqw']

# ========== Loop through all proteins in DUDE test set ==========
for i in dude_test:
    # Change working directory to DUDE test folder
    os.chdir(f"{current_dir}/test/DUDE")

    # Load test data CSV
    test_data = pd.read_csv(f"{i}.csv")
    test_data = test_data.drop(columns=['protein_pocket'])  # Drop unnecessary column

    # Predict probabilities using all 3 models
    y_proba_1 = model_1.predict_proba(test_data)
    y_proba_2 = model_2.predict_proba(test_data)
    y_proba_3 = model_3.predict_proba(test_data)

    # Save raw prediction probabilities for each model
    save_predictions(y_proba=y_proba_1, protein=i, proba_number=1)
    save_predictions(y_proba=y_proba_2, protein=i, proba_number=22)
    save_predictions(y_proba=y_proba_3, protein=i, proba_number=3)

    # Average the probabilities from 3 models
    average_proba = (y_proba_1 + y_proba_2 + y_proba_3) / 3

    # Get the row index of the highest average probability for class 1
    max_index = average_proba[1].idxmax()
    max_value = average_proba[1].max()  # Optional: get max prob value (not used here)
    label_value = test_data['label'].iloc[max_index]  # Get the actual label for that row

    # Check if the top predicted sample was a true positive
    if label_value == 1:
        b += 1  # Increment correct count

    c += 1  # Increment total count

    # Print current running accuracy
    print("current_accucary", b / c)

# ========== Final accuracy output ==========
print("accucary", b / len(pdbbind_test))  # Might be a mistake: DUDE used above, not pdbbind
