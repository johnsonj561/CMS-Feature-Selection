from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import datetime
import pandas as pd
import numpy as np
import os
import timeit

import sys
sys.path.append(os.environ['CMS_ROOT'])

from cms_modules.utils import (
    apply_ros_rus,
    get_binary_imbalance_ratio,
    split_on_binary_attribute)
from cms_modules.logging import Logger

logger = Logger()
logger.log_message('Executing Chi-Squared Feature Selection with Random Forest Learner')

data_path = os.environ['CMS_PARTB_PATH']
partB_train_normalized_key = 'partB_train_normalized'
partB_test_normalized_key = 'partB_test_normalized'
timestamp = datetime.now().strftime("%m.%d.%Y-%H:%M:%S")
results_file = f'./results.{timestamp}.csv'

train_data = pd.read_hdf(data_path, 'partB_train_normalized')
test_data = pd.read_hdf(data_path, 'partB_test_normalized')

logger.log_message('Data imbalance levels before sampling')
logger.log_message(get_binary_imbalance_ratio(train_data['exclusion']))
logger.log_message('Size of train data = ' + str(len(train_data)))

# initialize results
header = 'index,subset_size,minority_size,run,roc_auc,time_elapsed\n'
with open(results_file, 'a') as outfile:
    outfile.write(header)

repetitions = 30
ros_rate = 1
minority_ratios = [0.2889, 0.0575, 0.0318, 0.0013, 0.0005, 0.0003, 0.0002]
feature_subset_sizes = [125, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]

counter = 0

# iterating over all class imbalance levels
for minority_ratio in minority_ratios:

    # iterating over all feature subset levels
    for subset_size in feature_subset_sizes:
        logger.log_message(f'Evaluating class imbalance level {minority_ratio} and feature subset size {subset_size}')

        # iterate over 30 runs
        for run in range(repetitions):
            logger.log_message(f'Starting run {run}')

            # take random sample from the training data
            train_data = pd.read_hdf(data_path, 'partB_train_normalized')
            test_data = pd.read_hdf(data_path, 'partB_test_normalized')
            pos_train, neg_train = split_on_binary_attribute(train_data, attribute='exclusion', pos_label=1, neg_label=0)
            train_data = apply_ros_rus(pos_train, neg_train, ros_rate=ros_rate, rus_rate=minority_ratio)
            del pos_train
            del neg_train

            logger.log_message('Minority class ratio after sampling: ')
            logger.log_message(get_binary_imbalance_ratio(train_data['exclusion']))


            # separate features from labels
            train_y = train_data['exclusion']
            train_x = train_data.drop(columns=['exclusion'])
            test_y = test_data['exclusion']
            test_x = test_data.drop(columns=['exclusion'])

            # create subset of features
            feature_selector = SelectKBest(chi2, k=subset_size).fit(train_x, train_y)
            train_x_subset = feature_selector.transform(train_x)
            test_x_subset = feature_selector.transform(test_x)

            mask = feature_selector.get_support()
            features = train_x.columns[mask]
            logger.log_message(f'Using features {features.values}')

            start = timeit.default_timer()
            output = f'{counter},{subset_size},{minority_ratio},{run},'

            # train a random forest
            lr_model = LogisticRegression(n_jobs=10)
            lr_model.fit(train_x_subset, train_y)

            # record performance
            posteriors = lr_model.predict_proba(test_x_subset)
            roc_auc = roc_auc_score(test_y, posteriors[:, 1])

            # record results
            time_elapsed = timeit.default_timer() - start
            output += f'{roc_auc},{time_elapsed}\n'
            logger.log_message(f'Ending run {run}')
            logger.log_message(f'Results {output}')
            with open(results_file, 'a') as outfile:
                outfile.write(output)

            counter += 1
