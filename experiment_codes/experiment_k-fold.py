# This is an experiment for train and test data with k-fold cross validation test.


import functions.data as fdata
import models as mdl

# Paths
trial_label = [
    'A',
    'B',
    'C',
    'D'
]
data_path = [
    # TEST TRIAL
    'Path',
    'OF',
    'YOUR',
    'RAW_DATASETS'
]

# Parameters
EXPERIMENT_EPOCHS = 20  # k of the k-fold cross validation

NULL_MODE = False  # True: Test NULL model, False: Test genuine model
fs = 20  # Hz
time_bin = 100  # ms
class_organization = [
    ['A', 'C'],  # Class 0
    ['B', 'D']  # Class 1
]
train_ratio = 0.75

# Variables
RESULT_ACCURACY = []

# EXPERIMENT ITERATION
for epoch in range(0, EXPERIMENT_EPOCHS):
    # Notice experiment epoch
    print("Epoch: {}".format(epoch))

    # Import data
    data = fdata.import_dataset(trial_label, data_path, fs, time_bin)

    # Split to train and test datasets
    train_data, test_data = fdata.split_dataset(data, train_ratio, k_fold=EXPERIMENT_EPOCHS, current_epoch=epoch)

    # Feature selection
    data_train_feature = fdata.feature_selection(train_data)
    data_test_feature = fdata.feature_selection(test_data)

    # Make class
    classes_train = fdata.make_class(data_train_feature, class_organization)
    classes_test = fdata.make_class(data_test_feature, class_organization)

    # Make train dataset
    train_dataset, train_label = fdata.make_dataset2(classes_train, bool_shuffle=True, bool_null_mode=NULL_MODE)
    test_dataset, test_label = fdata.make_dataset2(classes_test, bool_null_mode=False)  # Do not shuffle test data

    # Train model
    model = mdl.train_model_xor(train_dataset, train_label)

    # Test model
    accuracy = mdl.test_model(model, test_dataset, test_label, percent=False)

    # Store accuracy
    RESULT_ACCURACY.append(accuracy)

# Report estimated performance
mdl.report_performance(RESULT_ACCURACY, bool_percent=False)
