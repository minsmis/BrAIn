import functions as fc
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
    '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_1.mat',
    '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_2.mat',
    '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_3.mat',
    '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_4.mat'

    # TEST NULL
    # '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_1.mat',
    # '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_1.mat',
    # '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_1.mat',
    # '/Users/minseokkim/Desktop/Social_Calcium_raw/IL212_1.mat'
]

# Parameters
EXPERIMENT_EPOCHS = 20

NULL_MODE = False  # True: Test NULL model, False: Test genuine model
fs = 20  # Hz
time_bin = 100  # ms
class_organization = [
    ['A', 'B'],  # Class 0
    ['C', 'D']  # Class 1
]
train_ratio = 0.75

# Variables
RESULT_ACCURACY = []

# EXPERIMENT ITERATION
for epoch in range(0, EXPERIMENT_EPOCHS):
    # Notice experiment epoch
    print("Epoch: {}".format(epoch))

    # Import data
    data = fc.import_dataset(trial_label, data_path, fs, time_bin)

    # Split to train and test datasets
    train_data, test_data = fc.split_dataset(data, train_ratio)

    # Feature selection
    data_train_feature = fc.feature_selection(train_data)
    data_test_feature = fc.feature_selection(test_data)

    # Make class
    classes_train = fc.make_class(data_train_feature, class_organization)
    classes_test = fc.make_class(data_test_feature, class_organization)

    # Make train dataset
    train_dataset, train_label = fc.make_dataset2(classes_train, bool_shuffle=True, bool_null_mode=NULL_MODE)
    test_dataset, test_label = fc.make_dataset2(classes_test, bool_null_mode=False)  # Do not shuffle test data

    # Train model
    model = mdl.train_model_xor(train_dataset, train_label)

    # Test model
    accuracy = mdl.test_model(model, test_dataset, test_label, percent=False)

    # Store accuracy
    RESULT_ACCURACY.append(accuracy)
