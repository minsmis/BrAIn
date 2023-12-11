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
NULL_MODE = True  # True: Test NULL model, False: Test genuine model
fs = 20  # Hz
time_bin = 100  # ms
class_organization = [
    ['A', 'C'],  # Class 0
    ['B', 'D']  # Class 1
]
train_ratio = 0.75

# Import data
data = fc.import_dataset(trial_label, data_path, fs, time_bin)

# Make class
classes = fc.make_class(data, class_organization)

# Make train dataset
train_dataset, train_label, test_dataset, test_label = fc.make_dataset(classes, train_ratio, bool_null_mode=NULL_MODE)

# Train model
model = mdl.train_model_xor(train_dataset, train_label)

# Test model
accuracy = mdl.test_model(model, test_dataset, test_label, percent=False)
print('ACCURACY: {}%'.format(accuracy))
