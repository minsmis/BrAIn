import random
import numpy as np
import functions as fc

from sklearn.model_selection import KFold


def import_dataset(list_trial_label, list_data_path, num_sampling_frequency, num_time_bin_ms):
    # Variables
    dict_data = {}

    for label, path in zip(list_trial_label, list_data_path):
        bool_temp_interaction, ndarr_temp_calcium = fc.import_data(path)
        dict_data[label] = fc.extract_interaction(bool_temp_interaction, ndarr_temp_calcium,
                                                  num_sampling_frequency, num_time_bin_ms, align=False,
                                                  shuffle=True, do_average=True)
    return dict_data


def split_dataset(dict_data, num_train_ratio=0.75, **kwargs):
    # Parameters
    # 'num_train_ratio' [int, Default = 0.75]: The ratio for train dataset size.
    # 'k_fold' [int, Default = 20]: k of the k-fold cross validation.
    # If 'k_fold' was given, 'num_train_ratio' will be ignored during the test code.
    # 'k_fold' must be same to EXPERIMENT_EPOCHS in the test code.
    # 'current_epoch' [int, Must be paired with 'k_fold' parameter]: The current epoch for k-fold cross validation.
    # 'current_epoch' must be given together with 'k_fold' parameter.

    def select_current_cross_validation_datasets(list_data, meta_kFold, num_epoch):
        for i, (list_train_idx, list_test_idx) in enumerate(meta_kFold.split(list_data)):
            if i == num_epoch:
                return np.array(list_data)[list_train_idx], np.array(list_data)[list_test_idx]

    # Mode selection
    STR_SPLIT_MODE = 'train_ratio'  # Default mode is 'train_ratio' method.
    if 'k_fold' and 'current_epoch' in kwargs:
        STR_SPLIT_MODE = 'k_fold'
    if 'k_fold' in kwargs and 'current_epoch' not in kwargs:  # Throwing mode selection errors
        STR_SPLIT_MODE = 'train_ratio'
        print("Check missed parameters: 'current_epoch'")
    if 'current_epoch' in kwargs and 'k_fold' not in kwargs:  # Throwing mode selection errors
        STR_SPLIT_MODE = 'train_ratio'
        print("Check missed parameters: 'k_fold'")

    # Output
    dict_train_data = {}
    dict_test_data = {}

    # Split datasets by k-fold cross validation
    if STR_SPLIT_MODE == 'k_fold':
        num_k = kwargs.get('k_fold')
        num_current_epoch = kwargs.get('current_epoch')
        k_fold_cross_validation = KFold(n_splits=num_k)
        if STR_SPLIT_MODE == 'k_fold' and isinstance(num_k, int) and isinstance(num_current_epoch, int):
            for key in dict_data.keys():
                list_temp_data_handle = dict_data[key]
                dict_train_data[key], dict_test_data[key] = select_current_cross_validation_datasets(
                    list_temp_data_handle, k_fold_cross_validation, num_current_epoch)
            return dict_train_data, dict_test_data

    # Split datasets by train ratio
    if STR_SPLIT_MODE == 'train_ratio':
        for key in dict_data.keys():
            list_temp_data_handle = dict_data[key]
            TRAIN_LENGTH = int(len(list_temp_data_handle) * num_train_ratio)
            dict_train_data[key] = list_temp_data_handle[0:TRAIN_LENGTH]
            dict_test_data[key] = list_temp_data_handle[TRAIN_LENGTH:]
        return dict_train_data, dict_test_data


def feature_selection(dict_data, num_iter=2, num_q=5):
    # Parameters
    # 'num_iter' [int, Default = 2]: Iteration time to extract features.
    # 'num_q' [int, Default = 5]: Number of features to extract from each cell.
    # 'align' [bool, Default = False]: Align T = 2qn long vectors to 1D array (n is number of neurons).

    def select_feature_from_random_bout(list_data, num_cell, ndarr_random_bout):
        list_output_features = []
        for bout in ndarr_random_bout:
            list_output_features.append(list_data[bout][num_cell])
        return list_output_features

    # Check validity
    if not isinstance(dict_data, dict):
        print("Given data is not 'dict'. Check the data type.")
        return 0

    # Output
    dict_data_features = {}

    # Feature selection
    for key in dict_data.keys():  # dict_data Keys
        list_temp_data_handle = dict_data[key]  # Data to handle
        list_temp_data_output = []  # Result of feature selection from data handle
        for iteration in range(0, num_iter):  # iteration
            for cell_index in range(0, len(list_temp_data_handle[0])):  # Cell number
                # Pick random bout of interaction for num_q times
                ndarr_random_selection = np.random.randint(0, len(list_temp_data_handle), size=num_q)
                # Select features
                list_temp_features = select_feature_from_random_bout(list_temp_data_handle, cell_index,
                                                                     ndarr_random_selection)
                # Merge to output data
                list_temp_data_output.append(list_temp_features)
        # Store data to output dictionary with same label keys
        dict_data_features[key] = list_temp_data_output

    # Align datasets
    for key in dict_data_features.keys():
        list_temp_data_handle = dict_data_features[key]
        dict_data_features[key] = np.reshape(list_temp_data_handle, (1, -1))
    return dict_data_features


def make_class(dict_data, list_class_organization):
    dict_class = {}

    for idx, org in enumerate(list_class_organization):
        temp = []
        for key in org:
            temp.extend(dict_data[key])
        dict_class[idx] = temp
    return dict_class


def make_null_dataset(list_train_dataset, list_train_label):
    random.shuffle(list_train_dataset)
    random.shuffle(list_train_label)
    return list_train_dataset, list_train_label


def make_dataset(dict_class, bool_shuffle=False, bool_null_mode=False):
    # NOTICE: This is for binary classes.
    # NULL_MODE: If it was TRUE, Train_dataset and Train_label are randomly shuffled each other.
    # However, Test_dataset and Test_label are not shuffled.

    # Variables
    list_data = []
    list_label = []

    list_class_A = dict_class[0]  # Class 0
    list_class_B = dict_class[1]  # Class 1

    while len(list_class_A) >= 2 and len(list_class_B) >= 2:
        num_random = random.randint(1, 4)  # Get random index

        if num_random == 1:  # Label = 0
            list_data.append(np.append(list_class_A.pop(), list_class_A.pop()))
            list_label.append(0)
        if num_random == 2:  # Label = 1
            list_data.append(np.append(list_class_A.pop(), list_class_B.pop()))
            list_label.append(1)
        if num_random == 3:  # Label = 1
            list_data.append(np.append(list_class_B.pop(), list_class_A.pop()))
            list_label.append(1)
        if num_random == 4:  # Label = 0
            list_data.append(np.append(list_class_B.pop(), list_class_B.pop()))
            list_label.append(0)

    # Random shuffle train dataset
    if bool_shuffle is True:
        MIX = list(zip(list_data, list_label))
        random.shuffle(MIX)
        list_data, list_label = zip(*MIX)
        list_data, list_label = list(list_data), list(list_label)

    # Make NULL dataset
    if bool_null_mode is True:
        list_data, list_label = make_null_dataset(list_data, list_label)

    return np.array(list_data), np.array(list_label)


def make_dataset2(dict_class, bool_shuffle=False, bool_null_mode=False):
    # NOTICE: This is for binary classes.
    # NULL_MODE: If it was TRUE, Train_dataset and Train_label are randomly shuffled each other.
    # However, Test_dataset and Test_label are not shuffled.

    # Variables
    list_class_A = dict_class[0]  # Class 0
    list_class_B = dict_class[1]  # Class 1

    list_data = np.concatenate([list_class_A, list_class_B], axis=0)
    list_label = np.concatenate([np.zeros(len(list_class_A)), np.ones(len(list_class_B))], axis=0)

    # Random shuffle train dataset
    if bool_shuffle is True:
        MIX = list(zip(list_data, list_label))
        random.shuffle(MIX)
        list_data, list_label = zip(*MIX)
        list_data, list_label = list(list_data), list(list_label)

    # Make NULL dataset
    if bool_null_mode is True:
        list_data, list_label = make_null_dataset(list_data, list_label)

    return np.array(list_data), np.array(list_label)
