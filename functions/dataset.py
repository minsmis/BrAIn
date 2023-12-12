import random
import numpy as np
import functions.importer as fimport


def import_dataset(list_trial_label, list_data_path, num_sampling_frequency, num_time_bin_ms):
    dict_data = {}

    for label, path in zip(list_trial_label, list_data_path):
        bool_temp_interaction, ndarr_temp_calcium = fimport.import_data(path)
        dict_data[label] = fimport.extract_interaction(bool_temp_interaction, ndarr_temp_calcium,
                                                       num_sampling_frequency, num_time_bin_ms, align=True,
                                                       shuffle=True)
    return dict_data


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


def make_dataset(dict_class, train_ratio, bool_null_mode=False):
    # NOTICE: This is for binary classes.
    # NULL_MODE: If it was TRUE, Train_dataset and Train_label are randomly shuffled each other.
    # However, Test_dataset and Test_label are not shuffled.

    # Variables
    list_train_test = []
    list_train_test_label = []

    list_class_A = dict_class[0]  # Class 0
    list_class_B = dict_class[1]  # Class 1

    while len(list_class_A) >= 2 and len(list_class_B) >= 2:
        num_random = random.randint(1, 4)  # Get random index

        if num_random == 1:  # Label = 0
            list_train_test.append(np.append(list_class_A.pop(), list_class_A.pop()))
            list_train_test_label.append(0)
        if num_random == 2:  # Label = 1
            list_train_test.append(np.append(list_class_A.pop(), list_class_B.pop()))
            list_train_test_label.append(1)
        if num_random == 3:  # Label = 1
            list_train_test.append(np.append(list_class_B.pop(), list_class_A.pop()))
            list_train_test_label.append(1)
        if num_random == 4:  # Label = 0
            list_train_test.append(np.append(list_class_B.pop(), list_class_B.pop()))
            list_train_test_label.append(0)

    # Divide train and test dataset
    TRAIN_LENGTH = int(len(list_train_test) * train_ratio)
    list_train = list_train_test[0:TRAIN_LENGTH]
    list_train_label = list_train_test_label[0:TRAIN_LENGTH]
    list_test = list_train_test[TRAIN_LENGTH:]
    list_test_label = list_train_test_label[TRAIN_LENGTH:]

    # Random shuffle train dataset
    MIX = list(zip(list_train, list_train_label))
    random.shuffle(MIX)
    list_train, list_train_label = zip(*MIX)
    list_train, list_train_label = list(list_train), list(list_train_label)

    # Make NULL dataset
    if bool_null_mode is True:
        list_train, list_train_label = make_null_dataset(list_train, list_train_label)

    return list_train, list_train_label, list_test, list_test_label
