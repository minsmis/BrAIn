import os

import joblib
import numpy as np
from sklearn import svm  # import sklearn svm package


def report_accuracy(num_accuracy, bool_percent):
    if bool_percent is True:
        template = "\n-----\nACCURACY: {}%\n-----\n"
        print(template.format(num_accuracy))
    if bool_percent is False:
        template = "\n-----\nACCURACY: {}\n-----\n"
        print(template.format(num_accuracy))


def report_performance(list_accuracy, bool_percent):
    # This function reports averaged performances for all EXPERIMENT_EPOCHS
    num_performance = np.average(list_accuracy)
    if bool_percent is True:
        template = "\n-----\nPERFORMANCE: {}%\n-----\n"
        print(template.format(num_performance))
    if bool_percent is False:
        template = "\n-----\nPERFORMANCE: {}\n-----\n"
        print(template.format(num_performance))


def train_model_xor(train_data, train_label, bool_verbose=False, **kwargs):
    # kwargs
    # 'storage' [str/None (Default)]: Save trained model into given directory.
    # 'load' [str/None (Default)]: Load model stored before.

    # Define model
    model = svm.SVC(verbose=bool_verbose)

    # Load trained model if 'load' path was given.
    if 'load' in kwargs:
        str_load = kwargs.get('load')
        if isinstance(str_load, str):
            model = joblib.load(str_load)
        if not isinstance(str_load, str):
            print("Check keyword arguments: 'load'")

    # Train model
    model = model.fit(train_data, train_label)

    # Save trained model
    if 'storage' in kwargs:
        str_storage = kwargs.get('storage')
        if isinstance(str_storage, str):
            joblib.dump(model, os.path.join(str_storage, 'trained_XOR.pkl'))
        if not isinstance(str_storage, str):
            print("Check keyword arguments: 'storage'")

    return model


def test_model(trained_model, test_data, test_label, **kwargs):
    # kwargs
    # 'percent' [True (Default)/False]: Calculate accuracy.

    # Variables
    num_accuracy = 0
    num_correct = 0
    bool_percent = False

    # Predict test_data
    prediction = trained_model.predict(test_data)

    # Compare prediction with test_label
    for predict, label in zip(prediction, test_label):
        if predict == label:
            num_correct += 1

    # Calculate accuracy
    num_total = len(prediction)
    if 'percent' in kwargs:
        bool_percent = kwargs.get('percent')
        if isinstance(bool_percent, bool):
            if bool_percent is True:
                num_accuracy = 100 * (num_correct / num_total)
            if bool_percent is False:
                num_accuracy = num_correct / num_total
        if not isinstance(bool_percent, bool):
            print("Check keyword arguments: 'percent'")
            num_accuracy = 100 * (num_correct / num_total)
    if 'percent' not in kwargs:
        num_accuracy = 100 * (num_correct / num_total)

    # Report accuracy
    report_accuracy(num_accuracy, bool_percent)

    return num_accuracy
