import mat73 as mat
import numpy as np
import random


# This function call bool_interaction data and num_calcium data from GP Datastorage
def import_data(data_path):
    INTERACTION_COLUMN_IDX = 2  # nd column (2D in matrix)
    CALCIUM_COLUMN_START_INDEX = 6  # th column (2D in matrix)

    mat_data = mat.loadmat(data_path)
    bool_interaction = mat_data['Datastorage'][:, INTERACTION_COLUMN_IDX - 1]
    ndarr_calcium = mat_data['Datastorage'][:, (CALCIUM_COLUMN_START_INDEX - 1):]
    return bool_interaction, ndarr_calcium


def extract_interaction(bool_interaction, ndarr_calcium, num_sampling_frequency, num_time_bin_ms, **kwargs):
    # **kwargs
    # 'align' [Ture/False (Default)]: Reshape Epoch x Cell x Bin {3D} to (Epoch x Cell) x Bin {2D}
    # 'shuffle' [True/False (Default)]: Shuffle epochs randomly
    # 'do_average' [True/False (Default)]: Average calcium transient within bin.

    def is_all_equal(array):
        return all(element == array[0] for element in array)

    # Variables
    BOOL_DO_AVERAGE = False

    # Get averaging mode
    if 'do_average' in kwargs:
        BOOL_DO_AVERAGE = kwargs.get('do_average')
        if not isinstance(BOOL_DO_AVERAGE, bool):
            BOOL_DO_AVERAGE = False
            print("Check keyword arguments: 'do_average'")

    # Output
    list_interaction_calcium_dataset = []
    num_excluded_blocks = 0

    # BIN_LENGTH_TO_INDEX indices need to make time_bin_ms bin.
    BIN_LENGTH_TO_INDEX = int(num_time_bin_ms / (1000 / num_sampling_frequency))

    for i in np.arange(0, len(bool_interaction), BIN_LENGTH_TO_INDEX):
        bool_temp_interaction_block = bool_interaction[i:i + BIN_LENGTH_TO_INDEX]
        ndarr_temp_calcium_block = ndarr_calcium[i:i + BIN_LENGTH_TO_INDEX, :]
        if BOOL_DO_AVERAGE is True:
            ndarr_temp_calcium_block = np.average(ndarr_temp_calcium_block, axis=0)  # Average bin

        if is_all_equal(bool_temp_interaction_block) is True:
            if bool_temp_interaction_block[0] == 1:  # When interaction
                # 1D: Interaction bout, 2D: Cell, 3D: Bin OR 1D: Interaction bout, 2D: Mean event of cell
                list_interaction_calcium_dataset.append(
                    np.transpose(ndarr_temp_calcium_block))

        if is_all_equal(bool_temp_interaction_block) is False:
            num_excluded_blocks += 1  # Increase # of excluded blocks

    # Reshape Epoch x Cell x Bin {3D} to (Epoch x Cell) x Bin {2D}
    if 'align' in kwargs:
        bool_align = kwargs.get('align')
        if isinstance(bool_align, bool):
            if bool_align is True:
                list_interaction_calcium_dataset = list(
                    np.reshape(list_interaction_calcium_dataset, (-1, BIN_LENGTH_TO_INDEX)))
        if not isinstance(bool_align, bool):
            print("Check keyword arguments: 'align'")

    # Shuffle epochs randomly
    if 'shuffle' in kwargs:
        bool_shuffle = kwargs.get('shuffle')
        if isinstance(bool_shuffle, bool):
            if bool_shuffle is True:
                random.shuffle(list_interaction_calcium_dataset)
        if not isinstance(bool_shuffle, bool):
            print("Check keyword arguments: 'shuffle'")

    return list_interaction_calcium_dataset
