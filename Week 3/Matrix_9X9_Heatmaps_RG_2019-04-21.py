import random
from math import exp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

num_input_nodes = 81
num_hidden_nodes = 10
num_output_nodes = 9
array_size_list = (num_input_nodes, num_hidden_nodes, num_output_nodes)


# Compute neuron activation using sigmoid transfer function
def compute_transfer_function(summed_neuron_input, alpha):
    activation = 1.0 / (1.0 + exp(-alpha * summed_neuron_input))
    return activation


# ------------------------------------------------------#

# Compute derivative of transfer function
def compute_transfer_function_derivative(neuron_output, alpha):
    return alpha * neuron_output * (1.0 - neuron_output)


# ------------------------------------------------------#
def matrix_dot_product(matrix1, matrix2):
    dot_product = np.dot(matrix1, matrix2)

    return dot_product


####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################

def initialize_weight():
    random_num = random.random()
    weight = 1 - 2 * random_num
    #    print weight

    return weight


####################################################################################################
####################################################################################################
#
# Function to initialize the node-to-node connection weight arrays
#
####################################################################################################
####################################################################################################

def initialize_weight_array(weight_array_size_list):
    # This procedure is also called directly from 'main.'
    #
    # This procedure takes in the two parameters, the number of nodes on the bottom (of any two layers),
    #   and the number of nodes in the layer just above it.
    #   It will use these two sizes to create a weight array.
    # The weights will initially be assigned random values here, and
    #   this array is passed back to the 'main' procedure.

    num_lower_nodes = weight_array_size_list[0]
    num_upper_nodes = weight_array_size_list[1]

    #    print ' '
    #    print ' inside procedure initialize_weight_array'
    #    print ' the number of lower nodes is', numLowerNodes
    #    print ' the number of upper nodes is', numUpperNodes
    #
    # Initialize the weight variables with random weights
    weight_array = np.zeros((num_upper_nodes, num_lower_nodes))  # iniitalize the weight matrix with 0's
    for row in range(num_upper_nodes):  # Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(num_lower_nodes):  # number of columns in matrix 2
            weight_array[row, col] = initialize_weight()

    #    print weightArray

    # We return the array to the calling procedure, 'main'.
    return weight_array


####################################################################################################
####################################################################################################
#
# Function to initialize the bias weight arrays
#
####################################################################################################
####################################################################################################

def initialize_bias_weight_array(num_bias_nodes):
    # This procedure is also called directly from 'main.'

    # Initialize the bias weight variables with random weights
    bias_weight_array = np.zeros(num_bias_nodes)  # iniitalize the weight matrix with 0's
    for node in range(num_bias_nodes):  # Number of nodes in bias weight set
        bias_weight_array[node] = initialize_weight()

    # Print the entire weights array.
    #    print biasWeightArray

    # We return the array to the calling procedure, 'main'.
    return bias_weight_array


####################################################################################################
####################################################################################################
#
# Function to return a trainingDataList
#
####################################################################################################
####################################################################################################

def obtain_selected_alphabet_training_values(data_set):
    # Note: Nine possible output classes: 0 .. 8 trainingDataListXX [4]
    training_data_list_a0 = (1,
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                              0,
                              0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 1, 'A', 0,
                             'A')  # training data list 1 selected for the letter 'A'
    training_data_list_b0 = (2,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 2, 'B', 1,
                             'B')  # training data list 2, letter 'E', courtesy AJM
    training_data_list_c0 = (3,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 'C', 2,
                             'C')  # training data list 3, letter 'C', courtesy PKVR
    training_data_list_d0 = (4,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 4, 'D', 3,
                             'O')  # training data list 4, letter 'D', courtesy TD
    training_data_list_e0 = (5,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 5, 'E', 4,
                             'E')  # training data list 5, letter 'E', courtesy BMcD
    training_data_list_f0 = (6,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 6, 'F', 4,
                             'E')  # training data list 6, letter 'F', courtesy SK
    training_data_list_g0 = (7,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 7, 'G', 1, 'C')
    training_data_list_h0 = (8,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 8, 'H', 0,
                             'A')  # training data list 8, letter 'H', courtesy JC
    training_data_list_i0 = (9,
                             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 9, 'I', 5,
                             'I')  # training data list 9, letter 'I', courtesy GR
    training_data_list_j0 = (10,
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                              0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                              1,
                              0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 10, 'J', 5,
                             'I')  # training data list 10 selected for the letter 'L', courtesy JT
    training_data_list_l0 = (12,
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 12, 'L', 7,
                             'L')  # training data list 12 selected for the letter 'L', courtesy PV
    training_data_list_m0 = (13,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                              0,
                              0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                              0,
                              1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], 13, 'M', 8,
                             'M')  # training data list 13 selected for the letter 'M', courtesy GR
    training_data_list_n0 = (14,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              1,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 14, 'N', 8,
                             'M')  # training data list 14 selected for the letter 'N'
    training_data_list_o0 = (15,
                             [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 15, 'O', 3,
                             'O')  # training data list 15, letter 'O', courtesy TD
    training_data_list_p0 = (16,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 16, 'P', 1,
                             'B')  # training data list 16, letter 'P', courtesy MT

    if data_set == 1:
        return training_data_list_a0
    elif data_set == 2:
        return training_data_list_b0
    elif data_set == 3:
        return training_data_list_c0
    elif data_set == 4:
        return training_data_list_d0
    elif data_set == 5:
        return training_data_list_e0
    elif data_set == 6:
        return training_data_list_f0
    elif data_set == 7:
        return training_data_list_g0
    elif data_set == 8:
        return training_data_list_h0
    elif data_set == 9:
        return training_data_list_i0
    elif data_set == 10:
        return training_data_list_j0
    elif data_set == 11:
        return training_data_list_j0
    elif data_set == 12:
        return training_data_list_l0
    elif data_set == 13:
        return training_data_list_m0
    elif data_set == 14:
        return training_data_list_n0
    elif data_set == 15:
        return training_data_list_o0
    elif data_set == 16:
        return training_data_list_p0


####################################################################################################
####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################
####################################################################################################

def obtain_random_alphabet_training_values(num_training_data_sets):
    data_set = random.randint(0, num_training_data_sets)

    training_data_list_a0 = (1,
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                              0,
                              0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 1, 'A', 1,
                             'A')  # training data list 1 selected for the letter 'A'
    training_data_list_b0 = (2,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 2, 'B', 2,
                             'B')  # training data list 2, letter 'E', courtesy AJM
    training_data_list_c0 = (3,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 'C', 3,
                             'C')  # training data list 3, letter 'C', courtesy PKVR
    training_data_list_d0 = (4,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 4, 'D', 4,
                             'O')  # training data list 4, letter 'D', courtesy TD
    training_data_list_e0 = (5,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 5, 'E', 5,
                             'E')  # training data list 5, letter 'E', courtesy BMcD
    training_data_list_f0 = (6,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 6, 'F', 5,
                             'E')  # training data list 6, letter 'F', courtesy SK
    training_data_list_g0 = (7,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 7, 'G', 3, 'C')
    training_data_list_h0 = (8,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 8, 'H', 1,
                             'A')  # training data list 8, letter 'H', courtesy JC
    training_data_list_i0 = (9,
                             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 9, 'I', 6,
                             'I')  # training data list 9, letter 'I', courtesy GR
    training_data_list_j0 = (10,
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                              0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                              1,
                              0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 10, 'J', 6,
                             'I')  # training data list 10 selected for the letter 'L', courtesy JT
    training_data_list_l0 = (12,
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 12, 'L', 8,
                             'L')  # training data list 12 selected for the letter 'L', courtesy PV
    training_data_list_m0 = (13,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                              0,
                              0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                              0,
                              1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], 13, 'M', 9,
                             'M')  # training data list 13 selected for the letter 'M', courtesy GR
    training_data_list_n0 = (14,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              1,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 14, 'N', 9,
                             'M')  # training data list 14 selected for the letter 'N'
    training_data_list_o0 = (15,
                             [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 15, 'O', 4,
                             'O')  # training data list 15, letter 'O', courtesy TD
    training_data_list_p0 = (16,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0,
                              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                              0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 16, 'P', 2,
                             'P')  # training data list 16, letter 'P', courtesy MT

    if data_set == 1:
        return training_data_list_a0
    elif data_set == 2:
        return training_data_list_b0
    elif data_set == 3:
        return training_data_list_c0
    elif data_set == 4:
        return training_data_list_d0
    elif data_set == 5:
        return training_data_list_e0
    elif data_set == 6:
        return training_data_list_f0
    elif data_set == 7:
        return training_data_list_g0
    elif data_set == 8:
        return training_data_list_h0
    elif data_set == 9:
        return training_data_list_i0
    elif data_set == 10:
        return training_data_list_j0
    elif data_set == 11:
        return training_data_list_j0
    elif data_set == 12:
        return training_data_list_l0
    elif data_set == 13:
        return training_data_list_m0
    elif data_set == 14:
        return training_data_list_n0
    elif data_set == 15:
        return training_data_list_o0
    elif data_set == 16:
        return training_data_list_p0


def compute_single_feedforward_pass_first_step(alpha, input_data_list, w_weight_array, bias_hidden_weight_array,
                                               hidden_array_length):
    hidden_array = np.zeros(hidden_array_length)

    sum_into_hidden_array = matrix_dot_product(w_weight_array, input_data_list)

    for node in range(hidden_array_length):  # Number of hidden nodes
        hidden_node_sum_input = sum_into_hidden_array[node] + bias_hidden_weight_array[node]
        hidden_array[node] = compute_transfer_function(hidden_node_sum_input, alpha)

    return hidden_array


def compute_single_feedforward_pass_second_step(alpha, hidden_array, v_weight_array, bias_output_weight_array,
                                                output_array_length):
    output_array = np.zeros(output_array_length)

    sum_into_output_array = matrix_dot_product(v_weight_array, hidden_array)

    for node in range(output_array_length):
        output_node_sum_input = sum_into_output_array[node] + bias_output_weight_array[node]
        output_array[node] = compute_transfer_function(output_node_sum_input, alpha)

    return output_array


def compute_outputs_across_all_training_data(alpha, num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array,
                                             input_array_length, output_array_length, hidden_array_length):
    selected_training_data_set = 1
    hidden_node_activations_list = []
    output_node_activations_list = []

    while selected_training_data_set < num_training_data_sets + 1:
        print()
        print(" the selected Training Data Set is ", selected_training_data_set)
        training_data_list = obtain_selected_alphabet_training_values(selected_training_data_set)

        training_data_input_list = training_data_list[1]

        input_data_list = []
        for node in range(input_array_length):
            training_data = training_data_input_list[node]
            input_data_list.append(training_data)

        letter_num = training_data_list[2]
        letter_char = training_data_list[3]
        print()
        print("  Data Set Number", selected_training_data_set, " for letter ", letter_char, " with letter number ",
              letter_num)

        hidden_array = compute_single_feedforward_pass_first_step(alpha, input_data_list, w_weight_array,
                                                                  bias_hidden_weight_array, hidden_array_length)

        print()
        print(" The hidden node activations are: ")
        print(hidden_array)

        output_array = compute_single_feedforward_pass_second_step(alpha, hidden_array, v_weight_array,
                                                                   bias_output_weight_array, output_array_length)

        print()
        print(" The output node activations are: ")
        print(output_array)

        hidden_node_activations_list.append(
            [letter_char, letter_num, training_data_list[4], '\n'.join([letter_char, training_data_list[5]]),
             hidden_array])
        output_node_activations_list.append(
            [letter_char, letter_num, training_data_list[4], '\n'.join([letter_char, training_data_list[5]]),
             output_array])

        desired_output_array = np.zeros(output_array_length)  # iniitalize the output array with 0's
        desired_class = training_data_list[4]  # identify the desired class
        desired_output_array[desired_class] = 1  # set the desired output for that class to 1

        print()
        print(" The desired output array values are: ")
        print(desired_output_array)

        error_array = np.zeros(output_array_length)

        new_sse = 0.0
        for node in range(output_array_length):  # Number of nodes in output set (classes)
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]

        print()
        print(" ' The error values are:")
        print(error_array)

        # Print the Summed Squared Error
        print("New SSE = %.6f" % new_sse)

        selected_training_data_set = selected_training_data_set + 1

    print("\n\nhiddenNodeActivationsList L-{letter} C-{class}")
    print("==============================================")
    print("==============================================\n")
    df = pd.DataFrame([lst[4] for lst in hidden_node_activations_list])
    df = df.transpose()
    df.columns = (x[3] for x in hidden_node_activations_list)
    df.columns = 'L-' + df.columns.str[0:1] + ' C-' + df.columns.str[2:3]
    print(df)

    # hidden nodes heatmap
    print("\n\n\n\n\n")
    sns.set(rc={'figure.figsize': (10.0, 4.0)})
    ax = sns.heatmap(df, cmap='YlOrRd', annot=False, linewidths=0.1, linecolor='white')
    ax.set_title('Hidden Node Activations\nL-{letter} C-{class}')
    ax.set_ylabel('Node')
    ax.set_xlabel('L-{Letter} C-{Class}')
    plt.show()

    # hidden nodes clustermap with row and column dendrogram
    print("\n\n\n\n\n")
    ax = sns.clustermap(df, cmap='YlOrRd', annot=False, linewidths=0.1, linecolor='white')
    ax.fig.suptitle('Hidden Node Activations\nClustered by Node and Letter\nL-{letter} C-{class}')
    plt.show()

    # hidden nodes clustermap with column dendrogram
    print("\n\n\n\n\n")
    ax = sns.clustermap(df, cmap='YlOrRd', annot=False, linewidths=0.1, linecolor='white', row_cluster=False)
    ax.fig.suptitle('Hidden Node Activations\nClustered Letter\nL-{letter} C-{class}')
    plt.show()


def backpropagate_output_to_hidden(alpha, eta, array_size_list, error_array, output_array, hidden_array,
                                   v_weight_array):
    hidden_array_length = array_size_list[1]
    output_array_length = array_size_list[2]

    transfer_func_deriv_array = np.zeros(output_array_length)

    for node in range(output_array_length):
        transfer_func_deriv_array[node] = compute_transfer_function_derivative(output_array[node], alpha)

    delta_v_wt_array = np.zeros((output_array_length, hidden_array_length))
    new_v_weight_array = np.zeros(
        (output_array_length, hidden_array_length))

    for row in range(output_array_length):
        for col in range(hidden_array_length):
            partial_sse_w_v_wt = -error_array[row] * transfer_func_deriv_array[row] * hidden_array[col]
            delta_v_wt_array[row, col] = -eta * partial_sse_w_v_wt
            new_v_weight_array[row, col] = v_weight_array[row, col] + delta_v_wt_array[row, col]

    return new_v_weight_array


def backpropagate_bias_output_weights(alpha, eta, array_size_list, error_array, output_array, bias_output_weight_array):
    output_array_length = array_size_list[2]

    delta_bias_output_array = np.zeros(output_array_length)
    new_bias_output_weight_array = np.zeros(output_array_length)
    transfer_func_deriv_array = np.zeros(output_array_length)

    for node in range(output_array_length):
        transfer_func_deriv_array[node] = compute_transfer_function_derivative(output_array[node], alpha)

    for node in range(output_array_length):
        partial_sse_w_bias_output = -error_array[node] * transfer_func_deriv_array[node]
        delta_bias_output_array[node] = -eta * partial_sse_w_bias_output
        new_bias_output_weight_array[node] = bias_output_weight_array[node] + delta_bias_output_array[node]

    return new_bias_output_weight_array


def backpropagate_hidden_to_input(alpha, eta, array_size_list, error_array, output_array, hidden_array,
                                  input_array, v_weight_array, w_weight_array, bias_hidden_weight_array,
                                  bias_output_weight_array):
    input_array_length = array_size_list[0]
    hidden_array_length = array_size_list[1]
    output_array_length = array_size_list[2]

    transfer_func_deriv_hidden_array = np.zeros(
        hidden_array_length)

    for node in range(hidden_array_length):
        transfer_func_deriv_hidden_array[node] = compute_transfer_function_derivative(hidden_array[node], alpha)

    error_times_t_func_deriv_output_array = np.zeros(output_array_length)
    transfer_func_deriv_output_array = np.zeros(output_array_length)
    weighted_error_array = np.zeros(hidden_array_length)

    for outputNode in range(output_array_length):
        transfer_func_deriv_output_array[outputNode] = compute_transfer_function_derivative(output_array[outputNode],
                                                                                            alpha)
        error_times_t_func_deriv_output_array[outputNode] = error_array[outputNode] * transfer_func_deriv_output_array[
            outputNode]

    for hiddenNode in range(hidden_array_length):
        weighted_error_array[hiddenNode] = 0
        for outputNode in range(output_array_length):
            weighted_error_array[hiddenNode] = weighted_error_array[hiddenNode] \
                                               + v_weight_array[outputNode, hiddenNode] * \
                                               error_times_t_func_deriv_output_array[
                                                   outputNode]

    delta_w_wt_array = np.zeros((hidden_array_length, input_array_length))
    new_w_weight_array = np.zeros(
        (hidden_array_length, input_array_length))

    for row in range(hidden_array_length):

        for col in range(input_array_length):
            partial_sse_w_w_wts = -transfer_func_deriv_hidden_array[row] * input_array[col] * weighted_error_array[row]
            delta_w_wt_array[row, col] = -eta * partial_sse_w_w_wts
            new_w_weight_array[row, col] = w_weight_array[row, col] + delta_w_wt_array[row, col]

    return new_w_weight_array


def backpropagate_bias_hidden_weights(alpha, eta, array_size_list, error_array, output_array, hidden_array,
                                      input_array, v_weight_array, w_weight_array, bias_hidden_weight_array,
                                      bias_output_weight_array):
    input_array_length = array_size_list[0]
    hidden_array_length = array_size_list[1]
    output_array_length = array_size_list[2]

    error_times_t_func_deriv_output_array = np.zeros(output_array_length)
    transfer_func_deriv_output_array = np.zeros(output_array_length)
    weighted_error_array = np.zeros(hidden_array_length)

    transfer_func_deriv_hidden_array = np.zeros(
        hidden_array_length)
    partial_sse_w_bias_hidden = np.zeros(hidden_array_length)
    delta_bias_hidden_array = np.zeros(hidden_array_length)
    new_bias_hidden_weight_array = np.zeros(hidden_array_length)

    for node in range(hidden_array_length):
        transfer_func_deriv_hidden_array[node] = compute_transfer_function_derivative(hidden_array[node], alpha)

    for outputNode in range(output_array_length):
        transfer_func_deriv_output_array[outputNode] = compute_transfer_function_derivative(output_array[outputNode],
                                                                                            alpha)
        error_times_t_func_deriv_output_array[outputNode] = error_array[outputNode] * transfer_func_deriv_output_array[
            outputNode]

    for hiddenNode in range(hidden_array_length):
        weighted_error_array[hiddenNode] = 0
        for outputNode in range(output_array_length):
            weighted_error_array[hiddenNode] = (weighted_error_array[hiddenNode]
                                                + v_weight_array[outputNode, hiddenNode] *
                                                error_times_t_func_deriv_output_array[
                                                    outputNode])

    for hiddenNode in range(hidden_array_length):
        partial_sse_w_bias_hidden[hiddenNode] = -transfer_func_deriv_hidden_array[hiddenNode] * weighted_error_array[
            hiddenNode]
        delta_bias_hidden_array[hiddenNode] = -eta * partial_sse_w_bias_hidden[hiddenNode]
        new_bias_hidden_weight_array[hiddenNode] = bias_hidden_weight_array[hiddenNode] + delta_bias_hidden_array[
            hiddenNode]

    return new_bias_hidden_weight_array


def print_letter(training_data_list):
    pixel_array = training_data_list[1]
    print(' ')
    grid_width = 9
    grid_height = 9
    iter_across_row = 0
    iter_over_all_rows = 0
    while iter_over_all_rows < grid_height:
        while iter_across_row < grid_width:
            array_element = pixel_array[iter_across_row + iter_over_all_rows * grid_width]
            if array_element < 0.9:
                print_element = ' '
            else:
                print_element = 'X'
            print(print_element, end='')
            iter_across_row = iter_across_row + 1
        print(' ')
        iter_over_all_rows = iter_over_all_rows + 1
        iter_across_row = 0  # re-initialize so the row-print can begin again
    print('The data set is for the letter', training_data_list[3], ', which is alphabet number ', training_data_list[2])
    if training_data_list[0] > 25:
        print('This is a variant pattern for letter ', training_data_list[3])

    return


def read_in_files(file):
    array = []
    with open(file, encoding='UTF-8') as f:
        raw = f.readlines()
        for item in raw:
            # array.append((float(item[:-2])))
            array.append(float(item))
    return array


def main():
    random.seed(15)
    input_array_length = array_size_list[0]
    hidden_array_length = array_size_list[1]
    output_array_length = array_size_list[2]

    alpha = 1.0
    eta = 0.5
    max_num_iterations = 5000
    epsilon = 0.01
    iteration = 0
    num_training_data_sets = 16

    w_weight_array_size_list = (input_array_length, hidden_array_length)
    v_weight_array_size_list = (hidden_array_length, output_array_length)
    bias_hidden_weight_array_size = hidden_array_length
    bias_output_weight_array_size = output_array_length

    w_weight_file = 'C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1wWeightFile.txt'
    v_weight_file = 'C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1vWeightFile.txt'
    w_bias_weight_file = 'C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1wBiasWeightFile.txt'
    v_bias_weight_file = 'C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1vBiasWeightFile.txt'

    # w_weight_array = np.reshape(read_in_files(w_weight_file), (81,))
    # v_weight_array = np.reshape(read_in_files(v_weight_file), (81,))

    # bias_hidden_weight_array = read_in_files(w_bias_weight_file)  # w file
    # bias_output_weight_array = read_in_files(v_bias_weight_file)  # v file

    w_weight_array = initialize_weight_array(w_weight_array_size_list)
    v_weight_array = initialize_weight_array(v_weight_array_size_list)

    bias_hidden_weight_array = initialize_bias_weight_array(bias_hidden_weight_array_size)
    bias_output_weight_array = initialize_bias_weight_array(bias_output_weight_array_size)

    print()
    print("  Before training:")

    compute_outputs_across_all_training_data(alpha, num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array,
                                             input_array_length, output_array_length, hidden_array_length)

    while iteration < max_num_iterations:

        iteration = iteration + 1

        data_set = random.randint(1, num_training_data_sets)

        training_data_list = obtain_selected_alphabet_training_values(data_set)

        input_data_list = []
        input_data_array = np.zeros(input_array_length)

        this_training_data_list = training_data_list[1]
        for node in range(input_array_length):
            training_data = this_training_data_list[node]
            input_data_list.append(training_data)
            input_data_array[node] = training_data

        desired_output_array = np.zeros(output_array_length)
        desired_class = training_data_list[4]
        desired_output_array[desired_class] = 1

        hidden_array = compute_single_feedforward_pass_first_step(alpha, input_data_array, w_weight_array,
                                                                  bias_hidden_weight_array, hidden_array_length)

        output_array = compute_single_feedforward_pass_second_step(alpha, hidden_array, v_weight_array,
                                                                   bias_output_weight_array, output_array_length)

        error_array = np.zeros(output_array_length)

        new_sse = 0.0
        for node in range(output_array_length):  # Number of nodes in output set (classes)
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]

        new_v_weight_array = backpropagate_output_to_hidden(alpha, eta, array_size_list, error_array, output_array,
                                                            hidden_array,
                                                            v_weight_array)
        new_bias_output_weight_array = backpropagate_bias_output_weights(alpha, eta, array_size_list, error_array,
                                                                         output_array,
                                                                         bias_output_weight_array)

        new_w_weight_array = backpropagate_hidden_to_input(alpha, eta, array_size_list, error_array, output_array,
                                                           hidden_array,
                                                           input_data_list, v_weight_array, w_weight_array,
                                                           bias_hidden_weight_array,
                                                           bias_output_weight_array)

        new_bias_hidden_weight_array = backpropagate_bias_hidden_weights(alpha, eta, array_size_list, error_array,
                                                                         output_array,
                                                                         hidden_array,
                                                                         input_data_list, v_weight_array,
                                                                         w_weight_array,
                                                                         bias_hidden_weight_array,
                                                                         bias_output_weight_array)

        v_weight_array = new_v_weight_array[:]

        bias_output_weight_array = new_bias_output_weight_array[:]

        w_weight_array = new_w_weight_array[:]

        bias_hidden_weight_array = new_bias_hidden_weight_array[:]

        hidden_array = compute_single_feedforward_pass_first_step(alpha, input_data_list, w_weight_array,
                                                                  bias_hidden_weight_array, hidden_array_length)

        output_array = compute_single_feedforward_pass_second_step(alpha, hidden_array, v_weight_array,
                                                                   bias_output_weight_array, output_array_length)

        new_sse = 0.0
        for node in range(output_array_length):
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]

        if new_sse < epsilon:
            break
    print("Out of while loop at iteration ", iteration)
    print()
    print("  After training:")

    compute_outputs_across_all_training_data(alpha, num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array,
                                             input_array_length, output_array_length, hidden_array_length)

    w_weight_list = list()
    num_upper_nodes = hidden_array_length
    num_lower_nodes = input_array_length
    for row in range(num_upper_nodes):
        for col in range(num_lower_nodes):
            local_weight = w_weight_array[row, col]
            w_weight_list.append(local_weight)

    w_weight_file = open('C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1wWeightFile.txt', 'w')

    for item in w_weight_list:
        w_weight_file.write("%s\n" % item)
    w_weight_file.close()

    v_weight_list = list()
    num_upper_nodes = output_array_length
    num_lower_nodes = hidden_array_length
    for row in range(num_upper_nodes):
        for col in range(num_lower_nodes):  # number of columns in matrix 2
            local_weight = v_weight_array[row, col]
            v_weight_list.append(local_weight)

    v_weight_file = open('C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1vWeightFile.txt', 'w')

    for item in v_weight_list:
        v_weight_file.write("%s\n" % item)

    v_weight_file.close()

    w_bias_weight_list = list()
    for node in range(hidden_array_length):
        local_bias_weight = bias_hidden_weight_array[node]
        w_bias_weight_list.append(local_bias_weight)

    w_bias_weight_file = open('C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1wBiasWeightFile.txt',
                              'w')

    for item in w_bias_weight_list:
        w_bias_weight_file.write("%s\n" % item)
    w_bias_weight_file.close()

    v_bias_weight_list = list()
    for node in range(output_array_length):  # Number of output nodes
        local_bias_weight = bias_output_weight_array[node]
        v_bias_weight_list.append(local_bias_weight)

    v_bias_weight_file = open('C:\\Users\\Ivan\\Documents\\School\\MSDS_458\\Week 3\\datafiles\\GB1vBiasWeightFile.txt',
                              'w')

    for item in v_bias_weight_list:
        v_bias_weight_file.write("%s\n" % item)
    v_bias_weight_file.close()

    print(" Completed training and storing connection weights to files")


if __name__ == "__main__":
    main()
