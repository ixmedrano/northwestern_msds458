import numpy as np
import random
import math

num_input_nodes = 81
num_hidden_nodes = 6
num_output_nodes = 9
array_size_list = (num_input_nodes, num_hidden_nodes, num_output_nodes)
alpha = 1.0
eta = 0.5
max_num_iterations = 5000
epsilon = 0.01
sse = 0.0
num_training_data_sets = 16
input_array_length = array_size_list[0]
hidden_array_length = array_size_list[1]
output_array_length = array_size_list[2]
w_weight_array_size_list = (input_array_length, hidden_array_length)
v_weight_array_size_list = (hidden_array_length, output_array_length)
bias_hidden_weight_array_size = hidden_array_length
bias_output_weight_array_size = output_array_length


def initialize_weight_array(weight_array_size_list):
    num_lower_nodes = weight_array_size_list[0]
    num_upper_nodes = weight_array_size_list[1]

    weight_array = np.zeros((num_upper_nodes, num_lower_nodes))
    for row in range(num_upper_nodes):
        for col in range(num_lower_nodes):
            weight_array[row, col] = initialize_weight()
    return weight_array


def initialize_bias_weight_array(num_bias_nodes):
    bias_weight_array = np.zeros(num_bias_nodes)
    for node in range(num_bias_nodes):
        bias_weight_array[node] = initialize_weight()
    return bias_weight_array


def initialize_weight():
    random_num = random.random()
    weight = 1 - 2 * random_num
    return weight


def compute_transfer_function(summed_neuron_input, alpha_parameter):
    activation = 1.0 / (1.0 + math.exp(-alpha_parameter * summed_neuron_input))
    return activation


def compute_transfer_function_derivative(neuron_output, alpha_parameter):
    return alpha_parameter * neuron_output * (1.0 - neuron_output)


def obtain_selected_alphabet_training_values(data_set):
    training_data_list_a0 = (1,
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 1, 'A', 0,
                             'A')  # training data list 1 selected for the letter 'A'
    training_data_list_b0 = (2,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 2, 'B', 1,
                             'B')  # training data list 2, letter 'E', courtesy AJM
    training_data_list_c0 = (3,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3, 'C', 2,
                             'C')  # training data list 3, letter 'C', courtesy PKVR
    training_data_list_d0 = (4,
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 4, 'D', 3,
                             'O')  # training data list 4, letter 'D', courtesy TD
    training_data_list_e0 = (5,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 5, 'E', 4,
                             'E')  # training data list 5, letter 'E', courtesy BMcD
    training_data_list_f0 = (6,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 6, 'F', 4,
                             'E')  # training data list 6, letter 'F', courtesy SK
    training_data_list_g0 = (7,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 7, 'G', 1, 'C')
    training_data_list_h0 = (8,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 8, 'H', 0,
                             'A')  # training data list 8, letter 'H', courtesy JC
    training_data_list_i0 = (9,
                             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 9, 'I', 5,
                             'I')  # training data list 9, letter 'I', courtesy GR
    training_data_list_j0 = (10,
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                              0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 10, 'J', 5,
                             'I')  # training data list 10 selected for the letter 'L', courtesy JT
    training_data_list_k0 = (11,
                             [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                              1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], 11, 'K', 6,
                             'K')  # training data list 11 selected for the letter 'K', courtesy EO
    training_data_list_l0 = (12,
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 12, 'L', 7,
                             'L')  # training data list 12 selected for the letter 'L', courtesy PV
    training_data_list_m0 = (13,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                              0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                              0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], 13, 'M', 8,
                             'M')  # training data list 13 selected for the letter 'M', courtesy GR
    training_data_list_n0 = (14,
                             [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 14, 'N', 8,
                             'M')  # training data list 14 selected for the letter 'N'
    training_data_list_o0 = (15,
                             [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], 15, 'O', 3,
                             'O')  # training data list 15, letter 'O', courtesy TD
    training_data_list_p0 = (16,
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 16, 'P', 1,
                             'B')  # training data list 16, letter 'P', courtesy MT

    if data_set == 1:
        return training_data_list_a0
    if data_set == 2:
        return training_data_list_b0
    if data_set == 3:
        return training_data_list_c0
    if data_set == 4:
        return training_data_list_d0
    if data_set == 5:
        return training_data_list_e0
    if data_set == 6:
        return training_data_list_f0
    if data_set == 7:
        return training_data_list_g0
    if data_set == 8:
        return training_data_list_h0
    if data_set == 9:
        return training_data_list_i0
    if data_set == 10:
        return training_data_list_j0
    if data_set == 11:
        return training_data_list_k0
    if data_set == 12:
        return training_data_list_l0
    if data_set == 13:
        return training_data_list_m0
    if data_set == 14:
        return training_data_list_n0
    if data_set == 15:
        return training_data_list_o0
    if data_set == 16:
        return training_data_list_p0


def compute_single_feed_forward_pass(alpha_parameter, array, weight_array, bias_weight_array, local_array_length):
    np.zeros(local_array_length)
    local_array = np.zeros(local_array_length)
    sum_into_array = np.dot(weight_array, array)
    for node in range(local_array_length):
        node_sum_input = sum_into_array[node] + bias_weight_array[node]
        local_array[node] = compute_transfer_function(node_sum_input, alpha_parameter)
    return local_array


def compute_outputs_across_all_training_data(local_alpha, local_num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array):
    selected_training_data_set = 1
    while selected_training_data_set < local_num_training_data_sets + 1:
        training_data_list = obtain_selected_alphabet_training_values(selected_training_data_set)[1]
        input_data_list = []
        for node in range(input_array_length):
            training_data = training_data_list[node]
            input_data_list.append(training_data)
        hidden_array = compute_single_feed_forward_pass(local_alpha, input_data_list, w_weight_array,
                                                        bias_hidden_weight_array, hidden_array_length)
        output_array = compute_single_feed_forward_pass(local_alpha, hidden_array, v_weight_array,
                                                        bias_output_weight_array, output_array_length)
        desired_output_array = np.zeros(output_array_length)
        desired_class = training_data_list[4]
        desired_output_array[desired_class] = 1
        error_array = np.zeros(output_array_length)
        new_sse = 0.0
        for node in range(output_array_length):
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]
        selected_training_data_set = selected_training_data_set + 1


def back_propagate_output_to_hidden(local_alpha, local_eta, local_array_size_list, error_array, output_array,
                                    hidden_array, v_weight_array):
    local_hidden_array_length = local_array_size_list[1]
    local_output_array_length = local_array_size_list[2]
    transfer_function_derivative_array = np.zeros(local_output_array_length)
    for node in range(local_output_array_length):
        transfer_function_derivative_array[node] = compute_transfer_function_derivative(output_array[node], local_alpha)
    delta_v_wt_array = np.zeros((local_output_array_length, local_hidden_array_length))
    new_v_weight_array = np.zeros(
        (local_output_array_length, local_hidden_array_length))
    for row in range(local_output_array_length):
        for col in range(local_hidden_array_length):
            partial_sse_w_v_wt = -error_array[row] * transfer_function_derivative_array[row] * hidden_array[col]
            delta_v_wt_array[row, col] = -local_eta * partial_sse_w_v_wt
            new_v_weight_array[row, col] = v_weight_array[row, col] + delta_v_wt_array[row, col]
    return new_v_weight_array


def back_propagate_bias_output_weights(local_alpha, local_eta, local_array_size_list, error_array, output_array,
                                       bias_output_weight_array):
    local_output_array_length = local_array_size_list[2]

    delta_bias_output_array = np.zeros(local_output_array_length)
    new_bias_output_weight_array = np.zeros(local_output_array_length)
    transfer_function_derivative_array = np.zeros(local_output_array_length)

    for node in range(local_output_array_length):
        transfer_function_derivative_array[node] = compute_transfer_function_derivative(output_array[node], local_alpha)

    for node in range(local_output_array_length):
        partial_sse_w_bias_output = -error_array[node] * transfer_function_derivative_array[node]
        delta_bias_output_array[node] = -local_eta * partial_sse_w_bias_output
        new_bias_output_weight_array[node] = bias_output_weight_array[node] + delta_bias_output_array[node]

    return new_bias_output_weight_array


def back_propagate_hidden_to_input(local_alpha, local_eta, local_array_size_list, error_array, output_array,
                                   hidden_array, input_array, v_weight_array, w_weight_array):
    local_input_array_length = local_array_size_list[0]
    local_hidden_array_length = local_array_size_list[1]
    local_output_array_length = local_array_size_list[2]
    transfer_function_derivative_hidden_array = np.zeros(local_hidden_array_length)

    for node in range(local_hidden_array_length):
        transfer_function_derivative_hidden_array[node] = compute_transfer_function_derivative(hidden_array[node],
                                                                                               local_alpha)

    error_times_t_function_derivative_output_array = np.zeros(local_output_array_length)
    transfer_function_derivative_output_array = np.zeros(local_output_array_length)
    weighted_error_array = np.zeros(local_hidden_array_length)

    for outputNode in range(local_output_array_length):
        transfer_function_derivative_output_array[outputNode] = \
            compute_transfer_function_derivative(output_array[outputNode], local_alpha)
        error_times_t_function_derivative_output_array[outputNode] = \
            error_array[outputNode] * transfer_function_derivative_output_array[outputNode]

    for hiddenNode in range(local_hidden_array_length):
        weighted_error_array[hiddenNode] = 0
        for outputNode in range(local_output_array_length):
            weighted_error_array[hiddenNode] = weighted_error_array[hiddenNode] \
                                               + v_weight_array[outputNode, hiddenNode] * \
                                               error_times_t_function_derivative_output_array[outputNode]

    delta_w_wt_array = np.zeros((local_hidden_array_length, local_input_array_length))
    new_w_weight_array = np.zeros((local_hidden_array_length, local_input_array_length))

    for row in range(local_hidden_array_length):
        for col in range(local_input_array_length):
            partial_sse_w_w_wts = \
                -transfer_function_derivative_hidden_array[row] * input_array[col] * weighted_error_array[row]
            delta_w_wt_array[row, col] = -local_eta * partial_sse_w_w_wts
            new_w_weight_array[row, col] = w_weight_array[row, col] + delta_w_wt_array[row, col]

    return new_w_weight_array


def back_propagate_bias_hidden_weights(local_alpha, local_eta, local_array_size_list, error_array, output_array,
                                       hidden_array, v_weight_array, bias_hidden_weight_array):
    local_hidden_array_length = local_array_size_list[1]
    local_output_array_length = local_array_size_list[2]

    error_times_t_func_deriv_output_array = np.zeros(local_output_array_length)
    transfer_func_deriv_output_array = np.zeros(local_output_array_length)
    weighted_error_array = np.zeros(local_hidden_array_length)

    transfer_func_deriv_hidden_array = np.zeros(local_hidden_array_length)
    partial_sse_w_bias_hidden = np.zeros(local_hidden_array_length)
    delta_bias_hidden_array = np.zeros(local_hidden_array_length)
    new_bias_hidden_weight_array = np.zeros(local_hidden_array_length)

    for node in range(local_hidden_array_length):
        transfer_func_deriv_hidden_array[node] = compute_transfer_function_derivative(hidden_array[node], local_alpha)

    for outputNode in range(local_output_array_length):
        transfer_func_deriv_output_array[outputNode] = compute_transfer_function_derivative(output_array[outputNode],
                                                                                            local_alpha)
        error_times_t_func_deriv_output_array[outputNode] = error_array[outputNode] * transfer_func_deriv_output_array[
            outputNode]

    for hiddenNode in range(local_hidden_array_length):
        weighted_error_array[hiddenNode] = 0
        for outputNode in range(local_output_array_length):
            weighted_error_array[hiddenNode] = (weighted_error_array[hiddenNode]
                                                + v_weight_array[outputNode, hiddenNode] *
                                                error_times_t_func_deriv_output_array[outputNode])

    for hiddenNode in range(local_hidden_array_length):
        partial_sse_w_bias_hidden[hiddenNode] = -transfer_func_deriv_hidden_array[hiddenNode] * weighted_error_array[
            hiddenNode]
        delta_bias_hidden_array[hiddenNode] = -local_eta * partial_sse_w_bias_hidden[hiddenNode]
        new_bias_hidden_weight_array[hiddenNode] = bias_hidden_weight_array[hiddenNode] + delta_bias_hidden_array[
            hiddenNode]

    return new_bias_hidden_weight_array


def iterate_over_data(w_weight_array, bias_hidden_weight_array, v_weight_array, bias_output_weight_array):
    iteration = 0
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
        hidden_array = compute_single_feed_forward_pass(alpha, input_data_array, w_weight_array,
                                                        bias_hidden_weight_array, hidden_array_length)

        output_array = compute_single_feed_forward_pass(alpha, hidden_array, v_weight_array,
                                                        bias_output_weight_array, output_array_length)
        error_array = np.zeros(output_array_length)
        new_sse = 0.0
        for node in range(output_array_length):
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]
        new_v_weight_array = back_propagate_output_to_hidden(alpha, eta, array_size_list, error_array, output_array,
                                                             hidden_array, v_weight_array)
        new_bias_output_weight_array = back_propagate_bias_output_weights(alpha, eta, array_size_list, error_array,
                                                                          output_array, bias_output_weight_array)

        new_w_weight_array = back_propagate_hidden_to_input(alpha, eta, array_size_list, error_array, output_array,
                                                            hidden_array, input_data_list, v_weight_array,
                                                            w_weight_array)

        new_bias_hidden_weight_array = back_propagate_bias_hidden_weights(alpha, eta, array_size_list, error_array,
                                                                          output_array, hidden_array, v_weight_array,
                                                                          bias_hidden_weight_array)

        v_weight_array = new_v_weight_array[:]
        bias_output_weight_array = new_bias_output_weight_array[:]
        w_weight_array = new_w_weight_array[:]
        bias_hidden_weight_array = new_bias_hidden_weight_array[:]
        hidden_array = compute_single_feed_forward_pass(alpha, input_data_list, w_weight_array,
                                                        bias_hidden_weight_array, hidden_array_length)

        output_array = compute_single_feed_forward_pass(alpha, hidden_array, v_weight_array,
                                                        bias_output_weight_array, output_array_length)

        new_sse = 0.0
        for node in range(output_array_length):
            error_array[node] = desired_output_array[node] - output_array[node]
            new_sse = new_sse + error_array[node] * error_array[node]

        if new_sse < epsilon:
            break


def save_weights_to_file(local_w_weight_array, num_upper_nodes, num_lower_nodes, file_name):
    weight_list = list()
    for row in range(num_upper_nodes):
        for col in range(num_lower_nodes):
            local_weight = local_w_weight_array[row, col]
            weight_list.append(local_weight)

    weight_file = open(file_name, 'w')
    for item in weight_list:
        weight_file.write("%s\n" % item)
    weight_file.close()


def save_bias_weights_to_file(local_array_length, bias_weight_array,file_name):
    bias_weight_list = list()
    for node in range(local_array_length):
        local_bias_weight = bias_weight_array[node]
        bias_weight_list.append(local_bias_weight)
    bias_weight_file = open(file_name, 'w')
    for item in bias_weight_list:
        bias_weight_file.write("%s\n" % item)
    bias_weight_file.close()


def main():
    w_weight_array = initialize_weight_array(w_weight_array_size_list)
    v_weight_array = initialize_weight_array(v_weight_array_size_list)
    bias_hidden_weight_array = initialize_bias_weight_array(bias_hidden_weight_array_size)
    bias_output_weight_array = initialize_bias_weight_array(bias_output_weight_array_size)

    compute_outputs_across_all_training_data(alpha, num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array)

    iterate_over_data(w_weight_array, bias_hidden_weight_array, v_weight_array, bias_output_weight_array)

    compute_outputs_across_all_training_data(alpha, num_training_data_sets, w_weight_array,
                                             bias_hidden_weight_array, v_weight_array, bias_output_weight_array)

    save_weights_to_file(w_weight_array, hidden_array_length, input_array_length,
                         'C:\\Users\\F985\\Documents\\Python\\datafiles\\GB1wWeightFile.txt')
    save_weights_to_file(v_weight_array, output_array_length, hidden_array_length,
                         'C:\\Users\\F985\\Documents\\Python\\datafiles\\GB1vWeightFile.txt')
    save_bias_weights_to_file(hidden_array_length, bias_hidden_weight_array,
                              'C:\\Users\\F985\\Documents\\Python\\datafiles\\GB1wBiasWeightFile.txt')
    save_bias_weights_to_file(output_array_length, bias_output_weight_array,
                              'C:\\Users\\F985\\Documents\\Python\\datafiles\\GB1vBiasWeightFile.txt')


if __name__ == '__main__':
    main()
