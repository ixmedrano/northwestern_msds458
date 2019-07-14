import math
import random
import numpy as np

number_input_nodes = 2
number_hidden_nodes = 2
number_output_nodes = 2
array_size_list = (number_input_nodes, number_hidden_nodes, number_output_nodes)
number_bottom_nodes = array_size_list[0]
number_upper_nodes = array_size_list[1]


def transfer_function(summed_neuron_input, alpha):
    return 1.0 / (1.0 + math.exp(-alpha * summed_neuron_input))


def transfer_function_derivative(neuron_output, alpha):
    return alpha * neuron_output * (1.0 - neuron_output)


def initialize_weight():
    weight = 1 - 2 * random.random()
    return weight


def initialize_weight_array(debug_initialize_off):
    weight0 = initialize_weight()
    weight01 = initialize_weight()
    weight10 = initialize_weight()
    weight11 = initialize_weight()
    weight_array = np.array([[weight0, weight01], [weight10, weight11]])

    # Debug mode: if debug is set to False, then we DO NOT do the prints
    if not debug_initialize_off:
        # Print the weights
        print(' ')
        print('  Inside initializeWeightArray')
        print('    The weights just initialized are: ')
        print('      weight00 = %.4f,' % weight0)
        print('      weight01 = %.4f,' % weight01)
        print('      weight10 = %.4f,' % weight10)
        print('      weight11 = %.4f,' % weight11)

    # Debug mode: if debug is set to False, then we DO NOT do the prints
    if not debug_initialize_off:
        # Print the entire weights array.
        print(' ')
        print('    The weight Array just established is: ', weight_array)
        print(' ')
        print('    Within this array: ')
        print('      weight00 = %.4f    weight10 = %.4f' % (weight_array[0, 0], weight_array[0, 1]))
        print('      weight01 = %.4f    weight11 = %.4f' % (weight_array[1, 0], weight_array[1, 1]))
        print('  Returning to calling procedure')

    # We return the array to the calling procedure, 'main'.
    return weight_array


def initialize_bias_weight_array():
    # Initialize the weight variables with random weights
    bias_weight0 = initialize_weight()
    bias_weight1 = initialize_weight()

    bias_weight_array = np.array([bias_weight0, bias_weight1])

    return bias_weight_array


def obtain_random_xor_training_values():
    training_dataset_number = random.randint(1, 4)
    if training_dataset_number > 1.1:  # The selection is for training lists between 2 & 4
        if training_dataset_number > 2.1:  # The selection is for training lists between 3 & 4
            if training_dataset_number > 3.1:  # The selection is for training list 4
                training_data_list = (1, 1, 0, 1, 3)  # training data list 4 selected
            else:
                training_data_list = (1, 0, 1, 0, 2)  # training data list 3 selected
        else:
            training_data_list = (0, 1, 1, 0, 1)  # training data list 2 selected
    else:
        training_data_list = (0, 0, 0, 1, 0)  # training data list 1 selected

    return training_data_list


def compute_single_neuron_activation(alpha, weight0, weight01, input0, input1, bias):
    # Obtain the inputs into the neuron; this is the sum of weights times inputs
    summed_neuron_input = weight0 * input0 + weight01 * input1 + bias

    # Pass the summedNeuronActivation and the transfer function parameter alpha into the transfer function
    activation = transfer_function(summed_neuron_input, alpha)

    return activation


def compute_single_feedforward_pass(alpha, input_data_list, w_weight_array, v_weight_array, bias_hidden_weight_array,
                                    bias_output_weight_array):
    input0 = input_data_list[0]
    input1 = input_data_list[1]

    # Assign the input-to-hidden weights to specific variables
    w_wt00 = w_weight_array[0, 0]
    w_wt10 = w_weight_array[0, 1]
    w_wt01 = w_weight_array[1, 0]
    w_wt11 = w_weight_array[1, 1]

    # Assign the hidden-to-output weights to specific variables
    v_wt00 = v_weight_array[0, 0]
    v_wt10 = v_weight_array[0, 1]
    v_wt01 = v_weight_array[1, 0]
    v_wt11 = v_weight_array[1, 1]

    bias_hidden0 = bias_hidden_weight_array[0]
    bias_hidden1 = bias_hidden_weight_array[1]
    bias_output0 = bias_output_weight_array[0]
    bias_output1 = bias_output_weight_array[1]

    # Obtain the activations of the hidden nodes
    hidden_activation0 = compute_single_neuron_activation(alpha, w_wt00, w_wt10, input0, input1, bias_hidden0)

    hidden_activation1 = compute_single_neuron_activation(alpha, w_wt01, w_wt11, input0, input1, bias_hidden1)

    # Obtain the activations of the output nodes
    output_activation0 = compute_single_neuron_activation(alpha, v_wt00, v_wt10, hidden_activation0, hidden_activation1,
                                                          bias_output0)
    output_activation1 = compute_single_neuron_activation(alpha, v_wt00, v_wt10, hidden_activation0, hidden_activation1,
                                                          bias_output0)

    actual_all_nodes_output_list = (hidden_activation0, hidden_activation1, output_activation0, output_activation1)

    return actual_all_nodes_output_list


def compute_sse_values(alpha, sse_initial_array, w_weight_array, v_weight_array, bias_hidden_weight_array
                       , bias_output_weight_array):
    # Compute a single feed-forward pass and obtain the Actual Outputs for zeroth data set
    input_data_list = (0, 0)
    actual_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list, w_weight_array,
                                                                   v_weight_array,
                                                                   bias_hidden_weight_array, bias_output_weight_array)
    actual_output0 = actual_all_nodes_output_list[2]
    actual_output1 = actual_all_nodes_output_list[3]
    error0 = 0.0 - actual_output0
    error1 = 1.0 - actual_output1
    sse_initial_array[0] = error0 ** 2 + error1 ** 2

    # Compute a single feed-forward pass and obtain the Actual Outputs for first data set
    input_data_list = (0, 1)
    actual_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list, w_weight_array,
                                                                   v_weight_array,
                                                                   bias_hidden_weight_array, bias_output_weight_array)

    actual_output0 = actual_all_nodes_output_list[2]
    actual_output1 = actual_all_nodes_output_list[3]
    error0 = 1.0 - actual_output0
    error1 = 0.0 - actual_output1
    sse_initial_array[1] = error0 ** 2 + error1 ** 2

    # Compute a single feed-forward pass and obtain the Actual Outputs for second data set
    input_data_list = (1, 0)
    actual_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list, w_weight_array,
                                                                   v_weight_array,
                                                                   bias_hidden_weight_array, bias_output_weight_array)

    actual_output0 = actual_all_nodes_output_list[2]
    actual_output1 = actual_all_nodes_output_list[3]
    error0 = 1.0 - actual_output0
    error1 = 0.0 - actual_output1
    sse_initial_array[2] = error0 ** 2 + error1 ** 2

    # Compute a single feed-forward pass and obtain the Actual Outputs for third data set
    input_data_list = (1, 1)
    actual_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list, w_weight_array,
                                                                   v_weight_array,
                                                                   bias_hidden_weight_array, bias_output_weight_array)

    actual_output0 = actual_all_nodes_output_list[2]
    actual_output1 = actual_all_nodes_output_list[3]
    error0 = 0.0 - actual_output0
    error1 = 1.0 - actual_output1
    sse_initial_array[3] = error0 ** 2 + error1 ** 2

    # Initialize an array of SSE values
    sse_initial_total = sse_initial_array[0] + sse_initial_array[1] + sse_initial_array[2] + sse_initial_array[3]

    sse_initial_array[4] = sse_initial_total

    return sse_initial_array


def print_and_trace_back_propagate_output_to_hidden(eta, error_list,
                                                    actual_all_nodes_output_list, trans_func_derivative_list,
                                                    delta_v_wt_array, v_weight_array,
                                                    new_v_weight_array):
    hidden_node0 = actual_all_nodes_output_list[0]
    hidden_node1 = actual_all_nodes_output_list[1]
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]
    trans_func_derivative0 = trans_func_derivative_list[0]
    trans_func_derivative1 = trans_func_derivative_list[1]

    error0 = error_list[0]
    error1 = error_list[1]

    print(' ')
    print('In Print and Trace for Back propagation: Hidden to Output Weights')
    print('  Assuming alpha = 1')
    print(' ')
    print('  The hidden node activations are:')
    print('    Hidden node 0: ', '  %.4f' % hidden_node0, '  Hidden node 1: ', '  %.4f' % hidden_node1)
    print(' ')
    print('  The output node activations are:')
    print('    Output node 0: ', '  %.3f' % output_node0, '   Output node 1: ', '  %.3f' % output_node1)
    print(' ')
    print('  The transfer function derivatives are: ')
    print('    Derive-F(0): ', '     %.3f' % trans_func_derivative0, '   Derive-F(1): ', '     %.3f'
          % trans_func_derivative1)

    print(' ')
    print('The computed values for the deltas are: ')
    print('                eta  *  error  *   trFncDeriv *   hidden')
    print('  deltaVWt00 = ', ' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % trans_func_derivative0,
          '  * %.4f' % hidden_node0)
    print('  deltaVWt01 = ', ' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % trans_func_derivative1,
          '  * %.4f' % hidden_node0)
    print('  deltaVWt10 = ', ' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % trans_func_derivative0,
          '  * %.4f' % hidden_node1)
    print('  deltaVWt11 = ', ' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % trans_func_derivative1,
          '  * %.4f' % hidden_node1)
    print(' ')
    print('Values for the hidden-to-output connection weights:')
    print('           Old:     New:      eta*Delta:')
    print('[0,0]:   %.4f' % v_weight_array[0, 0], '  %.4f' % new_v_weight_array[0, 0],
          '  %.4f' % delta_v_wt_array[0, 0])
    print('[0,1]:   %.4f' % v_weight_array[1, 0], '  %.4f' % new_v_weight_array[1, 0],
          '  %.4f' % delta_v_wt_array[1, 0])
    print('[1,0]:   %.4f' % v_weight_array[0, 1], '  %.4f' % new_v_weight_array[0, 1],
          '  %.4f' % delta_v_wt_array[0, 1])
    print('[1,1]:   %.4f' % v_weight_array[1, 1], '  %.4f' % new_v_weight_array[1, 1],
          '  %.4f' % delta_v_wt_array[1, 1])


def print_and_trace_back_propagate_hidden_to_input(eta, input_data_list, error_list,
                                                   actual_all_nodes_output_list, trans_func_derivative_hidden_list,
                                                   delta_w_wt_array, v_weight_array, w_weight_array,
                                                   new_w_weight_array):
    input_node0 = input_data_list[0]
    input_node1 = input_data_list[1]
    hidden_node0 = actual_all_nodes_output_list[0]
    hidden_node1 = actual_all_nodes_output_list[1]
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]
    trans_func_derive_hidden0 = trans_func_derivative_hidden_list[0]
    trans_func_derive_hidden1 = trans_func_derivative_hidden_list[1]

    v_wt00 = v_weight_array[0, 0]
    v_wt01 = v_weight_array[1, 0]
    v_wt10 = v_weight_array[0, 1]
    v_wt11 = v_weight_array[1, 1]

    error0 = error_list[0]
    error1 = error_list[1]

    sum_term_h0 = v_wt00 * error0 + v_wt01 * error1
    sum_term_h1 = v_wt10 * error0 + v_wt11 * error1

    print(' ')
    print('In Print and Trace for Backpropagation: Input to Hidden Weights')
    print('  Assuming alpha = 1')
    print(' ')
    print('  The hidden node activations are:')
    print('    Hidden node 0: ', '  %.4f' % hidden_node0, '  Hidden node 1: ', '  %.4f' % hidden_node1)
    print(' ')
    print('  The output node activations are:')
    print('    Output node 0: ', '  %.3f' % output_node0, '   Output node 1: ', '  %.3f' % output_node1)
    print(' ')
    print('  The transfer function derivatives at the hidden nodes are: ')
    print('    Deriv-F(0): ', '     %.3f' % trans_func_derive_hidden0, '   Deriv-F(1): ', '     %.3f'
          % trans_func_derive_hidden1)

    print(' ')
    print('The computed values for the deltas are: ')
    print('                eta  *  error  *   trFncDeriv *   input    * SumTerm for given H')
    print('  deltaWWt00 = ', ' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % trans_func_derive_hidden0,
          '  * %.4f' % input_node0, '  * %.4f' % sum_term_h0)
    print('  deltaWWt01 = ', ' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % trans_func_derive_hidden1,
          '  * %.4f' % input_node0, '  * %.4f' % sum_term_h1)
    print('  deltaWWt10 = ', ' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % trans_func_derive_hidden0,
          '  * %.4f' % input_node1, '  * %.4f' % sum_term_h0)
    print('  deltaWWt11 = ', ' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % trans_func_derive_hidden1,
          '  * %.4f' % input_node1, '  * %.4f' % sum_term_h1)
    print(' ')
    print('Values for the input-to-hidden connection weights:')
    print('           Old:     New:      eta*Delta:')
    print('[0,0]:   %.4f' % w_weight_array[0, 0], '  %.4f' % new_w_weight_array[0, 0],
          '  %.4f' % delta_w_wt_array[0, 0])
    print('[0,1]:   %.4f' % w_weight_array[1, 0], '  %.4f' % new_w_weight_array[1, 0],
          '  %.4f' % delta_w_wt_array[1, 0])
    print('[1,0]:   %.4f' % w_weight_array[0, 1], '  %.4f' % new_w_weight_array[0, 1],
          '  %.4f' % delta_w_wt_array[0, 1])
    print('[1,1]:   %.4f' % w_weight_array[1, 1], '  %.4f' % new_w_weight_array[1, 1],
          '  %.4f' % delta_w_wt_array[1, 1])


def back_propagate_output_to_hidden(alpha, eta, error_list, actual_all_nodes_output_list, v_weight_array):
    error0 = error_list[0]
    error1 = error_list[1]

    v_wt00 = v_weight_array[0, 0]
    v_wt01 = v_weight_array[1, 0]
    v_wt10 = v_weight_array[0, 1]
    v_wt11 = v_weight_array[1, 1]

    hidden_node0 = actual_all_nodes_output_list[0]
    hidden_node1 = actual_all_nodes_output_list[1]
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]

    trans_func_derive0 = transfer_function_derivative(output_node0, alpha)
    trans_func_derive1 = transfer_function_derivative(output_node1, alpha)
    trans_func_derive_list = (trans_func_derive0, trans_func_derive1)

    partial_sse_w_vwt00 = -error0 * trans_func_derive0 * hidden_node0
    partial_sse_w_vwt01 = -error1 * trans_func_derive1 * hidden_node0
    partial_sse_w_vwt10 = -error0 * trans_func_derive0 * hidden_node1
    partial_sse_w_vwt11 = -error1 * trans_func_derive1 * hidden_node1

    delta_v_wt00 = -eta * partial_sse_w_vwt00
    delta_v_wt01 = -eta * partial_sse_w_vwt01
    delta_v_wt10 = -eta * partial_sse_w_vwt10
    delta_v_wt11 = -eta * partial_sse_w_vwt11
    delta_v_wt_array = np.array([[delta_v_wt00, delta_v_wt10], [delta_v_wt01, delta_v_wt11]])

    v_wt00 = v_wt00 + delta_v_wt00
    v_wt01 = v_wt01 + delta_v_wt01
    v_wt10 = v_wt10 + delta_v_wt10
    v_wt11 = v_wt11 + delta_v_wt11

    new_v_weight_array = np.array([[v_wt00, v_wt10], [v_wt01, v_wt11]])

    print_and_trace_back_propagate_output_to_hidden(eta, error_list, actual_all_nodes_output_list,
                                                    trans_func_derive_list, delta_v_wt_array, v_weight_array,
                                                    new_v_weight_array)

    return new_v_weight_array


def back_propagate_bias_output_weights(alpha, eta, error_list, actual_all_nodes_output_list,
                                       bias_output_weight_array):
    error0 = error_list[0]
    error1 = error_list[1]

    # Unpack the bias_output_weight_array, we will only be modifying the biasOutput terms
    bias_output_wt0 = bias_output_weight_array[0]
    bias_output_wt1 = bias_output_weight_array[1]

    # Unpack the outputNodes
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]

    trans_func_derive0 = transfer_function_derivative(output_node0, alpha)
    trans_func_derive1 = transfer_function_derivative(output_node1, alpha)

    partial_sse_w_bias_output0 = -error0 * trans_func_derive0
    partial_sse_w_bias_output1 = -error1 * trans_func_derive1

    delta_bias_output0 = -eta * partial_sse_w_bias_output0
    delta_bias_output1 = -eta * partial_sse_w_bias_output1

    bias_output_wt0 = bias_output_wt0 + delta_bias_output0
    bias_output_wt1 = bias_output_wt1 + delta_bias_output1

    # Note that only the bias weights for the output nodes have been changed.
    new_bias_output_weight_array = np.array([bias_output_wt0, bias_output_wt1])

    return new_bias_output_weight_array


def back_propagate_hidden_to_input(alpha, eta, error_list, actual_all_nodes_output_list, input_data_list,
                                   v_weight_array, w_weight_array):
    # Unpack the error_list and the v_weight_array
    error0 = error_list[0]
    error1 = error_list[1]

    v_wt00 = v_weight_array[0, 0]
    v_wt01 = v_weight_array[1, 0]
    v_wt10 = v_weight_array[0, 1]
    v_wt11 = v_weight_array[1, 1]

    w_wt00 = w_weight_array[0, 0]
    w_wt01 = w_weight_array[1, 0]
    w_wt10 = w_weight_array[0, 1]
    w_wt11 = w_weight_array[1, 1]

    input_node0 = input_data_list[0]
    input_node1 = input_data_list[1]
    hidden_node0 = actual_all_nodes_output_list[0]
    hidden_node1 = actual_all_nodes_output_list[1]
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]

    trans_func_derive_hidden0 = transfer_function_derivative(hidden_node0, alpha)
    trans_func_derive_hidden1 = transfer_function_derivative(hidden_node1, alpha)

    # We also need the transfer function derivative applied to the output at the output node
    trans_func_derive_output0 = transfer_function_derivative(output_node0, alpha)
    trans_func_derive_output1 = transfer_function_derivative(output_node1, alpha)

    error_times_trans_d_output0 = error0 * trans_func_derive_output0
    error_times_trans_d_output1 = error1 * trans_func_derive_output1

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in these equations
    partial_sse_w_wwt00 = -trans_func_derive_hidden0 * input_node0 * (
            v_wt00 * error_times_trans_d_output0 + v_wt01 * error_times_trans_d_output1)
    partial_sse_w_wwt01 = -trans_func_derive_hidden1 * input_node0 * (
            v_wt10 * error_times_trans_d_output0 + v_wt11 * error_times_trans_d_output1)
    partial_sse_w_wwt10 = -trans_func_derive_hidden0 * input_node1 * (
            v_wt00 * error_times_trans_d_output0 + v_wt01 * error_times_trans_d_output1)
    partial_sse_w_wwt11 = -trans_func_derive_hidden1 * input_node1 * (
            v_wt10 * error_times_trans_d_output0 + v_wt11 * error_times_trans_d_output1)

    delta_w_wt00 = -eta * partial_sse_w_wwt00
    delta_w_wt01 = -eta * partial_sse_w_wwt01
    delta_w_wt10 = -eta * partial_sse_w_wwt10
    delta_w_wt11 = -eta * partial_sse_w_wwt11

    w_wt00 = w_wt00 + delta_w_wt00
    w_wt01 = w_wt01 + delta_w_wt01
    w_wt10 = w_wt10 + delta_w_wt10
    w_wt11 = w_wt11 + delta_w_wt11

    new_w_weight_array = np.array([[w_wt00, w_wt10], [w_wt01, w_wt11]])

    return new_w_weight_array


def back_propagate_bias_hidden_weights(alpha, eta, error_list, actual_all_nodes_output_list, v_weight_array,
                                       bias_hidden_weight_array):
    # Unpack the error_list and v_weight_array
    error0 = error_list[0]
    error1 = error_list[1]

    v_wt00 = v_weight_array[0, 0]
    v_wt01 = v_weight_array[1, 0]
    v_wt10 = v_weight_array[0, 1]
    v_wt11 = v_weight_array[1, 1]

    # Unpack the biasWeightArray, we will only be modifying the biasOutput terms, but need to have
    #   all the bias weights for when we redefine the biasWeightArray
    bias_hidden_wt0 = bias_hidden_weight_array[0]
    bias_hidden_wt1 = bias_hidden_weight_array[1]

    # Unpack the outputNodes
    hidden_node0 = actual_all_nodes_output_list[0]
    hidden_node1 = actual_all_nodes_output_list[1]
    output_node0 = actual_all_nodes_output_list[2]
    output_node1 = actual_all_nodes_output_list[3]

    trans_func_derive_output0 = transfer_function_derivative(output_node0, alpha)
    trans_func_derive_output1 = transfer_function_derivative(output_node1, alpha)
    trans_func_derive_hidden0 = transfer_function_derivative(hidden_node0, alpha)
    trans_func_derive_hidden1 = transfer_function_derivative(hidden_node1, alpha)

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in these equations

    error_times_trans_d_output0 = error0 * trans_func_derive_output0
    error_times_trans_d_output1 = error1 * trans_func_derive_output1

    partial_sse_w_bias_hidden0 = -trans_func_derive_hidden0 * (error_times_trans_d_output0 * v_wt00 +
                                                               error_times_trans_d_output1 * v_wt01)
    partial_sse_w_bias_hidden1 = -trans_func_derive_hidden1 * (error_times_trans_d_output0 * v_wt10 +
                                                               error_times_trans_d_output1 * v_wt11)

    delta_bias_hidden0 = -eta * partial_sse_w_bias_hidden0
    delta_bias_hidden1 = -eta * partial_sse_w_bias_hidden1

    bias_hidden_wt0 = bias_hidden_wt0 + delta_bias_hidden0
    bias_hidden_wt1 = bias_hidden_wt1 + delta_bias_hidden1

    # Note that only the bias weights for the hidden nodes have been changed.
    new_bias_hidden_weight_array = np.array([bias_hidden_wt0, bias_hidden_wt1])

    return new_bias_hidden_weight_array


def main():
    # Parameter definitions, to be replaced with user inputs
    alpha = 1  # parameter governing steepness of sigmoid transfer function
    max_num_iterations = 20000  # temporarily set to 10 for testing
    eta = 0.1  # training rate

    w_weight_array = initialize_weight_array(True)

    v_weight_array = initialize_weight_array(True)

    # The bias weights are stored in a 1-D array
    bias_hidden_weight_array = initialize_bias_weight_array()
    bias_output_weight_array = initialize_bias_weight_array()

    initial_w_weight_array = w_weight_array[:]
    initial_v_weight_array = v_weight_array[:]

    print()
    print('The initial weights for this neural network are:')
    print('       Input-to-Hidden                            Hidden-to-Output')
    print('  w(0,0) = %.4f   w(1,0) = %.4f         v(0,0) = %.4f   v(1,0) = %.4f' % (initial_w_weight_array[0, 0],
                                                                                     initial_w_weight_array[0, 1],
                                                                                     initial_v_weight_array[0, 0],
                                                                                     initial_v_weight_array[0, 1]))
    print('  w(0,1) = %.4f   w(1,1) = %.4f         v(0,1) = %.4f   v(1,1) = %.4f' % (initial_w_weight_array[1, 0],
                                                                                     initial_w_weight_array[1, 1],
                                                                                     initial_v_weight_array[1, 0],
                                                                                     initial_v_weight_array[1, 1]))
    print(' ')
    print('       Bias at Hidden Layer                          Bias at Output Layer')
    print('       b(hidden,0) = %.4f                           b(output,0) = %.4f' % (bias_hidden_weight_array[0],
                                                                                      bias_output_weight_array[0]))
    print('       b(hidden,1) = %.4f                           b(output,1) = %.4f' % (bias_hidden_weight_array[1],
                                                                                      bias_output_weight_array[1]))

    epsilon = 0.2
    iteration = 0

    # Initialize an array of SSE values
    # The first four SSE values are the SSE's for specific input/output pairs;
    #   the fifth is the sum of all the SSE's.
    sse_initial_array = [0, 0, 0, 0, 0]

    sse_initial_array = compute_sse_values(alpha, sse_initial_array, w_weight_array, v_weight_array,
                                           bias_hidden_weight_array, bias_output_weight_array)

    # Start the sse_array at the same values as the Initial SSE Array
    sse_array = sse_initial_array[:]
    sse_initial_total = sse_array[4]

    # Optionally, print a summary of the initial SSE Total (sum across SSEs for each training data set)
    #   and the specific SSE values
    # Set a local debug print parameter
    debug_sse_initial_computation_report_off = True

    if not debug_sse_initial_computation_report_off:
        print(' ')
        print('In main, SSE computations completed, Total of all SSEs = %.4f' % sse_array[4])
        print('  For input nodes (0,0), sse_array[0] = %.4f' % sse_array[0])
        print('  For input nodes (0,1), sse_array[1] = %.4f' % sse_array[1])
        print('  For input nodes (1,0), sse_array[2] = %.4f' % sse_array[2])
        print('  For input nodes (1,1), sse_array[3] = %.4f' % sse_array[3])

    while iteration < max_num_iterations:

        ####################################################################################################
        # Next step - Obtain a single set of input values for the X-OR problem; two integers - can be 0 or 1
        ####################################################################################################

        # Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        training_data_list = obtain_random_xor_training_values()
        input0 = training_data_list[0]
        input1 = training_data_list[1]
        desired_output0 = training_data_list[2]
        desired_output1 = training_data_list[3]
        set_number = training_data_list[4]
        print(' ')
        print('Randomly selecting XOR inputs for XOR, identifying desired outputs for this training pass:')
        print('          Input0 = ', input0, '            Input1 = ', input1)
        print(' Desired Output0 = ', desired_output0, '   Desired Output1 = ', desired_output1)
        print(' ')

        ####################################################################################################
        # Compute a single feed-forward pass
        ####################################################################################################

        # Create the inputData list
        input_data_list = (input0, input1)

        # Compute a single feed-forward pass and obtain the Actual Outputs
        actual_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list,
                                                                       w_weight_array, v_weight_array,
                                                                       bias_hidden_weight_array,
                                                                       bias_output_weight_array)

        # Assign the hidden and output values to specific different variables
        actual_hidden_output0 = actual_all_nodes_output_list[0]
        actual_hidden_output1 = actual_all_nodes_output_list[1]
        actual_output0 = actual_all_nodes_output_list[2]
        actual_output1 = actual_all_nodes_output_list[3]

        # Determine the error between actual and desired outputs

        error0 = desired_output0 - actual_output0
        error1 = desired_output1 - actual_output1
        error_list = (error0, error1)

        # Compute the Summed Squared Error, or SSE
        sse_initial = error0 ** 2 + error1 ** 2

        debug_main_compute_forward_pass_outputs_off = True

        # Debug print the actual outputs from the two output neurons
        if not debug_main_compute_forward_pass_outputs_off:
            print(' ')
            print('In main; have just completed a feed forward pass with training set inputs', input0, input1)
            print('  The activations (actual outputs) for the two hidden neurons are:')
            print('    actual_hidden_output0 = %.4f' % actual_hidden_output0)
            print('    actual_hidden_output1 = %.4f' % actual_hidden_output1)
            print('  The activations (actual outputs) for the two output neurons are:')
            print('    actual_output0 = %.4f' % actual_output0)
            print('    actual_output1 = %.4f' % actual_output1)
            print('  Initial SSE (before back propagation) = %.6f' % sse_initial)
            print('  Corresponding SSE (from initial SSE determination) = %.6f' % sse_array[set_number])

        # Perform first part of the back propagation of weight changes
        new_v_weight_array = back_propagate_output_to_hidden(alpha, eta, error_list, actual_all_nodes_output_list,
                                                             v_weight_array)

        new_w_weight_array = back_propagate_hidden_to_input(alpha, eta, error_list, actual_all_nodes_output_list,
                                                            input_data_list,
                                                            v_weight_array, w_weight_array)

        # Debug prints on the weight arrays
        debug_weight_array_off = False
        if not debug_weight_array_off:
            print(' ')
            print('    The weights before back propagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' % (w_weight_array[0, 0],
                                                                                               w_weight_array[0, 1],
                                                                                               v_weight_array[0, 0],
                                                                                               v_weight_array[0, 1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' % (w_weight_array[1, 0],
                                                                                               w_weight_array[1, 1],
                                                                                               v_weight_array[1, 0],
                                                                                               v_weight_array[1, 1]))
            print(' ')
            print('    The weights after back propagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' % (new_w_weight_array[0, 0],
                                                                                               new_w_weight_array[0, 1],
                                                                                               new_v_weight_array[0, 0],
                                                                                               new_v_weight_array[
                                                                                                   0, 1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' % (new_w_weight_array[1, 0],
                                                                                               new_w_weight_array[1, 1],
                                                                                               new_v_weight_array[1, 0],
                                                                                               new_v_weight_array[
                                                                                                   1, 1]))

        # Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
        v_weight_array = new_v_weight_array[:]

        # Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
        w_weight_array = new_w_weight_array[:]

        # Run the feed forward function again, to compare the results after just adjusting the hidden-to-output weights
        new_all_nodes_output_list = compute_single_feedforward_pass(alpha, input_data_list, w_weight_array,
                                                                    v_weight_array,
                                                                    bias_hidden_weight_array, bias_output_weight_array)

        new_output0 = new_all_nodes_output_list[2]
        new_output1 = new_all_nodes_output_list[3]

        # Determine the new error between actual and desired outputs
        new_error0 = desired_output0 - new_output0
        new_error1 = desired_output1 - new_output1
        new_error_list = (new_error0, new_error1)

        # Compute the new Summed Squared Error, or SSE
        sse0 = new_error0 ** 2
        sse1 = new_error1 ** 2
        new_sse = sse0 + sse1

        # Print the Summed Squared Error

        # Debug print the actual outputs from the two output neurons
        if not debug_main_compute_forward_pass_outputs_off:
            print(' ')
            print('In main; have just completed a single step of back propagation with inputs', input0, input1)
            print('    The new SSE (after back propagation) = %.6f' % new_sse)
            print('    Error(0) = %.4f,   Error(1) = %.4f' % (new_error0, new_error1))
            print('    SSE(0) =   %.4f,   SSE(1) =   %.4f' % (sse0, sse1))
            delta_sse = sse_initial - new_sse
            print('  The difference in initial and the resulting SSEs is: %.4f' % delta_sse)
            if delta_sse > 0:
                print(' ')
                print('   The training has resulted in improving the total SSEs')

            # Assign the SSE to the SSE for the appropriate training set
        sse_array[set_number] = new_sse

        # Obtain the previous SSE Total from the SSE array
        previous_sse_total = sse_array[4]
        print(' ')
        print('  The previous SSE Total was %.4f' % previous_sse_total)

        # Compute the new sum of SSEs (across all the different training sets)
        #   ... this will be different because we've changed one of the SSE's
        new_sse_total = sse_array[0] + sse_array[1] + sse_array[2] + sse_array[3]
        print('  The new SSE Total was %.4f' % new_sse_total)
        print('    For node 0: Desired Output = ', desired_output0, ' New Output = %.4f' % new_output0)
        print('    For node 1: Desired Output = ', desired_output1, ' New Output = %.4f' % new_output1)
        print('    Error(0) = %.4f,   Error(1) = %.4f' % (new_error0, new_error1))
        print('    sse0(0) =   %.4f,   SSE(1) =   %.4f' % (sse0, sse1))
        # Assign the new SSE to the final place in the SSE array
        sse_array[4] = new_sse_total
        delta_sse = previous_sse_total - new_sse_total
        print('  Delta in the SSEs is %.4f' % delta_sse)
        if delta_sse > 0:
            print('SSE improvement')
        else:
            print('NO improvement')

        print(' ')
        print('Iteration number ', iteration)
        iteration = iteration + 1

        if new_sse_total < epsilon:
            break
    print('Out of while loop')

    debug_ending_sse_comparison_off = False
    if not debug_ending_sse_comparison_off:
        sse_array[4] = new_sse_total
        print('  Initial Total SSE = %.4f' % sse_initial_total)
        print('  Final Total SSE = %.4f' % new_sse_total)
        final_delta_sse = sse_initial_total - new_sse_total
        print('  Delta in the SSEs is %.4f' % final_delta_sse)
        if final_delta_sse > 0:
            print('SSE total improvement')
        else:
            print('NO improvement in total SSE')


if __name__ == "__main__": main()
