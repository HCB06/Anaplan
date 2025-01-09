# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN

PLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf
ANAPLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj/
@contact: tchasancan@gmail.com
"""

import numpy as np
from colorama import Fore, Style
import sys

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .data_operations import normalization, decode_one_hot, batcher
from .loss_functions import binary_crossentropy, categorical_crossentropy
from .activation_functions import apply_activation, Softmax, all_activations
from .metrics import metrics
from.model_operations import get_acc, get_preds, get_preds_softmax
from .visualizations import (
    draw_neural_web,
    plot_evaluate,
    neuron_history,
    initialize_visualization_for_fit,
    update_weight_visualization_for_fit,
    update_decision_boundary_for_fit,
    update_validation_history_for_fit,
    display_visualization_for_fit,
    display_visualizations_for_learner,
    update_history_plots_for_learner,
    initialize_visualization_for_learner
)

### GLOBAL VARIABLES ###
bar_format_normal = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----

def fit(
    x_train,
    y_train,
    val=False,
    val_count=None,
    activation_potentiation=['linear'],
    x_val=None,
    y_val=None,
    show_training=None,
    interval=100,
    LTD=0,
    decision_boundary_status=True,
    train_bar=True,
    auto_normalization=True,
    neurons_history=False
):
    """
    Creates a model to fitting data.
    
    fit Args:

        x_train (list[num]): List or numarray of input data.

        y_train (list[num]): List or numarray of target labels. (one hot encoded)

        val (None or True): validation in training process ? None or True default: None (optional)

        val_count (None or int): After how many examples learned will an accuracy test be performed? default: 10=(%10) it means every approximately 10 step (optional)

        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: [None] (optional)

        x_val (list[num]): List of validation data. default: x_train (optional)

        y_val (list[num]): (list[num]): List of target labels. (one hot encoded) default: y_train (optional)

        show_training (bool, str): True or None default: None (optional)

        LTD (int): Long Term Depression Hyperparameter for train PLAN neural network default: 0 (optional)

        interval (float, int): frame delay (milisecond) parameter for Training Report (show_training=True) This parameter effects to your Training Report performance. Lower value is more diffucult for Low end PC's (33.33 = 30 FPS, 16.67 = 60 FPS) default: 100 (optional)

        decision_boundary_status (bool): If the visualization of validation and training history is enabled during training, should the decision boundaries also be visualized? True or False. Default is True. (optional)

        train_bar (bool): Training loading bar? True or False. Default is True. (optional)

        auto_normalization(bool): Normalization process during training. May effect training time and model quality. True or False. Default is True. (optional)

        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the CL (Cumulative Learning) stages. True or False. Default is False. (optional)

    Returns:
        numpyarray([num]): (Weight matrix).
    """
    # Pre-checks

    if train_bar and val:
        train_progress = initialize_loading_bar(total=len(x_train), ncols=71, desc='Fitting', bar_format=bar_format_normal)
    elif train_bar and val == False:
        train_progress = initialize_loading_bar(total=len(x_train), ncols=44, desc='Fitting', bar_format=bar_format_normal)

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have the same length.")

    if val and (x_val is None or y_val is None):
        x_val, y_val = x_train, y_train

    val_list = [] if val else None
    val_count = val_count or 10
    # Defining weights
    STPW = np.ones((len(y_train[0]), len(x_train[0].ravel())))  # STPW = SHORT TIME POTENTIATION WEIGHT
    LTPW = np.zeros((len(y_train[0]), len(x_train[0].ravel())))  # LTPW = LONG TIME POTENTIATION WEIGHT
    # Initialize visualization
    vis_objects = initialize_visualization_for_fit(val, show_training, neurons_history, x_train, y_train)

    # Training process
    for index, inp in enumerate(x_train):
        inp = np.array(inp).ravel()
        y_decoded = decode_one_hot(y_train)
        # Weight updates
        STPW = feed_forward(inp, STPW, is_training=True, Class=y_decoded[index], activation_potentiation=activation_potentiation, LTD=LTD)
        LTPW += normalization(STPW) if auto_normalization else STPW

        # Visualization updates
        if show_training:
            update_weight_visualization_for_fit(vis_objects['ax'][0, 0], LTPW, vis_objects['artist2'])
            if decision_boundary_status:
                update_decision_boundary_for_fit(vis_objects['ax'][0, 1], x_val, y_val, activation_potentiation, LTPW, vis_objects['artist1'])
            update_validation_history_for_fit(vis_objects['ax'][1, 1], val_list, vis_objects['artist3'])
        if train_bar:
             train_progress.update(1)

        STPW = np.ones((len(y_train[0]), len(x_train[0].ravel())))

    # Finalize visualization
    if show_training:
        display_visualization_for_fit(vis_objects['fig'], vis_objects['artist1'], interval)

    return normalization(LTPW)


def learner(x_train, y_train, x_test=None, y_test=None, strategy='accuracy', batch_size=1,
           neural_web_history=False, show_current_activations=False, auto_normalization=True,
           neurons_history=False, patience=None, depth=None, early_shifting=False,
           early_stop=False, loss='categorical_crossentropy', show_history=False,
           interval=33.33, target_acc=None, target_loss=None, except_this=None,
           only_this=None, start_this=None):
    """
    Optimizes the activation functions for a neural network by leveraging train data to find 
    the most accurate combination of activation potentiation for the given dataset.
    
    Args:

        x_train (array-like): Training input data.

        y_train (array-like): Labels for training data.

        x_test (array-like, optional): Test input data (for improve next gen generilization). If test data is not given then train feedback learning active

        y_test (array-like, optional): Test Labels (for improve next gen generilization). If test data is not given then train feedback learning active

        strategy (str, optional): Learning strategy. (options: 'accuracy', 'loss', 'f1', 'precision', 'recall', 'adaptive_accuracy', 'adaptive_loss', 'all'): 'accuracy', Maximizes test accuracy during learning. 'f1', Maximizes test f1 score during learning. 'precision', Maximizes test preciison score during learning. 'recall', Maximizes test recall during learning. loss', Minimizes test loss during learning. 'adaptive_accuracy', The model compares the current accuracy with the accuracy from the past based on the number specified by the patience value. If no improvement is observed it adapts to the condition by switching to the 'loss' strategy quickly starts minimizing loss and continues learning. 'adaptive_loss',The model adopts the 'loss' strategy until the loss reaches or falls below the value specified by the patience parameter. However, when the patience threshold is reached, it automatically switches to the 'accuracy' strategy and begins to maximize accuracy. 'all', Maximizes all test scores and minimizes test loss, 'all' strategy most strong and most robust strategy. Default is 'accuracy'.

        patience ((int, float), optional): patience value for adaptive strategies. For 'adaptive_accuracy' Default value: 5. For 'adaptive_loss' Default value: 0.150.

        depth (int, optional): The depth of the PLAN neural networks Aggreagation layers.

        batch_size (float, optional): Batch size is used in the prediction process to receive test feedback by dividing the test data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire test set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each test batch represents 8% of the test set. Default is 1. (%100 of test)

        auto_normalization (bool, optional): If auto normalization=False this makes more faster training times and much better accuracy performance for some datasets. Default is True.

        early_shifting (int, optional): Early shifting checks if the test accuracy improves after a given number of activation attempts while inside a depth. If there's no improvement, it automatically shifts to the next depth. Basically, if no progress, it’s like, “Alright, let’s move on!”  Default is False

        early_stop (bool, optional): If True, implements early stopping during training.(If test accuracy not improves in two depth stops learning.) Default is False.

        show_current_activations (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False

        show_history (bool, optional): If True, displays the training history after optimization. Default is False.

        loss (str, optional): For visualizing and monitoring. PLAN neural networks doesn't need any loss function in training(if strategy not 'loss'). options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.

        interval (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.

        target_acc (int, optional): The target accuracy to stop training early when achieved. Default is None.

        target_loss (float, optional): The target loss to stop training early when achieved. Default is None.

        except_this (list, optional): A list of activations to exclude from optimization. Default is None. (For avaliable activation functions, run this code: plan.activations_list())

        only_this (list, optional): A list of activations to focus on during optimization. Default is None. (For avaliable activation functions, run this code: plan.activations_list())
        
        start_this (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None

        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the TFL (Test or Train Feedback Learning) stages. True or False. Default is False.

        neural_web_history (bool, optional): Draws history of neural web. Default is False.

    Returns:
        tuple: A list for model parameters: [Weight matrix, Test loss, Test Accuracy, [Activations functions]].
    
    """
    print(Fore.WHITE + "\nRemember, optimization on large datasets can be very time-consuming and computationally expensive. Therefore, if you are working with such a dataset, our recommendation is to include activation function: ['circular'] in the 'except_this' parameter unless absolutely necessary, as they can significantly prolong the process. from: learner\n" + Fore.RESET)

    activation_potentiation = all_activations()

    if x_test is None and y_test is None:
        x_test = x_train
        y_test = y_train
        data = 'Train'
    else:
        data = 'Test'

    if early_shifting != False:
        shift_patience = early_shifting

    # Strategy initialization
    if strategy == 'adaptive_accuracy':
        strategy = 'accuracy'
        adaptive = True
        if patience == None:
            patience = 5
    elif strategy == 'adaptive_loss':
        strategy = 'loss'
        adaptive = True
        if patience == None:
            patience = 0.150
    else:
        adaptive = False

    # Filter activation functions
    if only_this != None:
        activation_potentiation = only_this
    if except_this != None:
        activation_potentiation = [item for item in activation_potentiation if item not in except_this]
    if depth is None:
        depth = len(activation_potentiation)

    # Initialize visualization components
    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)

    # Initialize progress bar
    if batch_size == 1:
        ncols = 76
    else:
        ncols = 89
    progress = initialize_loading_bar(total=len(activation_potentiation), desc="", ncols=ncols, bar_format=bar_format_learner)

    # Initialize variables
    activations = []
    if start_this is None:
        best_activations = []
        best_acc = 0
    else:
        best_activations = start_this
        x_test_batch, y_test_batch = batcher(x_test, y_test, batch_size=batch_size)
        W = fit(x_train, y_train, activation_potentiation=best_activations, train_bar=False, auto_normalization=auto_normalization)
        model = evaluate(x_test_batch, y_test_batch, W=W, loading_bar_status=False, activation_potentiation=activations)

        if loss == 'categorical_crossentropy':
            test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])
        else:
            test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])

        best_acc = model[get_acc()]
        best_loss = test_loss

    best_acc_per_depth_list = []
    postfix_dict = {}
    loss_list = []
    
    for i in range(depth):
        postfix_dict["Depth"] = str(i+1) + '/' + str(depth)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(len(activation_potentiation)):
            for k in range(len(best_activations)):
                activations.append(best_activations[k])

            activations.append(activation_potentiation[j])

            x_test_batch, y_test_batch = batcher(x_test, y_test, batch_size=batch_size)
            W = fit(x_train, y_train, activation_potentiation=activations, train_bar=False, auto_normalization=auto_normalization)
            model = evaluate(x_test_batch, y_test_batch, W=W, loading_bar_status=False, activation_potentiation=activations)

            acc = model[get_acc()]

            if strategy == 'loss' or strategy == 'all':
                if loss == 'categorical_crossentropy':
                    test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])
                else:
                    test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])

                if i == 0 and j == 0 and start_this is None:
                    best_loss = test_loss

            if strategy == 'f1' or strategy == 'precision' or strategy == 'recall' or strategy == 'all':
                precision_score, recall_score, f1_score = metrics(y_test_batch, model[get_preds()])

                if strategy == 'precision' or strategy == 'all':
                    if i == 0 and j == 0:
                        best_precision = precision_score

                if strategy == 'recall' or strategy == 'all':
                    if i == 0 and j == 0:
                        best_recall = recall_score

                if strategy == 'f1' or strategy == 'all':
                    if i == 0 and j == 0:
                        best_f1 = f1_score

            if early_shifting != False:
                if acc <= best_acc:
                    early_shifting -= 1
                    if early_shifting == 0:
                        early_shifting = shift_patience
                        break

            if ((strategy == 'accuracy' and acc >= best_acc) or 
                (strategy == 'loss' and test_loss <= best_loss) or 
                (strategy == 'f1' and f1_score >= best_f1) or 
                (strategy == 'precision' and precision_score >= best_precision) or 
                (strategy == 'recall' and recall_score >= best_recall) or 
                (strategy == 'all' and f1_score >= best_f1 and acc >= best_acc and 
                 test_loss <= best_loss and precision_score >= best_precision and 
                 recall_score >= best_recall)):

                current_best_activation = activation_potentiation[j]
                best_acc = acc
                
                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = best_acc
                else:
                    postfix_dict[f"{data} Batch Accuracy"] = acc
                progress.set_postfix(postfix_dict)
                
                final_activations = activations

                if show_current_activations:
                    print(f", Current Activations={final_activations}", end='')

                best_weights = W
                best_model = model

                if strategy != 'loss':
                    if loss == 'categorical_crossentropy':
                        test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])
                    else:
                        test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_softmax()])

                if batch_size == 1:
                    postfix_dict[f"{data} Loss"] = test_loss
                    best_loss = test_loss
                else:
                    postfix_dict[f"{data} Batch Loss"] = test_loss
                progress.set_postfix(postfix_dict)
                best_loss = test_loss

                # Update visualizations during training
                if show_history:
                    depth_list = range(1, len(best_acc_per_depth_list) + 2)
                    update_history_plots_for_learner(viz_objects, depth_list, loss_list + [test_loss], 
                                      best_acc_per_depth_list + [best_acc], x_train, final_activations)

                if neurons_history:
                    viz_objects['neurons']['artists'] = (
                        neuron_history(np.copy(best_weights), viz_objects['neurons']['ax'],
                                     viz_objects['neurons']['row'], viz_objects['neurons']['col'],
                                     y_train[0], viz_objects['neurons']['artists'],
                                     data=data, fig1=viz_objects['neurons']['fig'],
                                     acc=best_acc, loss=test_loss)
                    )

                if neural_web_history:
                    art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=viz_objects['web']['ax'],
                                                            G=viz_objects['web']['G'], return_objs=True)
                    art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                    viz_objects['web']['artists'].append(art5_list)

                # Check target accuracy
                if target_acc is not None and best_acc >= target_acc:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                        activation_potentiation=final_activations)

                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                    print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                    if data == 'Test':
                        print(f'Test Accuracy (%{batch_size * 100} of test samples): ', best_acc)
                        print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              test_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

                # Check target loss
                if target_loss is not None and best_loss <= target_loss:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                        activation_potentiation=final_activations)

                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                    print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                    if data == 'Test':
                        print(f'Test Accuracy (%{batch_size * 100} of test samples): ', best_acc)
                        print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              test_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

            progress.update(1)
            activations = []

        best_activations.append(current_best_activation)
        best_acc_per_depth_list.append(best_acc)
        loss_list.append(best_loss)

        # Check adaptive strategy conditions
        if adaptive == True and strategy == 'accuracy' and i + 1 >= patience:
            check = best_acc_per_depth_list[-patience:]
            if all(x == check[0] for x in check):
                strategy = 'loss'
                adaptive = False
        elif adaptive == True and strategy == 'loss' and best_loss <= patience:
            strategy = 'accuracy'
            adaptive = False

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_depth_list[i] == best_acc_per_depth_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                    activation_potentiation=final_activations)

                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                   y_pred_batch=train_model[get_preds_softmax()])

                print('\nActivations: ', final_activations)
                print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                if data == 'Test':
                    print(f'Test Accuracy (%{batch_size * 100} of test samples): ', best_acc)
                    print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

                # Display final visualizations
                display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                          test_loss, y_train, interval)
                return best_weights, best_model[get_preds()], best_acc, final_activations

    # Final evaluation
    progress.close()
    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                        activation_potentiation=final_activations)

    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])

    print('\nActivations: ', final_activations)
    print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
    print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
    if data == 'Test':
        print(f'Test Accuracy (%{batch_size * 100} of test samples): ', best_acc)
        print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

    # Display final visualizations
    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, test_loss, y_train, interval)
    return best_weights, best_model[get_preds()], best_acc, final_activations



def feed_forward(
    Input,               # list[num]: Input data.
    w,                   # num: Weight matrix of the neural network.
    is_training,        # bool: Flag indicating if the function is called during training (True or False).
    activation_potentiation,
    Class='?',               # int: Which class is, if training. # (list): Activation potentiation list for deep PLAN. (optional)
    LTD=0
) -> tuple:
    """
    Applies feature extraction process to the input data using synaptic potentiation.

    Args:
        Input (num): Input data.
        w (num): Weight matrix of the neural network.
        is_training (bool): Flag indicating if the function is called during training (True or False).
        Class (int): if is during training then which class(label) ? is isnt then put None.
        # activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)

    Returns:
        tuple: A tuple (vector) containing the neural layer result and the updated weight matrix.
        or
        num: neural network output
    """
    
    Output = apply_activation(Input, activation_potentiation)

    Input = Output

    if is_training == True:
        
        for _ in range(LTD):        

            depression_vector = np.random.rand(*Input.shape)

            Input -= depression_vector

        w[Class, :] = Input
        return w

    else:

        neural_layer = np.dot(w, Input)

        return neural_layer


def evaluate(
    x_test,         # list[num]: Test input data.
    y_test,         # list[num]: Test labels.
    W,               # list[num]: Weight matrix list of the neural network.
    activation_potentiation=['linear'], # (list): Activation potentiation list for deep PLAN. (optional)
    loading_bar_status=True,  # bar_status (bool): Loading bar for accuracy (True or None) (optional) Default: True
    show_metrics=None, # show_metrices (bool): (True or None) (optional) Default: None   
) -> tuple:
    """
    Tests the neural network model with the given test data.

    Args:
    
        x_test (list[num]): Test input data.

        y_test (list[num]): Test labels.

        W (list[num]): Weight matrix list of the neural network.

        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: [None]

        loading_bar_status:  Evaluate progress have a loading bar ? (True or False) Default: True.

        show_metrics (bool): (True or None) (optional) Default: None

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy of the model.
    """
    predict_probabilitys = []
    real_classes = []
    predict_classes = []

    try:

        true_predict = 0
        y_preds = []
        y_preds_raw = []
        acc_list = []

        
        if loading_bar_status == True:

            loading_bar = initialize_loading_bar(total=len(x_test), ncols=64, desc='Testing', bar_format=bar_format_normal)

        for inpIndex, Input in enumerate(x_test):
            Input = np.array(Input)
            Input = Input.ravel()
            neural_layer = Input

                
            neural_layer = feed_forward(neural_layer, W, is_training=False, Class='?', activation_potentiation=activation_potentiation)
            neural_layer = Softmax(neural_layer)

            max_value = max(neural_layer)

            predict_probabilitys.append(max_value)
            
            RealOutput = np.argmax(y_test[inpIndex])
            real_classes.append(RealOutput)
            PredictedOutput = np.argmax(neural_layer)
            predict_classes.append(PredictedOutput)

            if RealOutput == PredictedOutput:
                true_predict += 1
            acc = true_predict / len(y_test)


            acc_list.append(acc)
            y_preds.append(PredictedOutput)
            y_preds_raw.append(Softmax(neural_layer))

            if loading_bar_status == True:
                loading_bar.update(1)
                
            if inpIndex > 0 and loading_bar_status == True:
                loading_bar.set_postfix({"Test Accuracy": acc})

        if show_metrics == True:
            
            loading_bar.close()
            plot_evaluate(x_test, y_test, y_preds, acc_list, W=np.copy(W), activation_potentiation=activation_potentiation)

    except Exception as e:

        raise(e)

    return W, y_preds, acc, None, None, y_preds_raw