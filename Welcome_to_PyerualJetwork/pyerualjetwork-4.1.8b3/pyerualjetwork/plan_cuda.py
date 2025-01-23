# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN_CUDA

PLAN document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
PYERUALJETWORK document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj/
@contact: tchasancan@gmail.com
"""

import cupy as cp
from colorama import Fore
import math

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .data_operations_cuda import normalization, decode_one_hot, batcher
from .loss_functions_cuda import binary_crossentropy, categorical_crossentropy
from .activation_functions_cuda import apply_activation, Softmax, all_activations
from .metrics_cuda import metrics
from .model_operations_cuda import get_acc, get_preds, get_preds_softmax
from .memory_operations import transfer_to_gpu, transfer_to_cpu
from .visualizations_cuda import (
    draw_neural_web,
    update_neural_web_for_fit,
    plot_evaluate,
    update_neuron_history,
    initialize_visualization_for_fit,
    update_weight_visualization_for_fit,
    update_decision_boundary_for_fit,
    update_validation_history_for_fit,
    display_visualization_for_fit,
    display_visualizations_for_learner,
    update_history_plots_for_learner,
    initialize_visualization_for_learner,
    update_neuron_history_for_learner,
    show
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
    neurons_history=False,
    dtype=cp.float32,
    memory='gpu'
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

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

        memory (str): The memory parameter determines whether the dataset to be processed on the GPU will be stored in the CPU's RAM or the GPU's RAM. Options:  'gpu', 'cpu'. Default: 'gpu'.

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

    if memory == 'gpu':
        x_train = transfer_to_gpu(x_train, dtype=dtype)
        y_train = transfer_to_gpu(y_train, dtype=y_train.dtype)

        if val:
            x_val = transfer_to_gpu(x_val, dtype=dtype)
            y_val = transfer_to_gpu(y_val, dtype=y_train.dtype)

    elif memory == 'cpu':
        x_train = transfer_to_cpu(x_train, dtype=dtype)
        y_train = transfer_to_cpu(y_train, dtype=y_train.dtype)

        if val:
            x_val = transfer_to_cpu(x_val, dtype=dtype)
            y_val = transfer_to_cpu(y_val, dtype=y_train.dtype)

    else:
        raise ValueError("memory parameter must be 'cpu' or 'gpu'.")

    val_list = [] if val else None
    val_count = val_count or 10
    # Defining weights
    STPW = cp.ones((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)  # STPW = SHORT TERM POTENTIATION WEIGHT
    LTPW = cp.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)  # LTPW = LONG TERM POTENTIATION WEIGHT
    # Initialize visualization
    vis_objects = initialize_visualization_for_fit(val, show_training, neurons_history, x_train, y_train)
   
    # Training process
    for index, inp in enumerate(x_train):
        inp = transfer_to_gpu(inp, dtype=dtype).ravel()
        y_decoded = decode_one_hot(cp.array(y_train[index], copy=False, dtype=y_train.dtype))
        # Weight updates
        STPW = feed_forward(inp, STPW, is_training=True, Class=y_decoded, activation_potentiation=activation_potentiation, LTD=LTD)
        LTPW += normalization(STPW, dtype=dtype) if auto_normalization else STPW

        if val and index != 0:
            if index % math.ceil((val_count / len(x_train)) * 100) == 0:
                val_acc = evaluate(x_val, y_val, loading_bar_status=False, activation_potentiation=activation_potentiation, W=LTPW, memory=memory)[get_acc()]
                val_list.append(val_acc)

                # Visualization updates
                if show_training:
                    update_weight_visualization_for_fit(vis_objects['ax'][0, 0], LTPW, vis_objects['artist2'])
                    if decision_boundary_status:
                        update_decision_boundary_for_fit(vis_objects['ax'][0, 1], x_val, y_val, activation_potentiation, LTPW, vis_objects['artist1'])
                    update_validation_history_for_fit(vis_objects['ax'][1, 1], val_list, vis_objects['artist3'])
                    update_neural_web_for_fit(W=LTPW, G=vis_objects['G'], ax=vis_objects['ax'][1, 0], artist=vis_objects['artist4'])
                if neurons_history:
                    update_neuron_history(LTPW, row=vis_objects['row'], col=vis_objects['col'], class_count=len(y_train[0]), fig1=vis_objects['fig1'], ax1=vis_objects['ax1'], artist5=vis_objects['artist5'], acc=val_acc)
        if train_bar:
             train_progress.update(1)

        STPW = cp.ones((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False)

    if show_training:
        ani1 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist1'], interval)
        ani2 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist2'], interval)
        ani3 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist3'], interval)
        ani4 = display_visualization_for_fit(vis_objects['fig'], vis_objects['artist4'], interval)
        show()
    
    if neurons_history:
        ani5 = display_visualization_for_fit(vis_objects['fig1'], vis_objects['artist5'], interval)
        show()

    return normalization(LTPW, dtype=dtype)


def learner(x_train, y_train, optimizator, x_test=None, y_test=None, strategy='accuracy', gen=None, batch_size=1,
           neural_web_history=False, show_current_activations=False, auto_normalization=True,
           neurons_history=False, early_stop=False, loss='categorical_crossentropy', show_history=False,
           interval=33.33, target_acc=None, target_loss=None, except_this=None,
           only_this=None, start_this_act=None, start_this_W=None, target_fitness='max', dtype=cp.float32, memory='gpu'):
    """
    Optimizes the activation functions for a neural network by leveraging train data to find 
    the most accurate combination of activation potentiation for the given dataset.
    
    Why genetic optimization and not backpropagation?
    Because PLAN is different from other neural network architectures. In PLAN, the learnable parameters are not the weights; instead, the learnable parameters are the activation functions.
    Since activation functions are not differentiable, we cannot use gradient descent or backpropagation. However, I developed a more powerful genetic optimization algorithm: PLANEAT.

    Args:

        x_train (array-like): Training input data.

        y_train (array-like): Labels for training data.

        optimizator (function): PLAN optimization technique with hyperparameters. (PLAN using NEAT(PLANEAT) for optimization.) Please use this: from pyerualjetwork import planeat_cuda (and) optimizator = lambda *args, **kwargs: planeat_cuda.evolve(*args, 'here give your neat hyperparameters for example:  activation_add_prob=0.85', **kwargs) Example:
        ```python
        genetic_optimizator = lambda *args, **kwargs: planeat_cuda.evolve(*args,
                                                                    activation_add_prob=0.85,
                                                                    mutations=False,
                                                                    strategy='cross_over',
                                                                    **kwargs)

        model = plan_cuda.learner(x_train,
                             y_train,
                             optimizator=genetic_optimizator,
                             strategy='accuracy',
                             show_history=True,
                             target_acc=0.94,
                             interval=16.67)
        ```

        x_test (array-like, optional): Test input data (for improve next gen generilization). If test data is not given then train feedback learning active

        y_test (array-like, optional): Test Labels (for improve next gen generilization). If test data is not given then train feedback learning active

        strategy (str, optional): Learning strategy. (options: 'accuracy', 'f1', 'precision', 'recall'): 'accuracy', Maximizes train (or test if given) accuracy during learning. 'f1', Maximizes train (or test if given) f1 score during learning. 'precision', Maximizes train (or test if given) precision score during learning. 'recall', Maximizes train (or test if given) recall during learning. Default is 'accuracy'.

        patience ((int, float), optional): patience value for adaptive strategies. For 'adaptive_accuracy' Default value: 5. For 'adaptive_loss' Default value: 0.150.

        gen (int, optional): The generation count for genetic optimization.

        batch_size (float, optional): Batch size is used in the prediction process to receive test feedback by dividing the test data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire test set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each test batch represents 8% of the test set. Default is 1. (%100 of test)

        auto_normalization (bool, optional): If auto normalization=False this makes more faster training times and much better accuracy performance for some datasets. Default is True.

        early_stop (bool, optional): If True, implements early stopping during training.(If test accuracy not improves in two gen stops learning.) Default is False.

        show_current_activations (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False

        show_history (bool, optional): If True, displays the training history after optimization. Default is False.

        loss (str, optional): For visualizing and monitoring. PLAN neural networks doesn't need any loss function in training. options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.

        interval (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.

        target_acc (int, optional): The target accuracy to stop training early when achieved. Default is None.

        target_loss (float, optional): The target loss to stop training early when achieved. Default is None.

        except_this (list, optional): A list of activations to exclude from optimization. Default is None. (For avaliable activation functions, run this code: plan.activations_list())

        only_this (list, optional): A list of activations to focus on during optimization. Default is None. (For avaliable activation functions, run this code: plan.activations_list())
        
        start_this_act (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None

        start_this_W (cupy.array, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the weight matrix of this genome. Default is None
        
        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the TFL (Test or Train Feedback Learning) stages. True or False. Default is False.

        neural_web_history (bool, optional): Draws history of neural web. Default is False.

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

        memory (str): The memory parameter determines whether the dataset to be processed on the GPU will be stored in the CPU's RAM or the GPU's RAM. Options:  'gpu', 'cpu'. Default: 'gpu'.

    Returns:
        tuple: A list for model parameters: [Weight matrix, Preds, Accuracy, [Activations functions]]. You can acces this parameters in model_operations module. For example: model_operations.get_weights() for Weight matrix.
    
    """

    print(Fore.WHITE + "\nRemember, optimization on large datasets can be very time-consuming and computationally expensive. Therefore, if you are working with such a dataset, our recommendation is to include activation function: ['circular', 'spiral'] in the 'except_this' parameter unless absolutely necessary, as they can significantly prolong the process. from: learner\n" + Fore.RESET)

    activation_potentiation = all_activations()

    if x_test is None and y_test is None:
        x_test = x_train
        y_test = y_train
        data = 'Train'
    else:
        data = 'Test'
        
    if memory == 'gpu':
        x_train = transfer_to_gpu(x_train, dtype=dtype)
        y_train = transfer_to_gpu(y_train, dtype=y_train.dtype)

        x_test = transfer_to_gpu(x_test, dtype=dtype)
        y_test = transfer_to_gpu(y_test, dtype=y_train.dtype)

        from .data_operations_cuda import batcher

    elif memory == 'cpu':
        x_train = transfer_to_cpu(x_train, dtype=dtype)
        y_train = transfer_to_cpu(y_train, dtype=y_train.dtype)

        x_test = transfer_to_cpu(x_test, dtype=dtype)
        y_test = transfer_to_cpu(y_test, dtype=y_train.dtype)

        from .data_operations import batcher

    else:
        raise ValueError("memory parameter must be 'cpu' or 'gpu'.")

    # Filter activation functions
    if only_this != None:
        activation_potentiation = only_this
    if except_this != None:
        activation_potentiation = [item for item in activation_potentiation if item not in except_this]
    if gen is None:
        gen = len(activation_potentiation)

    # Initialize visualization components
    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)

    # Initialize progress bar
    if batch_size == 1:
        ncols = 76
    else:
        ncols = 89

    # Initialize variables
    act_pop = []
    weight_pop = []
    
    if start_this_act is None and start_this_W is None:
        best_acc = 0
    else:
        act_pop.append(start_this_act)
        weight_pop.append(start_this_W)
        
        x_test_batch, y_test_batch = batcher(x_test, y_test, batch_size=batch_size)
        
        model = evaluate(x_test_batch, y_test_batch, W=start_this_W, loading_bar_status=False, activation_potentiation=act_pop, dtype=dtype, memory=memory)

        if loss == 'categorical_crossentropy':
            test_loss = categorical_crossentropy(y_true_batch=transfer_to_gpu(y_test_batch, dtype=y_test_batch.dtype), y_pred_batch=model[get_preds_softmax()])
        else:
            test_loss = binary_crossentropy(y_true_batch=transfer_to_gpu(y_test_batch, dtype=y_test_batch.dtype), y_pred_batch=model[get_preds_softmax()])

        best_acc = model[get_acc()]
        best_loss = test_loss

    best_acc_per_gen_list = []
    postfix_dict = {}
    loss_list = []
    target_pop = []

    for i in range(len(activation_potentiation)):
        print(f"\rPre-Run {i}/{len(activation_potentiation)}",end='')

        x_test_batch, y_test_batch = batcher(x_test, y_test, batch_size=batch_size)
        W = fit(x_train, y_train, activation_potentiation=act_pop[-1], train_bar=False, auto_normalization=auto_normalization, dtype=dtype, memory=memory)
        model = evaluate(x_test_batch, y_test_batch, W=W, loading_bar_status=False, activation_potentiation=act_pop[-1], dtype=dtype, memory=memory)

        if i == 0 and start_this_act is not None:
            act_pop[0] = activation_potentiation[i]
            weight_pop[0] = W
        else:
            act_pop.append(activation_potentiation[i])
            weight_pop.append(W)

        if strategy == 'accuracy': target_pop.append(model[get_acc()])
        elif strategy == 'f1' or strategy == 'precision' or strategy == 'recall':
            precision_score, recall_score, f1_score = metrics(y_test_batch, model[get_preds()])
            
            if strategy == 'precision': target_pop.append(precision_score)
            if strategy == 'recall': target_pop.append(recall_score)
            if strategy == 'f1': target_pop.append(f1_score)

        

    weight_pop, act_pop = optimizator(cp.array(weight_pop, dtype=dtype), act_pop, 0, cp.array(target_pop, dtype=dtype), target_fitness=target_fitness, bar_status=False)
    progress = initialize_loading_bar(total=len(activation_potentiation), desc="", ncols=ncols, bar_format=bar_format_learner)

    for i in range(gen):
        postfix_dict["Gen"] = str(i+1) + '/' + str(gen)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(len(activation_potentiation)):

            x_test_batch, y_test_batch = batcher(x_test, y_test, batch_size=batch_size)
            model = evaluate(x_test_batch, y_test_batch, W=weight_pop[j], loading_bar_status=False, activation_potentiation=act_pop[j], dtype=dtype, memory=memory)

            acc = model[get_acc()]
            if strategy == 'accuracy': target_pop.append(acc)

            elif strategy == 'f1' or strategy == 'precision' or strategy == 'recall':
                precision_score, recall_score, f1_score = metrics(y_test_batch, model[get_preds()])

                if strategy == 'precision':
                    target_pop.append(precision_score)

                    if i == 0 and j == 0:
                        best_precision = precision_score

                if strategy == 'recall':
                    target_pop.append(recall_score)

                    if i == 0 and j == 0:
                        best_recall = recall_score

                if strategy == 'f1':
                    target_pop.append(f1_score)

                    if i == 0 and j == 0:
                        best_f1 = f1_score

            if ((strategy == 'accuracy' and acc >= best_acc) or 
                (strategy == 'f1' and f1_score >= best_f1) or 
                (strategy == 'precision' and precision_score >= best_precision) or 
                (strategy == 'recall' and recall_score >= best_recall)):

                best_acc = acc
                best_weights = weight_pop[j]
                final_activations = act_pop[j]
                best_model = model

                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = best_acc
                else:
                    postfix_dict[f"{data} Batch Accuracy"] = acc
                progress.set_postfix(postfix_dict)

                if show_current_activations:
                    print(f", Current Activations={final_activations}", end='')
            
                if loss == 'categorical_crossentropy':
                    test_loss = categorical_crossentropy(y_true_batch=transfer_to_gpu(y_test_batch, dtype=y_test_batch.dtype), y_pred_batch=model[get_preds_softmax()])
                else:
                    test_loss = binary_crossentropy(y_true_batch=transfer_to_gpu(y_test_batch, dtype=y_test_batch.dtype), y_pred_batch=model[get_preds_softmax()])

                if batch_size == 1:
                    postfix_dict[f"{data} Loss"] = test_loss
                    best_loss = test_loss
                else:
                    postfix_dict[f"{data} Batch Loss"] = test_loss
                progress.set_postfix(postfix_dict)
                best_loss = test_loss

                # Update visualizations during training
                if show_history:
                    gen_list = range(1, len(best_acc_per_gen_list) + 2)
                    update_history_plots_for_learner(viz_objects, gen_list, loss_list + [test_loss], 
                                      best_acc_per_gen_list + [best_acc], x_train, final_activations)

                if neurons_history:
                    viz_objects['neurons']['artists'] = (
                        update_neuron_history_for_learner(cp.copy(best_weights), viz_objects['neurons']['ax'],
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
                                        activation_potentiation=final_activations, dtype=dtype)

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
                                        activation_potentiation=final_activations, dtype=dtype)

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

        best_acc_per_gen_list.append(best_acc)
        loss_list.append(best_loss)

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, 
                                    activation_potentiation=final_activations, dtype=dtype, memory=memory)

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
                        activation_potentiation=final_activations, dtype=dtype, memory=memory)

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

            depression_vector = cp.random.rand(*Input.shape)

            Input -= depression_vector

        w[Class, :] = Input
        return w

    else:

        neural_layer = cp.dot(w, Input)

        return neural_layer
   

def evaluate(
    x_test,
    y_test,
    W,
    activation_potentiation=['linear'],
    loading_bar_status=True,
    show_metrics=False,
    dtype=cp.float32,
    memory='gpu'
) -> tuple:
    """
    Evaluates the neural network model using the given test data.

    Args:
        x_test (cp.ndarray): Test data.

        y_test (cp.ndarray): Test labels (one-hot encoded).

        W (list[cp.ndarray]): Neural net weight matrix.
        
        activation_potentiation (list): Activation list. Default = ['linear'].
        
        loading_bar_status (bool): Loading bar (optional). Default = True.

        show_metrics (bool): Visualize metrics ? (optional). Default = False.

        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

        memory (str): The memory parameter determines whether the dataset to be processed on the GPU will be stored in the CPU's RAM or the GPU's RAM. Options:  'gpu', 'cpu'. Default: 'gpu'.

    Returns:
        tuple: Model (list).
    """
    from memory_operations import transfer_to_cpu, transfer_to_gpu

    if memory == 'gpu':
        x_test = transfer_to_gpu(x_test, dtype=dtype)
        y_test = transfer_to_gpu(y_test, dtype=y_test.dtype)

    elif memory == 'cpu':
        x_test = transfer_to_cpu(x_test, dtype=dtype)
        y_test = transfer_to_cpu(y_test, dtype=y_test.dtype)

    else:
        raise ValueError("memory parameter must be 'cpu' or 'gpu'.")

    predict_probabilitys = cp.empty((len(x_test), W.shape[0]), dtype=dtype)
    real_classes = cp.empty(len(x_test), dtype=y_test.dtype)   
    predict_classes = cp.empty(len(x_test), dtype=y_test.dtype)

    true_predict = 0
    acc_list = cp.empty(len(x_test), dtype=dtype)

    if loading_bar_status:
        loading_bar = initialize_loading_bar(total=len(x_test), ncols=64, desc='Testing', bar_format=bar_format_normal)

    for inpIndex in range(len(x_test)):
        Input = transfer_to_gpu(x_test[inpIndex], dtype=dtype).ravel()
        neural_layer = Input

        neural_layer = feed_forward(neural_layer, cp.copy(W), is_training=False, Class='?', activation_potentiation=activation_potentiation)

        predict_probabilitys[inpIndex] = Softmax(neural_layer)

        RealOutput = decode_one_hot(transfer_to_gpu(y_test[inpIndex], dtype=y_test[inpIndex].dtype))
        real_classes[inpIndex] = RealOutput
        PredictedOutput = cp.argmax(neural_layer)
        predict_classes[inpIndex] = PredictedOutput

        if RealOutput == PredictedOutput:
            true_predict += 1
        
        acc = true_predict / (inpIndex + 1)
        acc_list[inpIndex] = acc

        if loading_bar_status:
            loading_bar.update(1)
            loading_bar.set_postfix({"Test Accuracy": acc})

    if loading_bar_status:
        loading_bar.close()

    if show_metrics:
        plot_evaluate(x_test, y_test, predict_classes, acc_list, W=cp.copy(W), activation_potentiation=activation_potentiation)

    return W, predict_classes, acc_list[-1], None, None, predict_probabilitys