# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN_CUDA

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PLAN document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
PYERUALJETWORK document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydilj/
@contact: tchasancan@gmail.com
"""

import cupy as cp

### LIBRARY IMPORTS ###
from .ui import loading_bars, initialize_loading_bar
from .data_operations_cuda import normalization
from .loss_functions_cuda import binary_crossentropy, categorical_crossentropy
from .activation_functions_cuda import apply_activation, all_activations
from .metrics_cuda import metrics
from model_operations_cuda import get_acc, get_preds, get_preds_softmax
from .memory_operations import transfer_to_gpu, transfer_to_cpu, optimize_labels
from .visualizations_cuda import (
    draw_neural_web,
    display_visualizations_for_learner,
    update_history_plots_for_learner,
    initialize_visualization_for_learner,
    update_neuron_history_for_learner
)

### GLOBAL VARIABLES ###
bar_format_normal = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----

def fit(
    x_train,
    y_train,
    activation_potentiation=['linear'],
    W=None,
    dtype=cp.float32
):
    """
    Creates a model to fitting data.,
    
    fit Args:

        x_train (aray-like[cupy]): List or cupy array of input data.

        y_train (aray-like[cupy]): List or cupy array of target labels. (one hot encoded)

        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: [None] (optional)

        W (cupy.ndarray): If you want to re-continue or update model
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        cupyarray: (Weight matrix).
    """
    # Pre-check
    
    if len(x_train) != len(y_train): raise ValueError("x_train and y_train must have the same length.")

    LTPW = cp.zeros((len(y_train[0]), len(x_train[0].ravel()))).astype(dtype, copy=False) if W is None else W

    x_train = apply_activation(x_train, activation_potentiation)
    LTPW += cp.array(y_train, dtype=optimize_labels(y_train, cuda=True).dtype).T @ x_train

    return normalization(LTPW, dtype=dtype)


def learner(x_train, y_train, optimizer, fit_start, strategy='accuracy', gen=None, batch_size=1,
           neural_web_history=False, show_current_activations=False,
           neurons_history=False, early_stop=False, loss='categorical_crossentropy', show_history=False,
           interval=33.33, target_acc=None, target_loss=None,
           start_this_act=None, start_this_W=None, dtype=cp.float32, memory='gpu'):
    """
    Optimizes the activation functions for a neural network by leveraging train data to find 
    the most accurate combination of activation potentiation for the given dataset.
    
    Why genetic optimization and not backpropagation?
    Because PLAN is different from other neural network architectures. In PLAN, the learnable parameters are not the weights; instead, the learnable parameters are the activation functions.
    Since activation functions are not differentiable, we cannot use gradient descent or backpropagation. However, I developed a more powerful genetic optimization algorithm: PLANEAT.

    Args:

        x_train (array-like): Training input data.

        y_train (array-like): Labels for training data.

        optimizer (function): PLAN optimization technique with hyperparameters. (PLAN using NEAT(PLANEAT) for optimization.) Please use this: from pyerualjetwork import planeat_cuda (and) optimizer = lambda *args, **kwargs: planeat_cuda.evolve(*args, 'here give your neat hyperparameters for example:  activation_add_prob=0.85', **kwargs) Example:
        ```python
        genetic_optimizer = lambda *args, **kwargs: planeat_cuda.evolver(*args,
                                                                    activation_add_prob=0.85,
                                                                    strategy='aggressive',
                                                                    **kwargs)

        model = plan_cuda.learner(x_train,
                                y_train,
                                optimizer=genetic_optimizer,
                                fit_start=True,
                                strategy='accuracy',
                                show_history=True,
                                gen=15,
                                batch_size=0.05,
                                interval=16.67)
        ```
        fit_start (bool): If the fit_start parameter is set to True, the initial generation population undergoes a simple short training process using the PLAN algorithm. This allows for a very robust starting point, especially for large and complex datasets. However, for small or relatively simple datasets, it may result in unnecessary computational overhead. When fit_start is True, completing the first generation may take slightly longer (this increase in computational cost applies only to the first generation and does not affect subsequent generations). If fit_start is set to False, the initial population will be entirely random. Options: True or False. The fit_start parameter is MANDATORY and must be provided.
        
        strategy (str, optional): Learning strategy. (options: 'accuracy', 'f1', 'precision', 'recall'): 'accuracy', Maximizes train (or test if given) accuracy during learning. 'f1', Maximizes train (or test if given) f1 score during learning. 'precision', Maximizes train (or test if given) precision score during learning. 'recall', Maximizes train (or test if given) recall during learning. Default is 'accuracy'.

        gen (int, optional): The generation count for genetic optimization.

        batch_size (float, optional): Batch size is used in the prediction process to receive train feedback by dividing the train data into chunks and selecting activations based on randomly chosen partitions. This process reduces computational cost and time while still covering the entire test set due to random selection, so it doesn't significantly impact accuracy. For example, a batch size of 0.08 means each train batch represents 8% of the train set. Default is 1. (%100 of train)

        early_stop (bool, optional): If True, implements early stopping during training.(If train accuracy not improves in two gen stops learning.) Default is False.

        show_current_activations (bool, optional): Should it display the activations selected according to the current strategies during learning, or not? (True or False) This can be very useful if you want to cancel the learning process and resume from where you left off later. After canceling, you will need to view the live training activations in order to choose the activations to be given to the 'start_this' parameter. Default is False

        show_history (bool, optional): If True, displays the training history after optimization. Default is False.

        loss (str, optional): For visualizing and monitoring. PLAN neural networks doesn't need any loss function in training. options: ('categorical_crossentropy' or 'binary_crossentropy') Default is 'categorical_crossentropy'.

        interval (int, optional): The interval at which evaluations are conducted during training. (33.33 = 30 FPS, 16.67 = 60 FPS) Default is 100.

        target_acc (int, optional): The target accuracy to stop training early when achieved. Default is None.

        target_loss (float, optional): The target loss to stop training early when achieved. Default is None.
        
        start_this_act (list, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the list of activation functions selected up to the learned portion to this parameter. Default is None

        start_this_W (cupy.array, optional): To resume a previously canceled or interrupted training from where it left off, or to continue from that point with a different strategy, provide the weight matrix of this genome. Default is None
        
        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the TFL (Test or Train Feedback Learning) stages. True or False. Default is False.

        neural_web_history (bool, optional): Draws history of neural web. Default is False.

        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

        memory (str): The memory parameter determines whether the dataset to be processed on the GPU will be stored in the CPU's RAM or the GPU's RAM. Options:  'gpu', 'cpu'. Default: 'gpu'.

    Returns:
        tuple: A list for model parameters: [Weight matrix, Preds, Accuracy, [Activations functions]]. You can acces this parameters in model_operations module. For example: model_operations.get_weights() for Weight matrix.
    
    """

    from planeat_cuda import define_genomes

    data = 'Train'

    activation_potentiation = all_activations() 

    activation_potentiation_len = len(activation_potentiation)

    y_train = optimize_labels(y_train, cuda=True)

    if memory == 'gpu':
        x_train = transfer_to_gpu(x_train, dtype=dtype)
        y_train = transfer_to_gpu(y_train, dtype=y_train.dtype)

        from data_operations_cuda import batcher

    elif memory == 'cpu':
        x_train = transfer_to_cpu(x_train, dtype=dtype)
        y_train = transfer_to_cpu(y_train, dtype=y_train.dtype)

        from data_operations import batcher

    else:
        raise ValueError("memory parameter must be 'cpu' or 'gpu'.")

    if strategy != 'accuracy' and strategy != 'f1' and strategy != 'recall' and strategy != 'precision': raise ValueError("Strategy parameter only be 'accuracy' or 'f1' or 'recall' or 'precision'.")
    if target_acc is not None and (target_acc < 0 or target_acc > 1): raise ValueError('target_acc must be in range 0 and 1')
    if fit_start is not True and fit_start is not False: raise ValueError('fit_start parameter only be True or False. Please read doc-string')

    # Initialize visualization components
    viz_objects = initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train)

    # Initialize progress bar
    if batch_size == 1:
        ncols = 76
    else:
        ncols = 89

    # Initialize variables
    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0 
    best_acc_per_gen_list = []
    postfix_dict = {}
    loss_list = []
    target_pop = []

    progress = initialize_loading_bar(total=activation_potentiation_len, desc="", ncols=ncols, bar_format=bar_format_learner)

    if fit_start is False:
        weight_pop, act_pop = define_genomes(input_shape=len(x_train[0]), output_shape=len(y_train[0]), population_size=activation_potentiation_len, dtype=dtype)
    
        if start_this_act is not None and start_this_W is not None:
            weight_pop[0] = start_this_W
            act_pop[0] = start_this_act

    else:
        weight_pop = []
        act_pop = []

    for i in range(gen):
        postfix_dict["Gen"] = str(i+1) + '/' + str(gen)
        progress.set_postfix(postfix_dict)

        progress.n = 0
        progress.last_print_n = 0
        progress.update(0)

        for j in range(activation_potentiation_len):

            x_train_batch, y_train_batch = batcher(x_train, y_train, batch_size=batch_size)
 
            if fit_start is True and i == 0:
                act_pop.append(activation_potentiation[j])
                W = fit(x_train_batch, y_train_batch, activation_potentiation=act_pop[-1], dtype=dtype)
                weight_pop.append(W)
            
            model = evaluate(x_train_batch, y_train_batch, W=weight_pop[j], activation_potentiation=act_pop[j])
            acc = model[get_acc()]
            
            if strategy == 'accuracy': target_pop.append(acc)

            elif strategy == 'f1' or strategy == 'precision' or strategy == 'recall':
                precision_score, recall_score, f1_score = metrics(y_train_batch, model[get_preds()])

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
                best_weights = cp.copy(weight_pop[j])
                final_activations = act_pop[j].copy() if isinstance(act_pop[j], list) else act_pop[j]

                best_model = model

                final_activations = [final_activations[0]] if len(set(final_activations)) == 1 else final_activations # removing if all same

                if batch_size == 1:
                    postfix_dict[f"{data} Accuracy"] = cp.round(best_acc, 3)
                else:
                    postfix_dict[f"{data} Batch Accuracy"] = cp.round(best_acc, 3)
                progress.set_postfix(postfix_dict)

                if show_current_activations:
                    print(f", Current Activations={final_activations}", end='')
            
                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(y_true_batch=transfer_to_gpu(y_train_batch, dtype=y_train_batch.dtype), y_pred_batch=model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(y_true_batch=transfer_to_gpu(y_train_batch, dtype=y_train_batch.dtype), y_pred_batch=model[get_preds_softmax()])

                if batch_size == 1:
                    postfix_dict[f"{data} Loss"] = cp.round(train_loss, 3)
                    best_loss = train_loss
                else:
                    postfix_dict[f"{data} Batch Loss"] = cp.round(train_loss, 3)
                progress.set_postfix(postfix_dict)
                best_loss = train_loss

                # Update visualizations during training
                if show_history:
                    gen_list = range(1, len(best_acc_per_gen_list) + 2)
                    update_history_plots_for_learner(viz_objects, gen_list, loss_list + [train_loss], 
                                      best_acc_per_gen_list + [best_acc], x_train, final_activations)

                if neurons_history:
                    viz_objects['neurons']['artists'] = (
                        update_neuron_history_for_learner(cp.copy(best_weights), viz_objects['neurons']['ax'],
                                     viz_objects['neurons']['row'], viz_objects['neurons']['col'],
                                     y_train[0], viz_objects['neurons']['artists'],
                                     data=data, fig1=viz_objects['neurons']['fig'],
                                     acc=best_acc, loss=train_loss)
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
                    print(f'Train Accuracy:', train_model[get_acc()])
                    print(f'Train Loss: ', train_loss, '\n')
                
                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              train_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

                # Check target loss
                if target_loss is not None and best_loss <= target_loss:
                    progress.close()
                    train_model = evaluate(x_train, y_train, W=best_weights, 
                                        activation_potentiation=final_activations)

                    if loss == 'categorical_crossentropy':
                        train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                           y_pred_batch=train_model[get_preds_softmax()])
                    else:
                        train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])

                    print('\nActivations: ', final_activations)
                    print(f'Train Accuracy: ', train_model[get_acc()])
                    print(f'Train Loss: ', train_loss, '\n')

                    # Display final visualizations
                    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                              train_loss, y_train, interval)
                    return best_weights, best_model[get_preds()], best_acc, final_activations

            progress.update(1)

        best_acc_per_gen_list.append(best_acc)
        loss_list.append(best_loss)
        
        weight_pop, act_pop = optimizer(cp.array(weight_pop, copy=False, dtype=dtype), act_pop, i, cp.array(target_pop, dtype=dtype, copy=False), bar_status=False)
        target_pop = []

        # Early stopping check
        if early_stop == True and i > 0:
            if best_acc_per_gen_list[i] == best_acc_per_gen_list[i-1]:
                progress.close()
                train_model = evaluate(x_train, y_train, W=best_weights,
                                    activation_potentiation=final_activations)

                if loss == 'categorical_crossentropy':
                    train_loss = categorical_crossentropy(y_true_batch=y_train, 
                                                       y_pred_batch=train_model[get_preds_softmax()])
                else:
                    train_loss = binary_crossentropy(y_true_batch=y_train, 
                                                   y_pred_batch=train_model[get_preds_softmax()])

                print('\nActivations: ', final_activations)
                print(f'Train Accuracy:', train_model[get_acc()])
                print(f'Train Loss: ', train_loss, '\n')

                # Display final visualizations
                display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, 
                                          train_loss, y_train, interval)
                return best_weights, best_model[get_preds()], best_acc, final_activations

    # Final evaluation
    progress.close()
    train_model = evaluate(x_train, y_train, W=best_weights,
                        activation_potentiation=final_activations)

    if loss == 'categorical_crossentropy':
        train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])
    else:
        train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_softmax()])

    print('\nActivations: ', final_activations)
    print(f'Train Accuracy: ', train_model[get_acc()])
    print(f'Train Loss : ', train_loss, '\n')

    # Display final visualizations
    display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, train_loss, y_train, interval)
    return best_weights, best_model[get_preds()], best_acc, final_activations

def evaluate(
    x_test,
    y_test,
    W,
    activation_potentiation=['linear']
) -> tuple:
    """
    Evaluates the neural network model using the given test data.

    Args:
        x_test (cp.ndarray): Test data.

        y_test (cp.ndarray): Test labels (one-hot encoded).

        W (cp.ndarray): Neural net weight matrix.
        
        activation_potentiation (list): Activation list. Default = ['linear'].
        
    Returns:
        tuple: Model (list).
    """

    x_test = apply_activation(x_test, activation_potentiation)
    result = x_test @ W.T
    softmax_preds = cp.exp(result) / cp.sum(cp.exp(result), axis=1, keepdims=True); accuracy = (cp.argmax(result, axis=1) == cp.argmax(y_test, axis=1)).mean()

    return W, None, accuracy, None, None, softmax_preds