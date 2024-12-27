# -*- coding: utf-8 -*-
"""

MAIN MODULE FOR PLAN

PLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf
ANAPLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydili.06/
@contact: tchasancan@gmail.com
"""



import pandas as pd
import numpy as np
from colorama import Fore, Style, init
from typing import List, Union
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import ArtistAnimation
import networkx as nx
import sys
import math

### LIBRARY IMPORTS ###
from .ui import speak, loading_bars
from .data_operations import normalization, decode_one_hot
from .visualizations import draw_neural_web, draw_activations, plot_decision_boundary, plot_evaluate, neuron_history
from .loss_functions import binary_crossentropy, categorical_crossentropy
from .activation_functions import apply_activation, Softmax
from .metrics import metrics


bar_format = loading_bars()[0]
bar_format_learner = loading_bars()[1]

# BUILD -----


def about_anaplan():

    speak("about_anaplan")

def fit(
    x_train: List[Union[int, float]],
    y_train: List[Union[int, float]], # One hot encoded
    val= None,
    val_count = None,
    activation_potentiation=['linear'], # activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)
    x_val= None,
    y_val= None,
    show_training = None,
    interval=100,
    LTD = 0, # LONG TERM DEPRESSION
    decision_boundary_status=True,
    train_bar = True,
    auto_normalization = True,
    neurons_history = False,
) -> str:
    """
    Creates and configures a PLAN model.
    
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

    if len(x_train) != len(y_train):

        print(Fore.RED + "ERROR301: x_train list and y_train list must be same length. from: fit" + Style.RESET_ALL)
        sys.exit()

    if val == True:

        val_postfix={}
        val_old_progress = 1
        bar_ncols=71

        try:

            if x_val == None and y_val == None:

                x_val = x_train
                y_val = y_train
                
        except:

            pass

        if val_count == None:

            val_count = 10
    
        val_list = [] * val_count

    else:
        
        bar_ncols=44

    if show_training == True:

        G = nx.Graph()

        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Train History')

        artist1 = []
        artist2 = []
        artist3 = []
        artist4 = []

        if val != True:

            print(Fore.RED + "ERROR115: For showing training, val parameter must be True. from: fit", Style.RESET_ALL)
            sys.exit()


    class_count = set()

    for sublist in y_train:

        class_count.add(tuple(sublist))

    class_count = list(class_count)

    y_train = [tuple(sublist) for sublist in y_train]
    
    x_train_0 = np.array(x_train[0])
    x_train__0_vec = x_train_0.ravel()
    x_train_size = len(x_train__0_vec)


    if neurons_history == True:

        row, col = find_closest_factors(len(x_train[0]))

        artist5 = []

        fig1, ax1 = plt.subplots(1, len(class_count), figsize=(18, 14))
            

    STPW = np.ones((len(class_count), x_train_size)) # STPW = SHORT TIME POTENTIATION WEIGHT
    LTPW = np.zeros((len(class_count), x_train_size)) # LTPW = LONG TIME POTENTIATION WEIGHT
    
    y = decode_one_hot(y_train)

    if train_bar == True:

        train_progress = tqdm(
        total=len(x_train), 
        leave=True, 
        desc="Fitting",
        ascii="▱▰",
        bar_format= bar_format,
        ncols=bar_ncols
    )

    for index, inp in enumerate(x_train):

        progress = (index + 1) / len(x_train) * 100

        inp = np.array(inp)
        inp = inp.ravel()

        neural_layer = inp

        STPW = fex(neural_layer, STPW, is_training=True, Class=y[index], activation_potentiation=activation_potentiation, LTD=LTD)
        
        if auto_normalization == True:

            LTPW += normalization(STPW)

        else:

            LTPW += STPW

        if neurons_history == True:

            artist5 = neuron_history(LTPW, ax1, row, col, class_count, artist5, fig1)


        if val == True:

                if int(progress) % val_count == 1 and val_old_progress != int(progress):
                    
                    val_old_progress = int(progress)

                    validation_model = evaluate(x_val, y_val, LTPW , loading_bar_status=False,  activation_potentiation=activation_potentiation, show_metrics=None)
                    val_acc = validation_model[get_acc()]

                    val_postfix['Validation Accuracy'] = val_acc
                    train_progress.set_postfix(val_postfix)
                    val_list.append(val_acc)
                    
                    
                    if show_training == True:

                        mat = LTPW
                        art2 = ax[0, 0].imshow(mat, interpolation='sinc', cmap='viridis')
                        suptitle_info = 'Weight Learning Progress'
                        
                        ax[0, 0].set_title(suptitle_info)

                        artist2.append([art2])
                        
                        if decision_boundary_status == True:

                            art1_1, art1_2 = plot_decision_boundary(x_val, y_val, activation_potentiation, LTPW, artist=artist1, ax=ax)
                            artist1.append([*art1_1.collections, art1_2])


                        period = list(range(1, len(val_list) + 1))

                        art3 = ax[1, 1].plot(
                        period, 
                        val_list, 
                        linestyle='--',   
                        color='g',       
                        marker='o',
                        markersize=6,
                        linewidth=2,      
                        label='Validation Accuracy'
                    )

                        ax[1, 1].set_title('Validation History')
                        ax[1, 1].set_xlabel('Time')
                        ax[1, 1].set_ylabel('Validation Accuracy')
                        ax[1, 1].set_ylim([0, 1])

                        artist3.append(art3)

                        art4_1, art4_2, art4_3 = draw_neural_web(W=LTPW, ax=ax[0, 1], G=G, return_objs=True)
                        art4_list = [art4_1] + [art4_2] + list(art4_3.values())

                        artist4.append(art4_list)


        STPW = np.ones((len(class_count), x_train_size)) # STPW = SHORT TIME POTENTIATION WEIGHT

        if train_bar == True:
            train_progress.update(1)

    if show_training == True:

        mat = LTPW

        for i in range(30):

            artist1.append([*art1_1.collections, art1_2])

            art2 = ax[0, 0].imshow(mat, interpolation='sinc', cmap='viridis')
            suptitle_info = 'Weight Learning Progress:'

            ax[0, 0].set_title(suptitle_info)

            artist2.append([art2])

            art3 = ax[1, 1].plot(
            period, 
            val_list, 
            linestyle='--', 
            color='g',        
            marker='o',      
            markersize=6,     
            linewidth=2,     
            label='Validation Accuracy'
            )
            
            ax[1, 1].set_title('Validation History')
            ax[1, 1].set_xlabel('Time')
            ax[1, 1].set_ylabel('Validation Accuracy')
            ax[1, 1].set_ylim([0, 1])

            artist3.append(art3)

            art4_1, art4_2, art4_3 = draw_neural_web(W=LTPW, ax=ax[0, 1], G=G, return_objs=True)
            art4_list = [art4_1] + [art4_2] + list(art4_3.values())

            artist4.append(art4_list)

        ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
        ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
        ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)
        ani4 = ArtistAnimation(fig, artist4, interval=interval, blit=True)

        plt.tight_layout()
        plt.show()

    if neurons_history == True:

        ani5 = ArtistAnimation(fig1, artist5, interval=interval, blit=True)
        
        plt.tight_layout()
        plt.show()

    LTPW = normalization(LTPW)


    if val == True:

        return LTPW, val_list, val_acc
    
    else:

        return LTPW

# FUNCTIONS -----

def find_closest_factors(a):

    root = int(math.sqrt(a))
    
    for i in range(root, 0, -1):
        if a % i == 0:
            j = a // i
            return i, j

def generate_fixed_positions(G, layout_type='circular'):
    pos = {}
    num_nodes = len(G.nodes())
    
    if layout_type == 'circular':
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        radius = 10
        for i, node in enumerate(G.nodes()):
            pos[node] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
    elif layout_type == 'grid':
        grid_size = int(np.ceil(np.sqrt(num_nodes)))
        for i, node in enumerate(G.nodes()):
            pos[node] = (i % grid_size, i // grid_size)
    else:
        raise ValueError("Unsupported layout_type. Use 'circular' or 'grid'.")

    return pos

def weight_normalization(
    W,
    class_count
) -> str:
    """
    Row(Neuron) based normalization. For unbalanced models.

    Args:
            W (list(num)): Trained weight matrix list.
            class_count (int): Class count of model.

    Returns:
        list([numpy_arrays],[...]): posttrained weight matices of the model. .
    """

    for i in range(class_count):
        
            W[0][i,:] = normalization(W[0][i,:])
    
    return W


def activations_list():

    activations_list = ['linear', 'spiral', 'sigmoid', 'relu', 'tanh', 'swish', 'sin_plus', 'circular', 'mod_circular', 'tanh_circular', 'leaky_relu', 'softplus', 'elu', 'gelu', 'selu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'exsig',  'acos',  'gla',  'srelu', 'qelu',  'isra',  'waveakt', 'arctan', 'bent_identity', 'sech',  'softsign',  'pwl', 'cubic',  'gaussian',  'sine', 'tanh_square', 'mod_sigmoid',  'quartic', 'square_quartic',  'cubic_quadratic',  'exp_cubic',  'sine_square', 'logarithmic',  'scaled_cubic', 'sine_offset']

    print('All avaliable activations: ',  activations_list, "\n\nYOU CAN COMBINE EVERY ACTIVATION. EXAMPLE: ['linear', 'tanh'] or ['waveakt', 'linear', 'sine'].")

    return activations_list


def batcher(x_test, y_test, batch_size=1):

    y_labels = np.argmax(y_test, axis=1)

    sampled_x, sampled_y = [], []
    
    for class_label in np.unique(y_labels):

        class_indices = np.where(y_labels == class_label)[0]
        
        num_samples = int(len(class_indices) * batch_size)
        
        sampled_indices = np.random.choice(class_indices, num_samples, replace=False)
        
        sampled_x.append(x_test[sampled_indices])
        sampled_y.append(y_test[sampled_indices])
    
    return np.concatenate(sampled_x), np.concatenate(sampled_y)


def learner(x_train, y_train, x_test=None, y_test=None, strategy='accuracy', batch_size=1, neural_web_history=False, show_current_activations=False, auto_normalization=True, neurons_history=False, patience=None, depth=None, early_shifting=False, early_stop=False, loss='categorical_crossentropy', show_history=False, interval=33.33, target_acc=None, target_loss=None, except_this=None, only_this=None, start_this=None):
    
    """
    Optimizes the activation functions for a neural network by leveraging train data to find the most accurate combination of activation potentiation for the given dataset.
    This next-generation generalization function includes an advanced learning feature that is specifically tailored to the PLAN algorithm.
    It uniquely adjusts hyperparameters based on test accuracy while training with model-specific training data, offering an unparalleled optimization technique.
    Designed to be used before model evaluation. This called TFL(Test or Train Feedback Learning).

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

    activation_potentiation = ['linear', 'tanh', 'bent_identity', 'waveakt', 'sine', 'tanh_circular', 'elu', 'gelu', 'selu', 'sigmoid', 'relu',  'swish', 'sin_plus', 'spiral', 'circular', 'mod_circular',  'leaky_relu', 'softplus',  'sinakt', 'p_squared', 'sglu', 'dlrelu', 'srelu', 'qelu',  'isra', 'arctan', 'sech',  'softsign',  'pwl', 'cubic', 'tanh_square', 'mod_sigmoid',  'quartic', 'square_quartic',  'cubic_quadratic',  'exp_cubic',  'sine_square', 'exsig',  'acos',  'gla', 'logarithmic',  'scaled_cubic', 'sine_offset', 'gaussian']


    if x_test is None and y_test is None:

        x_test = x_train
        y_test = y_train

        data = 'Train'

    else:

        data = 'Test'

    if early_shifting != False:
        shift_patience = early_shifting


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


    if only_this != None:
        activation_potentiation = only_this

    if except_this != None:
        activation_potentiation = [item for item in activation_potentiation if item not in except_this]

    if depth is None:
        depth = len(activation_potentiation)


    if show_history == True:

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))
        fig.suptitle('Learner History')
        depth_list = []
        artist1 = []
        artist2 = []
        artist3 = []

    if neurons_history == True:

        row, col = find_closest_factors(len(x_train[0]))
        artist4 = []

        if row != 0:
            fig1, ax1 = plt.subplots(1, len(y_train[0]), figsize=(18, 14))
            
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=(18, 14))

    if neural_web_history == True:

        artist5 = []
        G = nx.Graph()
        fig2, ax2 = plt.subplots(figsize=(18, 4))


    if batch_size == 1:
        ncols=76
    else:
        ncols=89

    progress = tqdm(total=len(activation_potentiation), ascii="▱▰",
    bar_format= bar_format_learner, ncols=ncols)

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

            test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

        if loss == 'binary_crossentropy':

            test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

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

                    test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

                if loss == 'binary_crossentropy':

                    test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

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


            if (strategy == 'accuracy' and acc >= best_acc) or (strategy == 'loss' and test_loss <= best_loss) or (strategy == 'f1' and f1_score >= best_f1) or (strategy == 'precision' and precision_score >= best_precision) or (strategy == 'recall' and recall_score >= best_recall) or (strategy == 'all' and f1_score >= best_f1 and acc >= best_acc and test_loss <= best_loss and precision_score >= best_precision and recall_score >= best_recall):

                current_best_activation = activation_potentiation[j]
                best_acc = acc
                
                if batch_size == 1:
                    
                    postfix_dict[f"{data} Accuracy"] = best_acc
                    progress.set_postfix(postfix_dict)
                    
                else:
                    
                    postfix_dict[f"{data} Batch Accuracy"] = acc
                    progress.set_postfix(postfix_dict)
                
                final_activations = activations

                if show_current_activations == True:
                    print(f", Current Activations={final_activations}", end='')

                best_weights = W
                best_model = model

                if strategy != 'loss':

                    if loss == 'categorical_crossentropy':

                        test_loss = categorical_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

                    elif loss == 'binary_crossentropy':

                        test_loss = binary_crossentropy(y_true_batch=y_test_batch, y_pred_batch=model[get_preds_raw()])

                if batch_size == 1:

                    postfix_dict[f"{data} Loss"] = test_loss
                    progress.set_postfix(postfix_dict)
                    best_loss = test_loss

                else:

                    postfix_dict[f"{data} Batch Loss"] = test_loss
                    progress.set_postfix(postfix_dict)
                    

                best_loss = test_loss

                if neurons_history == True:

                    artist4 = neuron_history(np.copy(best_weights), ax1, row, col, y_train[0], artist4, data=data, fig1=fig1, acc=best_acc, loss=test_loss)
                    

                if neural_web_history == True:

                    art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=ax2, G=G, return_objs=True)
                    art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                    artist5.append(art5_list)

                if target_acc is not None:

                    if best_acc >= target_acc:

                        progress.close()

                        train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, activation_potentiation=final_activations)

                        if loss == 'categorical_crossentropy':

                            train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])


                        elif loss == 'binary_crossentropy':

                            train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])
                                

                        print('\nActivations: ', final_activations)
                        print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                        print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                        if data == 'Test':
                            print(f'Test Accuracy (%{batch_size * 100} of test samples): ' , best_acc)
                            print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

                        if show_history == True:

                            if i != 0:

                                for _ in range(30):
                                    
                                    artist1.append(art1)
                                    artist2.append(art2)
                                    artist3.append(art3)
                                    
                                ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
                                ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
                                ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)

                                plt.tight_layout()
                                plt.show()

                            else:
                        
                                print(Fore.WHITE + 'Cannot visualize history because depth already 1.' + Fore.RESET)


                        if neurons_history == True:

                            for i in range(10):
                                artist4 = neuron_history(np.copy(best_weights), ax1, row, col, y_train[0], artist4, data=data, fig1=fig1, acc=best_acc, loss=test_loss)

                            ani4 = ArtistAnimation(fig1, artist4, interval=interval, blit=True)
                            plt.tight_layout()
                            plt.show()

                        if neural_web_history == True:

                            for j in range(30):
                                art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=ax2, G=G, return_objs=True)
                                art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                                artist5.append(art5_list)

                            ani5 = ArtistAnimation(fig2, artist5, interval=interval, blit=True)
                            plt.tight_layout()
                            plt.show()

                        return best_weights, best_model[get_preds()], best_acc, final_activations

                if target_loss is not None:

                    if best_loss <= target_loss:

                        progress.close()

                        train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, activation_potentiation=final_activations)

                        if loss == 'categorical_crossentropy':

                            train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])


                        elif loss == 'binary_crossentropy':

                            train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])
                                

                        print('\nActivations: ', final_activations)
                        print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                        print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                        if data == 'Test':
                            print(f'Test Accuracy (%{batch_size * 100} of test samples): ' , best_acc)
                            print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')


                        if show_history == True:

                            if i != 0:

                                for _ in range(30):
                                    
                                    artist1.append(art1)
                                    artist2.append(art2)
                                    artist3.append(art3)
                                    
                                ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
                                ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
                                ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)

                                plt.tight_layout()
                                plt.show()

                            else:
                        
                                print(Fore.WHITE + 'Cannot visualize history because depth already 1.' + Fore.RESET)


                        if neurons_history == True:

                            for i in range(10):
                                artist4 = neuron_history(np.copy(best_weights), ax1, row, col, y_train[0], artist4, data=data, fig1=fig1, acc=best_acc, loss=test_loss)

                            ani4 = ArtistAnimation(fig1, artist4, interval=interval, blit=True)
                            plt.tight_layout()
                            plt.show()

                        if neural_web_history == True:

                            for j in range(30):
                                art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=ax2, G=G, return_objs=True)
                                art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                                artist5.append(art5_list)

                            ani5 = ArtistAnimation(fig2, artist5, interval=interval, blit=True)
                            plt.tight_layout()
                            plt.show()

                        return best_weights, best_model[get_preds()], best_acc, final_activations

           
            progress.update(1)
            activations = []
            

        best_activations.append(current_best_activation)
        
        best_acc_per_depth_list.append(best_acc)
        loss_list.append(best_loss)

        if adaptive == True and strategy == 'accuracy' and i + 1 >= patience:
            
            check = best_acc_per_depth_list[-patience:]

            if all(x == check[0] for x in check):

                strategy = 'loss'
                adaptive = False

        elif adaptive == True and strategy == 'loss' and best_loss <= patience:

                strategy = 'accuracy'
                adaptive = False

        if show_history == True:

            depth_list = range(1, len(best_acc_per_depth_list) + 1)

            art1 = ax[0].plot(depth_list, loss_list, color='r', markersize=6, linewidth=2, label='Loss Over Depth')
            ax[0].set_title(f'{data} Loss Over Depth')
            artist1.append(art1)

            art2 = ax[1].plot(depth_list, best_acc_per_depth_list, color='g', markersize=6, linewidth=2, label='Accuracy Over Depth')
            ax[1].set_title(f'{data} Accuracy Over Depth')
            artist2.append(art2)

            x = np.linspace(np.min(x_train), np.max(x_train), len(x_train))
            translated_x_train = np.copy(x)
            
            for activation in final_activations:

                translated_x_train += draw_activations(x, activation)

            y = translated_x_train
            art3 = ax[2].plot(x, y, color='b', markersize=6, linewidth=2, label='Activations Over Depth')
            ax[2].set_title('Potentiation Shape Over Depth')
            artist3.append(art3)

        if early_stop == True and i > 0:

            if best_acc_per_depth_list[i] == best_acc_per_depth_list[i-1]:

                progress.close()

                train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, activation_potentiation=final_activations)

                if loss == 'categorical_crossentropy':

                    train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])


                elif loss == 'binary_crossentropy':

                    train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])
                        

                print('\nActivations: ', final_activations)
                print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
                print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
                if data == 'Test':
                    print(f'Test Accuracy (%{batch_size * 100} of test samples): ' , best_acc)
                    print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

                if show_history == True:

                    for _ in range(30):
                        
                        artist1.append(art1)
                        artist2.append(art2)
                        artist3.append(art3)
                        
                    ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
                    ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
                    ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)

                    plt.tight_layout()
                    plt.show()


                if neurons_history == True:

                    for i in range(10):
                        artist4 = neuron_history(np.copy(best_weights), ax1, row, col, y_train[0], artist4, data=data, fig1=fig1, acc=best_acc, loss=test_loss)

                    ani4 = ArtistAnimation(fig1, artist4, interval=interval, blit=True)
                    plt.tight_layout()
                    plt.show()

                if neural_web_history == True:

                    for j in range(30):
                        art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=ax2, G=G, return_objs=True)
                        art5_list = [art5_1] + [art5_2] + list(art5_3.values())
                        artist5.append(art5_list)

                    ani5 = ArtistAnimation(fig2, artist5, interval=interval, blit=True)
                    plt.tight_layout()
                    plt.show()


                return best_weights, best_model[get_preds()], best_acc, final_activations



    train_model = evaluate(x_train, y_train, W=best_weights, loading_bar_status=False, activation_potentiation=final_activations)

    if loss == 'categorical_crossentropy':

        train_loss = categorical_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])


    elif loss == 'binary_crossentropy':

        train_loss = binary_crossentropy(y_true_batch=y_train, y_pred_batch=train_model[get_preds_raw()])

    progress.close()

    print('\nActivations: ', final_activations)
    print(f'Train Accuracy (%{batch_size * 100} of train samples):', train_model[get_acc()])
    print(f'Train Loss (%{batch_size * 100} of train samples): ', train_loss, '\n')
    if data == 'Test':
        print(f'Test Accuracy (%{batch_size * 100} of test samples): ' , best_acc)
        print(f'Test Loss (%{batch_size * 100} of test samples): ', best_loss, '\n')

    if show_history == True:

        for _ in range(30):
            
            artist1.append(art1)
            artist2.append(art2)
            artist3.append(art3)
            
        ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
        ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
        ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)

        plt.tight_layout()
        plt.show()


    if neurons_history == True:

        for i in range(10):
            artist4 = neuron_history(np.copy(best_weights), ax1, row, col, y_train[0], artist4, data=data, fig1=fig1, acc=best_acc, loss=test_loss)

        ani4 = ArtistAnimation(fig1, artist4, interval=interval, blit=True)
        plt.tight_layout()
        plt.show()

    if neural_web_history == True:

        for j in range(30):
            art5_1, art5_2, art5_3 = draw_neural_web(W=best_weights, ax=ax2, G=G, return_objs=True)
            art5_list = [art5_1] + [art5_2] + list(art5_3.values())
            artist5.append(art5_list)

        ani5 = ArtistAnimation(fig2, artist5, interval=interval, blit=True)
        plt.tight_layout()
        plt.show()


    return best_weights, best_model[get_preds()], best_acc, final_activations



def fex(
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

        Wc = [0] * len(W)  # Wc = Weight copy
        true = 0
        y_preds = []
        y_preds_raw = []
        acc_list = []

        Wc = np.copy(W)
            
        
        if loading_bar_status == True:

            loading_bar = tqdm(total=len(x_test), leave=True, ascii="▱▰",
            bar_format= bar_format, desc='Testing', ncols=64)

        for inpIndex, Input in enumerate(x_test):
            Input = np.array(Input)
            Input = Input.ravel()
            neural_layer = Input

                
            neural_layer = fex(neural_layer, W, is_training=False, Class='?', activation_potentiation=activation_potentiation)


            W = np.copy(Wc)
            
            neural_layer = Softmax(neural_layer)

            max_value = max(neural_layer)

            predict_probabilitys.append(max_value)
            
            RealOutput = np.argmax(y_test[inpIndex])
            real_classes.append(RealOutput)
            PredictedOutput = np.argmax(neural_layer)
            predict_classes.append(PredictedOutput)

            if RealOutput == PredictedOutput:
                true += 1
            acc = true / len(y_test)


            acc_list.append(acc)
            y_preds.append(PredictedOutput)
            y_preds_raw.append(Softmax(neural_layer))

            if loading_bar_status == True:
                loading_bar.update(1)
                
            if inpIndex != 0 and loading_bar_status == True:
                loading_bar.set_postfix({"Test Accuracy": acc})

        if show_metrics == True:
            
            loading_bar.close()
            plot_evaluate(x_test, y_test, y_preds, acc_list, W=W, activation_potentiation=activation_potentiation)
        
            W = np.copy(Wc)

    except Exception as e:

        print(Fore.RED + 'ERROR:' + str(e) + Style.RESET_ALL)
        sys.exit()

    return W, y_preds, acc, None, None, y_preds_raw


def get_weights():

    return 0


def get_preds():

    return 1


def get_acc():

    return 2


def get_act_pot():

    return 3


def get_scaler():

    return 4

def get_preds_raw():

    return 5