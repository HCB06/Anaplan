# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024

@author: hasan can
"""

import pandas as pd
import numpy as np
import time
from colorama import Fore, Style
from typing import List, Union
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial import ConvexHull
from datetime import datetime
from scipy import io
import scipy.io as sio
from matplotlib.animation import ArtistAnimation
import networkx as nx

# BUILD -----


def fit(
    x_train: List[Union[int, float]],
    y_train: List[Union[int, float]], # One hot encoded
    val= None,
    val_count = None,
    activation_potentiation=[None], # activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)
    x_val= None,
    y_val= None,
    show_training = None,
    visible_layer=None, # For the future [DISABLED]
    interval=100,
    LTD = 0 # LONG TERM DEPRESSION
) -> str:

    infoPLAN = """
    Creates and configures a PLAN model.
    
    Args:
        x_train (list[num]): List or numarray of input data.
        y_train (list[num]): List or numarray of target labels. (one hot encoded)
        val (None or True): validation in training process ? None or True default: None (optional)
        val_count (None or int): After how many examples learned will an accuracy test be performed? default: 10=(%10) it means every approximately 10 step (optional)
        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: help(plan.activation_functions_list) default: [None] (optional)
        x_val (list[num]): List of validation data. (optional) Default: %10 of x_train (auto_balanced) it means every %1 of train progress starts validation default: x_train (optional)
        y_val (list[num]): (list[num]): List of target labels. (one hot encoded) (optional) Default: %10 of y_train (auto_balanced) it means every %1 of train progress starts validation default: y_train (optional)
        show_training (bool, str): True or None default: None (optional)
        visible_layer: For the future [DISABLED]
        LTD (int): Long Term Depression Hyperparameter for train PLAN neural network (optional)
        interval (float, int): frame delay (milisecond) parameter for Training Report (show_training=True) This parameter effects to your Training Report performance. Lower value is more diffucult for Low end PC's (33.33 = 30 FPS, 16.67 = 60 FPS) default: 100 (optional)

    Returns:
        list([num]): (Weight matrix).
        error handled ?: Process status ('e')
"""

    fit.__doc__ = infoPLAN
    
    visible_layer = None

    if len(x_train) != len(y_train):

        print(Fore.RED + "ERROR301: x_train list and y_train list must be same length. from: fit", infoPLAN + Style.RESET_ALL)
        return 'e'

    if val == True:

        try:

            if x_val == None and y_val == None:

                x_val = x_train
                y_val = y_train
                
        except:

            pass

        if val_count == None:

            val_count = 10
    
        val_bar = tqdm(total=1, desc="Validating Accuracy", ncols=120)
        v_iter = 0
        val_list = [] * val_count

    if show_training == True:

        G = nx.Graph()

        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Train Report')

        artist1 = []
        artist2 = []
        artist3 = []
        artist4 = []

        if val != True:

            print(Fore.RED + "ERROR115: For showing training, val parameter must be True. from: fit",
                  infoPLAN + Style.RESET_ALL)
            return 'e'


    class_count = set()

    for sublist in y_train:

        class_count.add(tuple(sublist))

    class_count = list(class_count)

    y_train = [tuple(sublist) for sublist in y_train]
    
    if visible_layer == None:

        layers = ['fex']
    else:

        layers = ['fex'] * visible_layer

    x_train[0] = np.array(x_train[0])
    x_train[0] = x_train[0].ravel()
    x_train_size = len(x_train[0])
    
    if visible_layer == None:

        STPW = [None]
        STPW[0] = np.ones((len(class_count), x_train_size)) # STPW = SHORT TIME POTENTIATION WEIGHT

    else:

        if visible_layer == 1:
            fex_count = visible_layer
        else:
            fex_count = visible_layer - 1

        fex_neurons = [None] * fex_count

        for i in range(fex_count):

            fex_neurons[i] = [x_train_size]
        
        cat_neurons = [len(class_count), x_train_size]

        STPW = weight_identification(
            len(layers), len(class_count), fex_neurons, cat_neurons, x_train_size) # STPW = SHORT TIME POTENTIATION WEIGHT

    LTPW = [0] * len(STPW) # LTPW = LONG TIME POTENTIATION WEIGHT

    y = decode_one_hot(y_train)

    train_progress = tqdm(total=len(x_train),leave=False, desc="Training",ncols= 120)
    
    max_w = len(STPW) - 1

    for index, inp in enumerate(x_train):

        progress = index / len(x_train) * 100

        inp = np.array(inp)
        inp = inp.ravel()
        
        if x_train_size != len(inp):
            print(Fore.RED + "ERROR304: All input matrices or vectors in x_train list, must be same size. from: fit",
                  infoPLAN + Style.RESET_ALL)
            return 'e'

        neural_layer = inp

        for Lindex, Layer in enumerate(STPW):

            
            STPW[Lindex] = fex(neural_layer, STPW[Lindex], True, y[index], activation_potentiation, index=Lindex, max_w=max_w, LTD=LTD)
                
        
        for i in range(len(STPW)):
            STPW[i] = normalization(STPW[i])
        
        for i, w in enumerate(STPW):
                LTPW[i] = LTPW[i] + w
            
        if val == True:

                if int(progress) % val_count == 1:

                    validation_model = evaluate(x_val, y_val, LTPW ,bar_status=False, activation_potentiation=activation_potentiation, show_metrices=None)
                    val_acc = validation_model[get_acc()]

                    val_list.append(val_acc)
                    
                    if show_training == True:

                        
                        mat = LTPW[0]
                        art2 = ax[0, 0].imshow(mat, interpolation='sinc', cmap='viridis')
                        suptitle_info = 'Weight Learning Progress'
                        
                        ax[0, 0].set_title(suptitle_info)

                        artist2.append([art2])
                        
                        artist1 = plot_decision_boundary(ax, x_val, y_val, activation_potentiation, LTPW, artist=artist1)

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

                        for i in range(LTPW[0].shape[0]):
                            for j in range(LTPW[0].shape[1]):
                                if LTPW[0][i, j] != 0:
                                    G.add_edge(f'Motor Neuron{i}', f'Sensory Neuron{j}', ltpw=LTPW[0][i, j])

                        edges = G.edges(data=True)
                        weights = [edata['ltpw'] for _, _, edata in edges]
                        pos = generate_fixed_positions(G, layout_type='circular')

                        art4_1 = nx.draw_networkx_nodes(G, pos, ax=ax[0, 1], node_size=1000, node_color='lightblue')
                        art4_2 = nx.draw_networkx_edges(G, pos, ax=ax[0, 1], edge_color=weights, edge_cmap=plt.cm.Blues)
                        art4_3 = nx.draw_networkx_labels(G, pos, ax=ax[0, 1], font_size=10, font_weight='bold')
                        ax[0, 1].set_title('Neural Web')
                        
                        art4_list = [art4_1] + [art4_2] + list(art4_3.values())

                        artist4.append(art4_list)
            

                    if v_iter == 0:
                        
                        val_bar.update(val_acc)
                    
                    if v_iter != 0:
                        
                        val_acc = val_acc - val_list[v_iter - 1]
                        val_bar.update(val_acc)
                    
                    v_iter += 1
 
        if visible_layer == None:
            STPW = [None]
            STPW[0] = np.ones((len(class_count), x_train_size)) # STPW = SHORT TIME POTENTIATION WEIGHT

        else:      
            STPW = weight_identification(
                len(layers), len(class_count), fex_neurons, cat_neurons, x_train_size)

        train_progress.update(1)

    if show_training == True:

        mat = LTPW[0]

        for i in range(30):

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

        for i in range(28):

            art4_1 = nx.draw_networkx_nodes(G, pos, ax=ax[0, 1], node_size=1000, node_color='lightblue')
            art4_2 = nx.draw_networkx_edges(G, pos, ax=ax[0, 1], edge_color=weights, edge_cmap=plt.cm.Blues)
            art4_3 = nx.draw_networkx_labels(G, pos, ax=ax[0, 1], font_size=10, font_weight='bold')
            ax[0, 1].set_title('Neural Web')
            
            art4_list = [art4_1] + [art4_2] + list(art4_3.values())

            artist4.append(art4_list)


        artist1 = plot_decision_boundary(ax, x_val, y_val, activation_potentiation, LTPW, artist=artist1, draw_is_finished=True)

        ani1 = ArtistAnimation(fig, artist1, interval=interval, blit=True)
        ani2 = ArtistAnimation(fig, artist2, interval=interval, blit=True)
        ani3 = ArtistAnimation(fig, artist3, interval=interval, blit=True)
        ani4 = ArtistAnimation(fig, artist4, interval=interval, blit=True)

        plt.show()

    LTPW = normalization(LTPW)

    return LTPW

# FUNCTIONS -----

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

def weight_identification(
    layer_count,      # int: Number of layers in the neural network.
    class_count,      # int: Number of classes in the classification task.
    fex_neurons,
    cat_neurons,         # list[num]: List of neuron counts for each layer.
    x_train_size        # int: Size of the input data.
) -> str:
    """
    Identifies the weights for a neural network model.

    Args:
        layer_count (int): Number of layers in the neural network.
        class_count (int): Number of classes in the classification task.
        neurons (list[num]): List of neuron counts for each layer.
        x_train_size (int): Size of the input data.

    Returns:
        list([numpy_arrays],[...]): pretrained weight matices of the model. .
    """

    W = [None] * (len(fex_neurons) + 1)

    for i in range(len(fex_neurons)):
        W[i] = np.ones((fex_neurons[i]))

    W[i + 1] = np.ones((cat_neurons[0], cat_neurons[1]))

    return W

# ACTIVATION FUNCTIONS -----

def Softmax(
    x  # num: Input data to be transformed using softmax function.
):
    """
    Applies the softmax function to the input data.

    Args:
        (num): Input data to be transformed using softmax function.

    Returns:
       (num): Transformed data after applying softmax function.
    """

    return softmax(x)


def Sigmoid(
    x  # num: Input data to be transformed using sigmoid function.
):
    """
    Applies the sigmoid function to the input data.

    Args:
        (num): Input data to be transformed using sigmoid function.

    Returns:
        (num): Transformed data after applying sigmoid function.
    """
    return expit(x)


def Relu(
    x  # num: Input data to be transformed using ReLU function.
):
    """
    Applies the Rectified Linear Unit (ReLU) function to the input data.

    Args:
        (num): Input data to be transformed using ReLU function.

    Returns:
        (num): Transformed data after applying ReLU function.
    """

    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

def circular_activation(x):
    return (np.sin(x) + 1) / 2

def modular_circular_activation(x, period=2*np.pi):
    return np.mod(x, period) / period

def tanh_circular_activation(x):
    return (np.tanh(x) + 1) / 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softplus(x):
    return np.log(1 + np.exp(x))

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def selu(x, lambda_=1.0507, alpha=1.6733):
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 1. Sinusoids Activation (SinAkt)
def sinakt(x):
    return np.sin(x) + np.cos(x)

# 2. Parametric Squared Activation (P-Squared)
def p_squared(x, alpha=1.0, beta=0.0):
    return alpha * x**2 + beta * x

def sglu(x, alpha=1.0):
    return softmax(alpha * x) * x

# 4. Double Leaky ReLU (DLReLU)
def dlrelu(x):
    return np.maximum(0.01 * x, x) + np.minimum(0.01 * x, 0.1 * x)

# 5. Exponential Sigmoid (ExSig)
def exsig(x):
    return 1 / (1 + np.exp(-x**2))

# 6. Adaptive Cosine Activation (ACos)
def acos(x, alpha=1.0, beta=0.0):
    return np.cos(alpha * x + beta)

# 7. Gaussian-like Activation (GLA)
def gla(x, alpha=1.0, mu=0.0):
    return np.exp(-alpha * (x - mu)**2)

# 8. Swish ReLU (SReLU)
def srelu(x):
    return x * (1 / (1 + np.exp(-x))) + np.maximum(0, x)

# 9. Quadratic Exponential Linear Unit (QELU)
def qelu(x):
    return x**2 * np.exp(x) - 1

# 10. Inverse Square Root Activation (ISRA)
def isra(x):
    return x / np.sqrt(np.abs(x) + 1)

def waveakt(x, alpha=1.0, beta=2.0, gamma=3.0):
    return np.sin(alpha * x) * np.cos(beta * x) * np.sin(gamma * x)

def arctan(x):
    return np.arctan(x)

def bent_identity(x):
    return (np.sqrt(x**2 + 1) - 1) / 2 + x

def sech(x):
    return 2 / (np.exp(x) + np.exp(-x))

def softsign(x):
    return x / (1 + np.abs(x))

def pwl(x, alpha=0.5, beta=1.5):
    return np.where(x <= 0, alpha * x, beta * x)

def cubic(x):
    return x**3

def gaussian(x, alpha=1.0, mu=0.0):
    return np.exp(-alpha * (x - mu)**2)
                
def sine(x, alpha=1.0):
    return np.sin(alpha * x)

def tanh_square(x):
    return np.tanh(x)**2

def mod_sigmoid(x, alpha=1.0, beta=0.0):
    return 1 / (1 + np.exp(-alpha * x + beta))

def quartic(x):
    return x**4

def square_quartic(x):
    return (x**2)**2

def cubic_quadratic(x):
    return x**3 * (x**2)

def exp_cubic(x):
    return np.exp(x**3)

def sine_square(x):
    return np.sin(x)**2

def logarithmic(x):
    return np.log(x**2 + 1)

def power(x, p):
    return x**p

def scaled_cubic(x, alpha=1.0):
    return alpha * x**3

def sine_offset(x, beta=0.0):
    return np.sin(x + beta)


def fex(
    Input,               # list[num]: Input data.
    w,                   # num: Weight matrix of the neural network.
    is_training,        # bool: Flag indicating if the function is called during training (True or False).
    Class,               # int: Which class is, if training.
    activation_potentiation, # (list): Activation potentiation list for deep PLAN. (optional)
    index,
    max_w,
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

    Output = np.zeros(len(Input))
    
    for activation in activation_potentiation:

        if activation == 'sigmoid':
            Output += Sigmoid(Input)

        if activation == 'swish':
            Output += swish(Input)

        if activation == 'circular':
            Output += circular_activation(Input)

        if activation == 'mod_circular':
            Output += modular_circular_activation(Input)

        if activation == 'tanh_circular':
            Output += tanh_circular_activation(Input)

        if activation == 'leaky_relu':
            Output += leaky_relu(Input)

        if activation == 'relu':
            Output += Relu(Input)
        
        if activation == 'softplus':
            Output += softplus(Input)

        if activation == 'elu':
            Output += elu(Input)

        if activation == 'gelu':
            Output += gelu(Input)

        if activation == 'selu':
            Output += selu(Input)    

        if activation == 'softmax':
            Output += Softmax(Input)

        if activation == 'tanh':
            Output += tanh(Input)

        if activation == 'sinakt':
            Output += sinakt(Input)

        if activation == 'p_squared':
            Output += p_squared(Input)

        if activation == 'sglu':
            Output += sglu(Input, alpha=1.0)

        if activation == 'dlrelu':
            Output += dlrelu(Input)

        if activation == 'exsig':
            Output += exsig(Input)

        if activation == 'acos':
            Output += acos(Input, alpha=1.0, beta=0.0)

        if activation == 'gla':
            Output += gla(Input, alpha=1.0, mu=0.0)

        if activation == 'srelu':
            Output += srelu(Input)

        if activation == 'qelu':
            Output += qelu(Input)

        if activation == 'isra':
            Output += isra(Input)

        if activation == 'waveakt':
            Output += waveakt(Input) 

        if activation == 'arctan':
            Output += arctan(Input) 
        
        if activation == 'bent_identity':
            Output += bent_identity(Input)

        if activation == 'sech':
            Output += sech(Input)

        if activation == 'softsign':
            Output += softsign(Input)

        if activation == 'pwl':
            Output += pwl(Input)

        if activation == 'cubic':
            Output += cubic(Input)

        if activation == 'gaussian':
            Output += gaussian(Input)

        if activation == 'sine':
            Output += sine(Input)

        if activation == 'tanh_square':
            Output += tanh_square(Input)

        if activation == 'mod_sigmoid':
            Output += mod_sigmoid(Input)

        if activation == None or activation == 'linear':
            Output += Input

        if activation == 'quartic':
            Output += quartic(Input)

        if activation == 'square_quartic':
            Output += square_quartic(Input)

        if activation == 'cubic_quadratic':
            Output += cubic_quadratic(Input)

        if activation == 'exp_cubic':
            Output += exp_cubic(Input)

        if activation == 'sine_square':
            Output += sine_square(Input)

        if activation == 'logarithmic':
            Output += logarithmic(Input)

        if activation == 'scaled_cubic':
            Output += scaled_cubic(Input, 1.0)

        if activation == 'sine_offset':
            Output += sine_offset(Input, 1.0)


    Input = Output


    if is_training == True:
        
        for i in range(LTD):        

            depression_vector = np.random.rand(*Input.shape)

            Input -= depression_vector

        w[Class, :] = Input

        return w

           
    elif is_training == False:
        
        neural_layer = np.dot(w, Input)
    
        return neural_layer
    
    elif is_training == False and max_w != 0:


        if index == max_w:

            neural_layer = np.dot(w, Input)
            return neural_layer

        else:

            neural_layer = [None] * len(w)

            for i in range(len(w)):

                neural_layer[i] = Input[i] * w[i]

            neural_layer = np.array(neural_layer)

            return neural_layer


def normalization(
    Input  # num: Input data to be normalized.
):
    """
    Normalizes the input data using maximum absolute scaling.

    Args:
        Input (num): Input data to be normalized.

    Returns:
        (num) Scaled input data after normalization.
    """

    AbsVector = np.abs(Input)

    MaxAbs = np.max(AbsVector)

    ScaledInput = Input / MaxAbs

    return ScaledInput


def evaluate(
    x_test,         # list[num]: Test input data.
    y_test,         # list[num]: Test labels.
    W,               # list[num]: Weight matrix list of the neural network.
    activation_potentiation=[None], # (list): Activation potentiation list for deep PLAN. (optional)
    bar_status=True,  # bar_status (bool): Loading bar for accuracy (True or None) (optional) Default: True
    show_metrices=None     # show_metrices (bool): (True or None) (optional) Default: None
) -> tuple:
    infoTestModel = """
    Tests the neural network model with the given test data.

    Args:
        x_test (list[num]): Test input data.
        y_test (list[num]): Test labels.
        W (list[num]): Weight matrix list of the neural network.
        activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: help(plan.activation_functions_list) default: [None]
        bar_status (bool): Loading bar for accuracy (True or None) (optional) Default: True
        show_metrices (bool): (True or None) (optional) Default: None

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy of the model.
    """
    evaluate.__doc__ = infoTestModel

    predict_probabilitys = []
    real_classes = []
    predict_classes = []
    
    layer_count = len(W)

    try:
        layers = ['fex'] * layer_count

        Wc = [0] * len(W)  # Wc = Weight copy
        true = 0
        y_preds = []
        acc_list = []
        max_w = len(W) - 1

        for i, w in enumerate(W):
            Wc[i] = np.copy(w)
            
        
        if bar_status == True:

            test_progress = tqdm(total=len(x_test),leave=False, desc='Testing',ncols=120)
            acc_bar = tqdm(total=1, desc="Test Accuracy", ncols=120)
        

        for inpIndex, Input in enumerate(x_test):
            Input = np.array(Input)
            Input = Input.ravel()
            neural_layer = Input

            for index, Layer in enumerate(W):

                
                neural_layer = fex(neural_layer, W[index], False, None, activation_potentiation, index=index, max_w=max_w)


            for i, w in enumerate(Wc):
                W[i] = np.copy(w)
                
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
            
            if bar_status == True:
                test_progress.update(1)
                if inpIndex == 0:
                    acc_bar.update(acc)
                    
                else:
                    acc = acc - acc_list[inpIndex - 1]
                    acc_bar.update(acc)
            
        if show_metrices == True:
            plot_evaluate(x_test, y_test, y_preds, acc_list, W=W, activation_potentiation=activation_potentiation)
        
        
        for i, w in enumerate(Wc):
            W[i] = np.copy(w)
    
    except:

        print(Fore.RED + 'ERROR:' + infoTestModel + Style.RESET_ALL)

    return W, y_preds, acc


def multiple_evaluate(
    x_test,         # list[num]: Test input data.
    y_test,         # list[num]: Test labels.
    show_metrices,      # show_metrices (bool): Visualize test progress ? (True or False)
    MW,                  # list[list[num]]: Weight matrix of the neural network.
    activation_potentiation=None # (float or None): Threshold value for comparison. (optional)
) -> tuple:
    infoTestModel = """
    Tests the neural network model with the given test data.

    Args:
        x_test (list[num]): Test input data.
        y_test (list[num]): Test labels.
        show_metrices (bool): (True or False)
        MW (list(list[num])): Multiple Weight matrix list of the neural network. (Multiple model testing)

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy of the model.
    """

    layers = ['fex', 'cat']

    try:
        y_preds = [-1] * len(y_test)
        acc_list = []
        print(Fore.GREEN + "\n\nTest Started with 0 ERROR\n" + Style.RESET_ALL)
        start_time = time.time()
        true = 0
        for inpIndex, Input in enumerate(x_test):

            output_layer = 0

            for m, Model in enumerate(MW):

                W = Model

                Wc = [0] * len(W)  # Wc = weight copy

                y_preds = [None] * len(y_test)
                for i, w in enumerate(W):
                    Wc[i] = np.copy(w)

                Input = np.array(Input)
                Input = Input.ravel()
                uni_start_time = time.time()
                neural_layer = Input

                for index, Layer in enumerate(layers):

                    neural_layer = normalization(neural_layer)

                    if Layer == 'fex':
                        neural_layer = fex(neural_layer, W[index], False, None, activation_potentiation)

                output_layer += neural_layer

                for i, w in enumerate(Wc):
                    W[i] = np.copy(w)
            for i, w in enumerate(Wc):
                W[i] = np.copy(w)
            RealOutput = np.argmax(y_test[inpIndex])
            PredictedOutput = np.argmax(output_layer)
            if RealOutput == PredictedOutput:
                true += 1
            acc = true / len(y_test)
            if show_metrices == True:
                acc_list.append(acc)
            y_preds[inpIndex] = PredictedOutput
            

            uni_end_time = time.time()

            calculating_est = round(
                (uni_end_time - uni_start_time) * (len(x_test) - inpIndex), 3)

            if calculating_est < 60:
                print('\rest......(sec):', calculating_est, '\n', end="")
                print('\rTest accuracy: ', acc, "\n", end="")

            elif calculating_est > 60 and calculating_est < 3600:
                print('\rest......(min):', calculating_est/60, '\n', end="")
                print('\rTest accuracy: ', acc, "\n", end="")

            elif calculating_est > 3600:
                print('\rest......(h):', calculating_est/3600, '\n', end="")
                print('\rTest accuracy: ', acc, "\n", end="")
        if show_metrices == True:
            plot_evaluate(y_test, y_preds, acc_list)
        
        EndTime = time.time()
        for i, w in enumerate(Wc):
            W[i] = np.copy(w)

        calculating_est = round(EndTime - start_time, 2)

        print(Fore.GREEN + "\nTest Finished with 0 ERROR\n")

        if calculating_est < 60:
            print('Total testing time(sec): ', calculating_est)

        elif calculating_est > 60 and calculating_est < 3600:
            print('Total testing time(min): ', calculating_est/60)

        elif calculating_est > 3600:
            print('Total testing time(h): ', calculating_est/3600)

        if acc >= 0.8:
            print(Fore.GREEN + '\nTotal Test accuracy: ',
                  acc, '\n' + Style.RESET_ALL)

        elif acc < 0.8 and acc > 0.6:
            print(Fore.MAGENTA + '\nTotal Test accuracy: ',
                  acc, '\n' + Style.RESET_ALL)

        elif acc <= 0.6:
            print(Fore.RED + '\nTotal Test accuracy: ',
                  acc, '\n' + Style.RESET_ALL)

    except:

        print(Fore.RED + "ERROR: Testing model parameters like 'activation_potentiation' must be same as trained model. Check parameters. Are you sure weights are loaded ? from: evaluate" + infoTestModel + Style.RESET_ALL)
        return 'e'

    return W, y_preds, acc


def save_model(model_name,
               model_type,
               class_count,
               test_acc,
               weights_type,
               weights_format,
               model_path,
               W,
               scaler_params=None,
               activation_potentiation=[None]
               ):

    infosave_model = """
    Function to save a potentiation learning model.

    Arguments:
    model_name (str): Name of the model.
    model_type (str): Type of the model.(options: PLAN)
    test_acc (float): Test accuracy of the model.
    weights_type (str): Type of weights to save (options: 'txt', 'npy', 'mat').
    WeightFormat (str): Format of the weights (options: 'd', 'f', 'raw').
    model_path (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/
    scaler_params (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.
    W: Weights of the model.
    activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: help(plan.activation_functions_list) default: [None]
    
    Returns:
    str: Message indicating if the model was saved successfully or encountered an error.
    """

    save_model.__doc__ = infosave_model

    class_count = W[0].shape[0]

    layers = ['fex']

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat':
        print(Fore.RED + "ERROR110: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' from: save_model" +
              infosave_model + Style.RESET_ALL)
        return 'e'

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        print(Fore.RED + "ERROR111: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" +
              infosave_model + Style.RESET_ALL)
        return 'e'

    NeuronCount = 0
    SynapseCount = 0


    try:
        for w in W:
            NeuronCount += np.shape(w)[0] + np.shape(w)[1]
            SynapseCount += np.shape(w)[0] * np.shape(w)[1]
    except:

        print(Fore.RED + "ERROR: Weight matrices has a problem from: save_model" +
              infosave_model + Style.RESET_ALL)
        return 'e'

    if scaler_params != None:

        if len(scaler_params) > len(activation_potentiation):

            activation_potentiation += ['']

        elif len(activation_potentiation) > len(scaler_params):

            for i in range(len(activation_potentiation) - len(scaler_params)):

                scaler_params.append(' ')

    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'LAYER COUNT': len(layers),
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'TEST ACCURACY': float(test_acc),
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'MODEL PATH': model_path,
            'STANDARD SCALER': scaler_params,
            'ACTIVATION POTENTIATION': activation_potentiation
            }
    try:

        df = pd.DataFrame(data)

        df.to_csv(model_path + model_name + '.txt', sep='\t', index=False)

    except:

        print(Fore.RED + "ERROR: Model log not saved probably model_path incorrect. Check the log parameters from: save_model" +
                    infosave_model + Style.RESET_ALL)
        return 'e'

    try:

        if weights_type == 'txt' and weights_format == 'd':

            for i, w in enumerate(W):
                np.savetxt(model_path + model_name + '_weights.txt',  w, fmt='%d')

        if weights_type == 'txt' and weights_format == 'f':

            for i, w in enumerate(W):
                np.savetxt(model_path + model_name + '_weights.txt',  w, fmt='%f')

        if weights_type == 'txt' and weights_format == 'raw':

            for i, w in enumerate(W):
                np.savetxt(model_path + model_name + '_weights.txt',  w)

        ###

        if weights_type == 'npy' and weights_format == 'd':

            for i, w in enumerate(W):
                np.save(model_path + model_name + '_weights.npy', w.astype(int))

        if weights_type == 'npy' and weights_format == 'f':

            for i, w in enumerate(W):
                np.save(model_path + model_name + '_weights.npy',  w, w.astype(float))

        if weights_type == 'npy' and weights_format == 'raw':

            for i, w in enumerate(W):
                np.save(model_path + model_name + '_weights.npy',  w)

        ###

        if weights_type == 'mat' and weights_format == 'd':

            for i, w in enumerate(W):
                w = {'w': w.astype(int)}
                io.savemat(model_path + model_name + '_weights.mat', w)

        if weights_type == 'mat' and weights_format == 'f':

            for i, w in enumerate(W):
                w = {'w': w.astype(float)}
                io.savemat(model_path + model_name + '_weights.mat', w)

        if weights_type == 'mat' and weights_format == 'raw':

            for i, w in enumerate(W):
                w = {'w': w}
                io.savemat(model_path + model_name + '_weights.mat', w)

    except:

        print(Fore.RED + "ERROR: Model Weights not saved. Check the Weight parameters. SaveFilePath expl: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: save_model" + infosave_model + Style.RESET_ALL)
        return 'e'
    print(df)
    message = (
        Fore.GREEN + "Model Saved Successfully\n" +
        Fore.MAGENTA + "Don't forget, if you want to load model: model log file and weight files must be in the same directory." +
        Style.RESET_ALL
    )

    return print(message)


def load_model(model_name,
               model_path,
               ):
    infoload_model = """
   Function to load a potentiation learning model.

   Arguments:
   model_name (str): Name of the model.
   model_path (str): Path where the model is saved.

   Returns:
   lists: W(list[num]), activation_potentiation, DataFrame of the model
    """

    load_model.__doc__ = infoload_model

    try:

        df = pd.read_csv(model_path + model_name + '.' + 'txt', delimiter='\t')

    except:

        print(Fore.RED + "ERROR: Model Path error. accaptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" +
              infoload_model + Style.RESET_ALL)

    model_name = str(df['MODEL NAME'].iloc[0])
    layer_count = int(df['LAYER COUNT'].iloc[0])
    WeightType = str(df['WEIGHTS TYPE'].iloc[0])

    W = [0] * layer_count

    if WeightType == 'txt':
        for i in range(layer_count):
            W[i] = np.loadtxt(model_path + model_name + '_weights.txt')
    elif WeightType == 'npy':
        for i in range(layer_count):
            W[i] = np.load(model_path + model_name + '_weights.npy')
    elif WeightType == 'mat':
        for i in range(layer_count):
            W[i] = sio.loadmat(model_path + model_name + '_weights.mat')
    else:
        raise ValueError(
            Fore.RED + "Incorrect weight type value. Value must be 'txt', 'npy' or 'mat' from: load_model." + infoload_model + Style.RESET_ALL)
    print(Fore.GREEN + "Model loaded succesfully" + Style.RESET_ALL)
    return W, df


def predict_model_ssd(Input, model_name, model_path):

    infopredict_model_ssd = """
    Function to make a prediction using a divided potentiation learning artificial neural network (PLAN).

    Arguments:
    Input (list or ndarray): Input data for the model (single vector or single matrix).
    model_name (str): Name of the model.
    Returns:
    ndarray: Output from the model.
    """

    predict_model_ram.__doc__ = infopredict_model_ssd

    W, df = load_model(model_name, model_path)
    
    activation_potentiation = list(df['ACTIVATION POTENTIATION'])

    scaler_params = df['STANDARD SCALER'].tolist()

    scaler_params = [item for item in scaler_params if item != ' ']

    try:

        scaler_params = [np.fromstring(arr.strip('[]'), sep=' ') for arr in scaler_params]

        Input = standard_scaler(None, Input, scaler_params)

    except:
    
        pass

    layers = ['fex']

    Wc = [0] * len(W)
    for i, w in enumerate(W):
        Wc[i] = np.copy(w)
    try:
        neural_layer = Input
        neural_layer = np.array(neural_layer)
        neural_layer = neural_layer.ravel()
        max_w = len(W) - 1
        for index, Layer in enumerate(W):
        
            neural_layer = fex(neural_layer, W[index], False, None, activation_potentiation, index=index, max_w=max_w)

    except:
        print(Fore.RED + "ERROR: The input was probably entered incorrectly. from: predict_model_ssd" +
              infopredict_model_ssd + Style.RESET_ALL)
        return 'e'
    for i, w in enumerate(Wc):
        W[i] = np.copy(w)
    return neural_layer


def predict_model_ram(Input, W, scaler_params=None, activation_potentiation=[None]):

    infopredict_model_ram = """
    Function to make a prediction using a divided potentiation learning artificial neural network (PLAN).
    from weights and parameters stored in memory.

    Arguments:
    Input (list or ndarray): Input data for the model (single vector or single matrix).
    W (list of ndarrays): Weights of the model.
    scaler_params (list): standard scaler params list: mean,std. (optional) Default: None.
    activation_potentiation (list): ac list for deep PLAN. default: [None] ('linear') (optional)

    Returns:
    ndarray: Output from the model.
    """

    predict_model_ram.__doc__ = infopredict_model_ram

    try:
        if scaler_params != None:

            Input = standard_scaler(None, Input, scaler_params)
    except:
            Input = standard_scaler(None, Input, scaler_params)

    layers = ['fex']

    Wc = [0] * len(W)
    for i, w in enumerate(W):
        Wc[i] = np.copy(w)
    try:
        neural_layer = Input
        neural_layer = np.array(neural_layer)
        neural_layer = neural_layer.ravel()

        max_w = len(W) - 1

        for index, Layer in enumerate(W):


            neural_layer = fex(neural_layer, W[index], False, None, activation_potentiation, index=index, max_w=max_w)

        for i, w in enumerate(Wc):
            W[i] = np.copy(w)
        return neural_layer

    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_ram." +
            infopredict_model_ram + Style.RESET_ALL)
    return 'e'

def auto_balancer(x_train, y_train):

    infoauto_balancer = """
   Function to balance the training data across different classes.

   Arguments:
   x_train (list): Input data for training.
   y_train (list): Labels corresponding to the input data.

   Returns:
   tuple: A tuple containing balanced input data and labels.
   """
    
    auto_balancer.__doc__ = infoauto_balancer

    classes = np.arange(y_train.shape[1])
    class_count = len(classes)

    try:
        ClassIndices = {i: np.where(np.array(y_train)[:, i] == 1)[
            0] for i in range(class_count)}
        classes = [len(ClassIndices[i]) for i in range(class_count)]

        if len(set(classes)) == 1:
            print(Fore.WHITE + "INFO: Data have already balanced. from: auto_balancer" + Style.RESET_ALL)
            return x_train, y_train

        MinCount = min(classes)

        BalancedIndices = []
        for i in tqdm(range(class_count),leave=False,desc='Balancing Data',ncols=120):
            if len(ClassIndices[i]) > MinCount:
                SelectedIndices = np.random.choice(
                    ClassIndices[i], MinCount, replace=False)
            else:
                SelectedIndices = ClassIndices[i]
            BalancedIndices.extend(SelectedIndices)

        BalancedInputs = [x_train[idx] for idx in BalancedIndices]
        BalancedLabels = [y_train[idx] for idx in BalancedIndices]

        print(Fore.GREEN + "Data Succesfully Balanced from: " + str(len(x_train)
                                                                                 ) + " to: " + str(len(BalancedInputs)) + ". from: auto_balancer " + Style.RESET_ALL)
    except:
        print(Fore.RED + "ERROR: Inputs and labels must be same length check parameters" + infoauto_balancer)
        return 'e'

    return np.array(BalancedInputs), np.array(BalancedLabels)


def synthetic_augmentation(x_train, y_train):
    """
    Generates synthetic examples to balance classes with fewer examples.

    Arguments:
    x -- Input dataset (examples) - list format
    y -- Class labels (one-hot encoded) - list format

    Returns:
    x_balanced -- Balanced input dataset (list format)
    y_balanced -- Balanced class labels (one-hot encoded, list format)
    """
    x = x_train
    y = y_train
    classes = np.arange(y_train.shape[1])
    class_count = len(classes)

    class_distribution = {i: 0 for i in range(class_count)}
    for label in y:
        class_distribution[np.argmax(label)] += 1

    max_class_count = max(class_distribution.values())

    x_balanced = list(x)
    y_balanced = list(y)

    for class_label in tqdm(range(class_count), leave=False, desc='Augmenting Data',ncols= 120):
        class_indices = [i for i, label in enumerate(
            y) if np.argmax(label) == class_label]
        num_samples = len(class_indices)

        if num_samples < max_class_count:
            while num_samples < max_class_count:

                random_indices = np.random.choice(
                    class_indices, 2, replace=False)
                sample1 = x[random_indices[0]]
                sample2 = x[random_indices[1]]

                synthetic_sample = sample1 + \
                    (np.array(sample2) - np.array(sample1)) * np.random.rand()

                x_balanced.append(synthetic_sample.tolist())
                y_balanced.append(y[class_indices[0]])

                num_samples += 1

    return np.array(x_balanced), np.array(y_balanced)


def standard_scaler(x_train, x_test=None, scaler_params=None):
    info_standard_scaler = """
  Standardizes training and test datasets. x_test may be None.

  Args:
    train_data: numpy.ndarray
      Training data
    test_data: numpy.ndarray
      Test data (optional)

  Returns:
    list:
    Scaler parameters: mean and std
    tuple
      Standardized training and test datasets
  """
    
    standard_scaler.__doc__ = info_standard_scaler

    try:

        x_train = x_train.tolist()
        x_test = x_test.tolist()

    except:

        pass

    try:
            
        if scaler_params == None and x_test != None:
            
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)

            train_data_scaled = (x_train - mean) / std
            test_data_scaled = (x_test - mean) / std
            
            train_data_scaled = np.nan_to_num(train_data_scaled, nan=0)
            test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)
            
            scaler_params = [mean, std]

            return scaler_params, train_data_scaled, test_data_scaled
        
        if scaler_params == None and x_test == None:
            
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)
            train_data_scaled = (x_train - mean) / std
            
            train_data_scaled = np.nan_to_num(train_data_scaled, nan=0)
            
            scaler_params = [mean, std]
            
            return scaler_params, train_data_scaled
            
        if scaler_params != None:

            try:

                test_data_scaled = (x_test - scaler_params[0]) / scaler_params[1]
                test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)

            except:

                test_data_scaled = (x_test - scaler_params[0]) / scaler_params[1]
                test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)
            
            return test_data_scaled

    except:
    
        print(
        Fore.RED + "ERROR: x_train and x_test must be list[numpyarray] from standard_scaler" + info_standard_scaler + Style.RESET_ALL)
    
        return('e')

def encode_one_hot(y_train, y_test):
    """
    Performs one-hot encoding on y_train and y_test data..

    Args:
        y_train (numpy.ndarray): Eğitim etiketi verisi.
        y_test (numpy.ndarray): Test etiketi verisi.

    Returns:
        tuple: One-hot encoded y_train ve y_test verileri.
    """
    classes = np.unique(y_train)
    class_count = len(classes)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    y_train_encoded = np.zeros((y_train.shape[0], class_count))
    for i, label in enumerate(y_train):
        y_train_encoded[i, class_to_index[label]] = 1

    y_test_encoded = np.zeros((y_test.shape[0], class_count))
    for i, label in enumerate(y_test):
        y_test_encoded[i, class_to_index[label]] = 1

    return y_train_encoded, y_test_encoded


def split(X, y, test_size, random_state):
    """
    Splits the given X (features) and y (labels) data into training and testing subsets.

    Args:
        X (numpy.ndarray): Features data.
        y (numpy.ndarray): Labels data.
        test_size (float or int): Proportion or number of samples for the test subset.
        random_state (int or None): Seed for random state.

    Returns:
        tuple: x_train, x_test, y_train, y_test as ordered training and testing data subsets.
    """
    num_samples = X.shape[0]

    if isinstance(test_size, float):
        test_size = int(test_size * num_samples)
    elif isinstance(test_size, int):
        if test_size > num_samples:
            raise ValueError(
                "test_size cannot be larger than the number of samples.")
    else:
        raise ValueError("test_size should be float or int.")

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test


def metrics(y_ts, test_preds, average='weighted'):
    """
    Calculates precision, recall and F1 score for a classification task.
    
    Args:
        y_ts (list or numpy.ndarray): True labels.
        test_preds (list or numpy.ndarray): Predicted labels.
        average (str): Type of averaging ('micro', 'macro', 'weighted').

    Returns:
        tuple: Precision, recall, F1 score.
    """
    y_test_d = decode_one_hot(y_ts)
    y_test_d = np.array(y_test_d)
    y_pred = np.array(test_preds)

    if y_test_d.ndim > 1:
        y_test_d = y_test_d.reshape(-1)
    if y_pred.ndim > 1:
        y_pred = y_pred.reshape(-1)

    tp = {}
    fp = {}
    fn = {}

    classes = np.unique(np.concatenate((y_test_d, y_pred)))

    for c in classes:
        tp[c] = 0
        fp[c] = 0
        fn[c] = 0

    for c in classes:
        for true, pred in zip(y_test_d, y_pred):
            if true == c and pred == c:
                tp[c] += 1
            elif true != c and pred == c:
                fp[c] += 1
            elif true == c and pred != c:
                fn[c] += 1

    precision = {}
    recall = {}
    f1 = {}

    for c in classes:
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0

    if average == 'micro':
        precision_val = np.sum(list(tp.values())) / (np.sum(list(tp.values())) + np.sum(list(fp.values()))) if (np.sum(list(tp.values())) + np.sum(list(fp.values()))) > 0 else 0
        recall_val = np.sum(list(tp.values())) / (np.sum(list(tp.values())) + np.sum(list(fn.values()))) if (np.sum(list(tp.values())) + np.sum(list(fn.values()))) > 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

    elif average == 'macro':
        precision_val = np.mean(list(precision.values()))
        recall_val = np.mean(list(recall.values()))
        f1_val = np.mean(list(f1.values()))

    elif average == 'weighted':
        weights = np.array([np.sum(y_test_d == c) for c in classes])
        weights = weights / np.sum(weights)
        precision_val = np.sum([weights[i] * precision[classes[i]] for i in range(len(classes))])
        recall_val = np.sum([weights[i] * recall[classes[i]] for i in range(len(classes))])
        f1_val = np.sum([weights[i] * f1[classes[i]] for i in range(len(classes))])

    else:
        raise ValueError("Invalid value for 'average'. Choose from 'micro', 'macro', 'weighted'.")

    return precision_val, recall_val, f1_val


def decode_one_hot(encoded_data):
    """
    Decodes one-hot encoded data to original categorical labels.

    Args:
        encoded_data (numpy.ndarray): One-hot encoded data with shape (n_samples, n_classes).

    Returns:
        numpy.ndarray: Decoded categorical labels with shape (n_samples,).
    """

    decoded_labels = np.argmax(encoded_data, axis=1)

    return decoded_labels


def roc_curve(y_true, y_score):
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters:
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by decision_function on some classifiers).

    Returns:
    fpr : array, shape = [n]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr : array, shape = [n]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds : array, shape = [n]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
    """
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Only binary classification is supported.")

    
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]


    fpr = []
    tpr = []
    thresholds = []
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    tp = 0
    fp = 0
    prev_score = None

    
    for i, score in enumerate(y_score):
        if score != prev_score:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
            thresholds.append(score)
            prev_score = score

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / n_neg)
    tpr.append(tp / n_pos)
    thresholds.append(score)

    return np.array(fpr), np.array(tpr), np.array(thresholds)


def confusion_matrix(y_true, y_pred, class_count):
    """
    Computes confusion matrix.

    Args:
        y_true (numpy.ndarray): True class labels (1D array).
        y_pred (numpy.ndarray): Predicted class labels (1D array).
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    confusion = np.zeros((class_count, class_count), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        confusion[true_label, pred_label] += 1

    return confusion


def plot_evaluate(x_test, y_test, y_preds, acc_list, W, activation_potentiation):
    
    
    acc = acc_list[len(acc_list) - 1]
    y_true = decode_one_hot(y_test)

    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    Class = np.unique(decode_one_hot(y_test))

    precision, recall, f1 = metrics(y_test, y_preds)
    
    
    cm = confusion_matrix(y_true, y_preds, len(Class))
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    sns.heatmap(cm, annot=True, fmt='d', ax=axs[0, 0])
    axs[0, 0].set_title("Confusion Matrix")
    axs[0, 0].set_xlabel("Predicted Class")
    axs[0, 0].set_ylabel("Actual Class")
    
    if len(Class) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_preds)
   
        roc_auc = np.trapz(tpr, fpr)
        axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[1, 0].set_xlim([0.0, 1.0])
        axs[1, 0].set_ylim([0.0, 1.05])
        axs[1, 0].set_xlabel('False Positive Rate')
        axs[1, 0].set_ylabel('True Positive Rate')
        axs[1, 0].set_title('Receiver Operating Characteristic (ROC) Curve')
        axs[1, 0].legend(loc="lower right")
        axs[1, 0].legend(loc="lower right")
    else:

        for i in range(len(Class)):
            
            y_true_copy = np.copy(y_true)
            y_preds_copy = np.copy(y_preds)
        
            y_true_copy[y_true_copy == i] = 0
            y_true_copy[y_true_copy != 0] = 1
            
            y_preds_copy[y_preds_copy == i] = 0
            y_preds_copy[y_preds_copy != 0] = 1
            

            fpr, tpr, thresholds = roc_curve(y_true_copy, y_preds_copy)
            
            roc_auc = np.trapz(tpr, fpr)
            axs[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[1, 0].set_xlim([0.0, 1.0])
            axs[1, 0].set_ylim([0.0, 1.05])
            axs[1, 0].set_xlabel('False Positive Rate')
            axs[1, 0].set_ylabel('True Positive Rate')
            axs[1, 0].set_title('Receiver Operating Characteristic (ROC) Curve')
            axs[1, 0].legend(loc="lower right")
            axs[1, 0].legend(loc="lower right")
        
        
        """
            accuracy_per_class = []
        
            for cls in Class:
                correct = np.sum((y_true == cls) & (y_preds == cls))
                total = np.sum(y_true == cls)
                accuracy_cls = correct / total if total > 0 else 0.0
                accuracy_per_class.append(accuracy_cls)
        
            axs[2, 0].bar(Class, accuracy_per_class, color='b', alpha=0.7)
            axs[2, 0].set_xlabel('Class')
            axs[2, 0].set_ylabel('Accuracy')
            axs[2, 0].set_title('Class-wise Accuracy')
            axs[2, 0].set_xticks(Class)
            axs[2, 0].grid(True)
"""



    

    metric = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, acc]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    
    bars = axs[0, 1].bar(metric, values, color=colors)
    
    
    for bar, value in zip(bars, values):
        axs[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{value:.2f}',
                       ha='center', va='bottom', fontsize=12, color='white', weight='bold')
    
    axs[0, 1].set_ylim(0, 1) 
    axs[0, 1].set_xlabel('Metrics')
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].set_title('Precision, Recall, F1 Score, and Accuracy (Weighted)')
    axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    feature_indices=[0, 1]

    h = .02
    x_min, x_max = x_test[:, feature_indices[0]].min() - 1, x_test[:, feature_indices[0]].max() + 1
    y_min, y_max = x_test[:, feature_indices[1]].min() - 1, x_test[:, feature_indices[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_full = np.zeros((grid.shape[0], x_test.shape[1]))
    grid_full[:, feature_indices] = grid
    
    Z = [None] * len(grid_full)

    predict_progress = tqdm(total=len(grid_full),leave=False, desc="Predicts For Desicion Boundary",ncols= 120)

    for i in range(len(grid_full)):

        Z[i] = np.argmax(predict_model_ram(grid_full[i], W=W, activation_potentiation=activation_potentiation))
        predict_progress.update(1)

    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    axs[1,1].contourf(xx, yy, Z, alpha=0.8)
    axs[1,1].scatter(x_test[:, feature_indices[0]], x_test[:, feature_indices[1]], c=decode_one_hot(y_test), edgecolors='k', marker='o', s=20, alpha=0.9)
    axs[1,1].set_xlabel(f'Feature {0 + 1}')
    axs[1,1].set_ylabel(f'Feature {1 + 1}')
    axs[1,1].set_title('Decision Boundary')

    plt.show()


def plot_decision_boundary(ax, x, y, activation_potentiation, W, artist, draw_is_finished=False):
    feature_indices = [0, 1]

    h = .02
    x_min, x_max = x[:, feature_indices[0]].min() - 1, x[:, feature_indices[0]].max() + 1
    y_min, y_max = x[:, feature_indices[1]].min() - 1, x[:, feature_indices[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_full = np.zeros((grid.shape[0], x.shape[1]))
    grid_full[:, feature_indices] = grid
    
    Z = [None] * len(grid_full)

    for i in range(len(grid_full)):
        Z[i] = np.argmax(predict_model_ram(grid_full[i], W=W, activation_potentiation=activation_potentiation))

    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    
    if draw_is_finished == False:

        art1_1 = ax[1, 0].contourf(xx, yy, Z, alpha=0.8)
        art1_2 = ax[1, 0].scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=decode_one_hot(y), edgecolors='k', marker='o', s=20, alpha=0.9)
        ax[1, 0].set_xlabel(f'Feature {0 + 1}')
        ax[1, 0].set_ylabel(f'Feature {1 + 1}')
        ax[1, 0].set_title('Decision Boundary')
        artist.append([*art1_1.collections, art1_2])

    else:

        for i in range(30):

            art1_1 = ax[1, 0].contourf(xx, yy, Z, alpha=0.8)
            art1_2 = ax[1, 0].scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=decode_one_hot(y), edgecolors='k', marker='o', s=20, alpha=0.9)
            ax[1, 0].set_xlabel(f'Feature {0 + 1}')
            ax[1, 0].set_ylabel(f'Feature {1 + 1}')
            ax[1, 0].set_title('Decision Boundary')
            artist.append([*art1_1.collections, art1_2])

    return artist

def pca(X, n_components):
    """
    
    Parameters:
    X (numpy array): (n_samples, n_features)
    n_components (int):
    
    Returns:
    X_reduced (numpy array): (n_samples, n_components)
    """
    
    X_meaned = X - np.mean(X, axis=0)
    
    covariance_matrix = np.cov(X_meaned, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    eigenvectors_subset = sorted_eigenvectors[:, :n_components]
    
    X_reduced = np.dot(X_meaned, eigenvectors_subset)
    
    return X_reduced

def plot_decision_space(x, y, y_preds=None, s=100, color='tab20'):
    
    if x.shape[1] > 2:

        X_pca = pca(x, n_components=2)
    else:
        X_pca = x

    if y_preds == None:
        y_preds = decode_one_hot(y)

    y = decode_one_hot(y)
    num_classes = len(np.unique(y))
    
    cmap = plt.get_cmap(color)


    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=50, cmap=cmap, norm=norm)
    

    for cls in range(num_classes):

        class_points = []


        for i in range(len(y)):
            if y_preds[i] == cls:
                class_points.append(X_pca[i])
        
        class_points = np.array(class_points)
        

        if len(class_points) > 2:
            hull = ConvexHull(class_points)
            hull_points = class_points[hull.vertices]

            hull_points = np.vstack([hull_points, hull_points[0]])
            
            plt.fill(hull_points[:, 0], hull_points[:, 1], color=cmap(norm(cls)), alpha=0.3, edgecolor='k', label=f'Class {cls} Hull')

    plt.title("Decision Space (Data Distribution)")

    plt.draw()
    
    
def manuel_balancer(x_train, y_train, target_samples_per_class):
    """
    Generates synthetic examples to balance classes to the specified number of examples per class.

    Arguments:
    x_train -- Input dataset (examples) - NumPy array format
    y_train -- Class labels (one-hot encoded) - NumPy array format
    target_samples_per_class -- Desired number of samples per class

    Returns:
    x_balanced -- Balanced input dataset (NumPy array format)
    y_balanced -- Balanced class labels (one-hot encoded, NumPy array format)
    """
    try:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    except:
        print(Fore.GREEN + "x_tarin and y_train already numpyarray." + Style.RESET_ALL)
        pass
    classes = np.arange(y_train.shape[1])
    class_count = len(classes)
    
    x_balanced = []
    y_balanced = []

    for class_label in tqdm(range(class_count),leave=False, desc='Augmenting Data',ncols= 120):
        class_indices = np.where(np.argmax(y_train, axis=1) == class_label)[0]
        num_samples = len(class_indices)
        
        if num_samples > target_samples_per_class:
      
            selected_indices = np.random.choice(class_indices, target_samples_per_class, replace=False)
            x_balanced.append(x_train[selected_indices])
            y_balanced.append(y_train[selected_indices])
            
        else:
            
            x_balanced.append(x_train[class_indices])
            y_balanced.append(y_train[class_indices])

            if num_samples < target_samples_per_class:
                
                samples_to_add = target_samples_per_class - num_samples
                additional_samples = np.zeros((samples_to_add, x_train.shape[1]))
                additional_labels = np.zeros((samples_to_add, y_train.shape[1]))
                
                for i in range(samples_to_add):

                    random_indices = np.random.choice(class_indices, 2, replace=False)
                    sample1 = x_train[random_indices[0]]
                    sample2 = x_train[random_indices[1]]

                    
                    synthetic_sample = sample1 + (sample2 - sample1) * np.random.rand()

                    additional_samples[i] = synthetic_sample
                    additional_labels[i] = y_train[class_indices[0]]
                    
                    
                x_balanced.append(additional_samples)
                y_balanced.append(additional_labels)
    
    x_balanced = np.vstack(x_balanced)
    y_balanced = np.vstack(y_balanced)

    return x_balanced, y_balanced

def get_weights():

    return 0


def get_df():

    return 2


def get_preds():

    return 1


def get_acc():

    return 2