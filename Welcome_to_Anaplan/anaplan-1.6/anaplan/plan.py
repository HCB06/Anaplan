# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:32:16 2024\n\n

@author: Hasan Can Beydili\n
@YouTube: https://www.youtube.com/@HasanCanBeydili\n
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/\n
@Instagram: https://www.instagram.com/canbeydili.06/\n\n

@contact: tchasancan@gmail.com
"""

import pandas as pd
import numpy as np
from colorama import Fore, Style, init
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
import sys
import math
import pickle
try:
    from moviepy.editor import VideoFileClip
    import sounddevice as sd
    import soundfile as sf
    import pkg_resources
except:
    pass


GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"

bar_format = f"{GREEN}{{bar}}{GREEN} {RESET} {{l_bar}} {{remaining}} {{postfix}}"
bar_format_learner = f"{GREEN}{{bar}}{GREEN} {RESET} {{remaining}} {{postfix}}"


def speak(message):

    message = pkg_resources.resource_filename('anaplan', f'{message}')
    video = VideoFileClip(message + ".mp4")

    audio = video.audio
    audio.write_audiofile("extracted_audio.wav")

    data, samplerate = sf.read("extracted_audio.wav")
    sd.play(data, samplerate)
    sd.wait()

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

def neuron_history(LTPW, ax1, row, col, class_count, artist5, data, fig1, acc=False, loss=False):

    for j in range(len(class_count)):
        
            if acc != False and loss != False:
                suptitle_info = data + ' Accuracy:' + str(acc) + '\n' + data + ' Loss:' + str(loss) + '\nNeurons Memory:'
            else:
                suptitle_info = 'Neurons Memory:'

            mat = LTPW[j,:].reshape(row, col)

            title_info = f'{j+1}. Neuron'
            
            art5 = ax1[j].imshow(mat, interpolation='sinc', cmap='viridis')

            ax1[j].set_aspect('equal')
            ax1[j].set_xticks([])
            ax1[j].set_yticks([])
            ax1[j].set_title(title_info)

           
            artist5.append([art5])

    fig1.suptitle(suptitle_info, fontsize=16)

    return artist5



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


# ACTIVATION FUNCTIONS -----

def spiral_activation(x):

    r = np.sqrt(np.sum(x**2))
    
    theta = np.arctan2(x[1:], x[:-1])

    spiral_x = r * np.cos(theta + r)
    spiral_y = r * np.sin(theta + r)
    

    spiral_output = np.concatenate(([spiral_x[0]], spiral_y))
    
    return spiral_output


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

def sobel_activation(image):
   
    import cv2

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    
    
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    

    sobel_magnitude = (sobel_magnitude / np.max(sobel_magnitude)) * 255
    return np.ravel(sobel_magnitude.astype(np.uint8))

def fast_activation_vector(image):

    import cv2
   
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image


    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(image_gray, None)


    mask = np.zeros_like(image_gray)
    for kp in keypoints:
        x, y = map(int, kp.pt)
        mask[y, x] = 255

    fast_vector = mask.flatten()

    return fast_vector

def gabor_activation_vector(image, frequency=0.6):

    import cv2

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    
 
    gabor_filter_real, gabor_filter_imag = cv2.getGaborKernel((21, 21), 8.0, 0, frequency, 0.5, 0, ktype=cv2.CV_32F), cv2.getGaborKernel((21, 21), 8.0, np.pi/2, frequency, 0.5, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image_gray, cv2.CV_64F, gabor_filter_real)
    

    filtered_image = (filtered_image / np.max(filtered_image)) * 255
    gabor_vector = filtered_image.flatten()
    
    return gabor_vector


def canny_activation_vector(image):
  
    import cv2

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
  
    canny_edges = cv2.Canny(image_gray, 100, 200)
    
 
    canny_vector = canny_edges.flatten()
    
    return canny_vector

def sift_activation_vector(image):

    import cv2

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    sift = cv2.SIFT_create()
    
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    if descriptors is not None:
        sift_vector = descriptors.flatten()
    else:
        sift_vector = np.array([])
    
    return sift_vector


def laplacian_activation_vector(image):

    import cv2

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    

    laplacian_edges = cv2.Laplacian(image_gray, cv2.CV_64F)
    
    laplacian_edges = np.abs(laplacian_edges)
    laplacian_edges = (laplacian_edges / np.max(laplacian_edges)) * 255
    
    laplacian_vector = laplacian_edges.flatten()
    
    return laplacian_vector


def tanh(x):
    return np.tanh(x)

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

def sin_plus(x):
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

def sinakt(x):
    return np.sin(x) + np.cos(x)

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

def circular_activation(x, scale=2.0, frequency=1.0, shift=0.0):    
    
    n_features = x.shape[0]
    
    spiral_output = np.zeros_like(x)
    
    for i in range(n_features):
        
        r = np.sqrt(np.sum(x**2))
        theta = 2 * np.pi * (i / n_features) + shift
        
        spiral_x = r * np.cos(theta + frequency * r) * scale
        spiral_y = r * np.sin(theta + frequency * r) * scale
        
        if i % 2 == 0:
            spiral_output[i] = spiral_x
        else:
            spiral_output[i] = spiral_y
    
    return spiral_output

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

def scaled_cubic(x, alpha=1.0):
    return alpha * x**3

def sine_offset(x, beta=0.0):
    return np.sin(x + beta)

def activations_list():

    activations_list = ['linear', 'spiral', 'sigmoid', 'relu', 'tanh', 'swish', 'sin_plus', 'circular', 'mod_circular', 'tanh_circular', 'leaky_relu', 'softplus', 'elu', 'gelu', 'selu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'exsig',  'acos',  'gla',  'srelu', 'qelu',  'isra',  'waveakt', 'arctan', 'bent_identity', 'sech',  'softsign',  'pwl', 'cubic',  'gaussian',  'sine', 'tanh_square', 'mod_sigmoid',  'quartic', 'square_quartic',  'cubic_quadratic',  'exp_cubic',  'sine_square', 'logarithmic',  'scaled_cubic', 'sine_offset', 'sift', 'fast', 'gabor', 'laplacian', 'canny', 'sobel']

    print('All avaliable activations: ',  activations_list, "\n\nYOU CAN COMBINE EVERY ACTIVATION. EXAMPLE: ['linear', 'tanh'] or ['waveakt', 'linear', 'sine'].")

    return activations_list

# LOSS FUNCTIONS (OPTIONAL)

def categorical_crossentropy(y_true_batch, y_pred_batch):
    epsilon = 1e-7
    y_pred_batch = np.clip(y_pred_batch, epsilon, 1. - epsilon)
    
    losses = -np.sum(y_true_batch * np.log(y_pred_batch), axis=1)

    mean_loss = np.mean(losses)
    return mean_loss


def binary_crossentropy(y_true_batch, y_pred_batch):
    epsilon = 1e-7
    y_pred_batch = np.clip(y_pred_batch, epsilon, 1. - epsilon)
    
    losses = -np.mean(y_true_batch * np.log(y_pred_batch) + (1 - y_true_batch) * np.log(1 - y_pred_batch), axis=1)

    mean_loss = np.mean(losses)
    return mean_loss

def draw_activations(x_train, activation):


    if activation == 'sigmoid':
        result = Sigmoid(x_train)

    elif activation == 'swish':
        result = swish(x_train)

    elif activation == 'circular':
        result = circular_activation(x_train)

    elif activation == 'mod_circular':
        result = modular_circular_activation(x_train)

    elif activation == 'tanh_circular':
        result = tanh_circular_activation(x_train)

    elif activation == 'leaky_relu':
        result = leaky_relu(x_train)

    elif activation == 'relu':
        result = Relu(x_train)

    elif activation == 'softplus':
        result = softplus(x_train)

    elif activation == 'elu':
        result = elu(x_train)

    elif activation == 'gelu':
        result = gelu(x_train)

    elif activation == 'selu':
        result = selu(x_train)

    elif activation == 'softmax':
        result = Softmax(x_train)

    elif activation == 'tanh':
        result = tanh(x_train)

    elif activation == 'sinakt':
        result = sinakt(x_train)

    elif activation == 'p_squared':
        result = p_squared(x_train)

    elif activation == 'sglu':
        result = sglu(x_train, alpha=1.0)

    elif activation == 'dlrelu':
        result = dlrelu(x_train)

    elif activation == 'exsig':
        result = exsig(x_train)

    elif activation == 'sin_plus':
        result = sin_plus(x_train)

    elif activation == 'acos':
        result = acos(x_train, alpha=1.0, beta=0.0)

    elif activation == 'gla':
        result = gla(x_train, alpha=1.0, mu=0.0)

    elif activation == 'srelu':
        result = srelu(x_train)

    elif activation == 'qelu':
        result = qelu(x_train)

    elif activation == 'isra':
        result = isra(x_train)

    elif activation == 'waveakt':
        result = waveakt(x_train)

    elif activation == 'arctan':
        result = arctan(x_train)

    elif activation == 'bent_identity':
        result = bent_identity(x_train)

    elif activation == 'sech':
        result = sech(x_train)

    elif activation == 'softsign':
        result = softsign(x_train)

    elif activation == 'pwl':
        result = pwl(x_train)

    elif activation == 'cubic':
        result = cubic(x_train)

    elif activation == 'gaussian':
        result = gaussian(x_train)

    elif activation == 'sine':
        result = sine(x_train)

    elif activation == 'tanh_square':
        result = tanh_square(x_train)

    elif activation == 'mod_sigmoid':
        result = mod_sigmoid(x_train)

    elif activation == 'linear':
        result = x_train

    elif activation == 'quartic':
        result = quartic(x_train)

    elif activation == 'square_quartic':
        result = square_quartic(x_train)

    elif activation == 'cubic_quadratic':
        result = cubic_quadratic(x_train)

    elif activation == 'exp_cubic':
        result = exp_cubic(x_train)

    elif activation == 'sine_square':
        result = sine_square(x_train)

    elif activation == 'logarithmic':
        result = logarithmic(x_train)

    elif activation == 'scaled_cubic':
        result = scaled_cubic(x_train, 1.0)

    elif activation == 'sine_offset':
        result = sine_offset(x_train, 1.0)

    elif activation == 'spiral':
        result = spiral_activation(x_train)

    elif activation == 'sobel':
        result = sobel_activation(x_train)

    elif activation == 'canny':
        result = canny_activation_vector(x_train)

    elif activation == 'laplacian':
        result = laplacian_activation_vector(x_train)

    elif activation == 'gabor':
        result = gabor_activation_vector(x_train)

    elif activation == 'fast':
        result = fast_activation_vector(x_train)

    elif activation == 'sift':
        result = sift_activation_vector(x_train)

    return result


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


def draw_neural_web(W, ax, G, return_objs=False):
    """
    Visualizes a neural web by drawing the neural network structure.

    Parameters:
    W : numpy.ndarray
        A 2D array representing the connection weights of the neural network.
    ax : matplotlib.axes.Axes
        The matplotlib axes where the graph will be drawn.
    G : networkx.Graph
        The NetworkX graph representing the neural network structure.
    return_objs : bool, optional
        If True, returns the drawn objects (nodes and edges). Default is False.

    Returns:
    art1 : matplotlib.collections.PathCollection or None
        Returns the node collection if return_objs is True; otherwise, returns None.
    art2 : matplotlib.collections.LineCollection or None
        Returns the edge collection if return_objs is True; otherwise, returns None.
    art3 : matplotlib.collections.TextCollection or None
        Returns the label collection if return_objs is True; otherwise, returns None.

    Example:
    art1, art2, art3 = draw_neural_web(W, ax, G, return_objs=True)
    plt.show()
    """

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                G.add_edge(f'Output{i}', f'Input{j}', ltpw=W[i, j])

    edges = G.edges(data=True)
    weights = [edata['ltpw'] for _, _, edata in edges]
    pos = {}
    num_motor_neurons = W.shape[0]
    num_sensory_neurons = W.shape[1]

    for j in range(num_sensory_neurons):
        pos[f'Input{j}'] = (0, j)

    motor_y_start = (num_sensory_neurons - num_motor_neurons) / 2
    for i in range(num_motor_neurons):
        pos[f'Output{i}'] = (1, motor_y_start + i) 


    art1 = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='lightblue')
    art2 = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
    art3 = nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Neural Web')

    if return_objs == True:

        return art1, art2, art3


def learner(x_train, y_train, x_test=None, y_test=None, strategy='accuracy', batch_size=1, neural_web_history=False, show_current_activations=False, auto_normalization=True, neurons_history=False, patience=None, depth=None, early_shifting=False, early_stop=False, loss='categorical_crossentropy', show_history=False, interval=33.33, target_acc=None, target_loss=None, except_this=None, only_this=None, start_this=None):
    
    """
    Optimizes the activation functions for a neural network by leveraging train data to find the most accurate combination of activation potentiation for the given dataset.
    This next-generation generalization function includes an advanced learning feature that is specifically tailored to the PLAN algorithm.
    It uniquely adjusts hyperparameters based on test accuracy while training with model-specific training data, offering an unparalleled optimization technique.
    Designed to be used before model evaluation. This called TFL(Test Feedback Learning).

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
        neurons_history (bool, optional): Shows the history of changes that neurons undergo during the TFL (Test Feedback Learning) stages. True or False. Default is False.
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

    Output = np.zeros(len(Input))
   
    for activation in activation_potentiation:

        if activation == 'sigmoid':
            Output += Sigmoid(Input)

        elif activation == 'swish':
            Output += swish(Input)

        elif activation == 'circular':
            Output += circular_activation(Input)

        elif activation == 'mod_circular':
            Output += modular_circular_activation(Input)

        elif activation == 'tanh_circular':
            Output += tanh_circular_activation(Input)

        elif activation == 'leaky_relu':
            Output += leaky_relu(Input)

        elif activation == 'relu':
            Output += Relu(Input)
        
        elif activation == 'softplus':
            Output += softplus(Input)

        elif activation == 'elu':
            Output += elu(Input)

        elif activation == 'gelu':
            Output += gelu(Input)

        elif activation == 'selu':
            Output += selu(Input)    

        elif activation == 'softmax':
            Output += Softmax(Input)

        elif activation == 'tanh':
            Output += tanh(Input)

        elif activation == 'sinakt':
            Output += sinakt(Input)

        elif activation == 'p_squared':
            Output += p_squared(Input)

        elif activation == 'sglu':
            Output += sglu(Input, alpha=1.0)

        elif activation == 'dlrelu':
            Output += dlrelu(Input)

        elif activation == 'exsig':
            Output += exsig(Input)

        elif activation == 'sin_plus':
            Output += sin_plus(Input)

        elif activation == 'acos':
            Output += acos(Input, alpha=1.0, beta=0.0)

        elif activation == 'gla':
            Output += gla(Input, alpha=1.0, mu=0.0)

        elif activation == 'srelu':
            Output += srelu(Input)

        elif activation == 'qelu':
            Output += qelu(Input)

        elif activation == 'isra':
            Output += isra(Input)

        elif activation == 'waveakt':
            Output += waveakt(Input) 

        elif activation == 'arctan':
            Output += arctan(Input) 
        
        elif activation == 'bent_identity':
            Output += bent_identity(Input)

        elif activation == 'sech':
            Output += sech(Input)

        elif activation == 'softsign':
            Output += softsign(Input)

        elif activation == 'pwl':
            Output += pwl(Input)

        elif activation == 'cubic':
            Output += cubic(Input)

        elif activation == 'gaussian':
            Output += gaussian(Input)

        elif activation == 'sine':
            Output += sine(Input)

        elif activation == 'tanh_square':
            Output += tanh_square(Input)

        elif activation == 'mod_sigmoid':
            Output += mod_sigmoid(Input)

        elif activation == 'linear':
            Output += Input

        elif activation == 'quartic':
            Output += quartic(Input)

        elif activation == 'square_quartic':
            Output += square_quartic(Input)

        elif activation == 'cubic_quadratic':
            Output += cubic_quadratic(Input)

        elif activation == 'exp_cubic':
            Output += exp_cubic(Input)

        elif activation == 'sine_square':
            Output += sine_square(Input)

        elif activation == 'logarithmic':
            Output += logarithmic(Input)

        elif activation == 'scaled_cubic':
            Output += scaled_cubic(Input, 1.0)

        elif activation == 'sine_offset':
            Output += sine_offset(Input, 1.0)

        elif activation == 'spiral':
            Output += spiral_activation(Input)

        elif activation == 'sobel':
            Output += sobel_activation(Input)

        elif activation == 'canny':
            Output += canny_activation_vector(Input)
        
        elif activation == 'laplacian':
            Output += laplacian_activation_vector(Input)

        elif activation == 'gabor':
            Output += gabor_activation_vector(Input)
        
        elif activation == 'fast':
            Output += fast_activation_vector(Input)
        
        elif activation == 'sift':
            Output += sift_activation_vector(Input)

        else:
            print(Fore.RED + '\nERROR120:' + '"' + activation + '"' + 'is not available. Please enter this code for avaliable activation function list: plan.activations_list()' + '' + Style.RESET_ALL)
            sys.exit()


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

    MaxAbs = np.max(np.abs(Input))  # Direkt maksimumu hesapla
    return (Input / MaxAbs)  # Normalizasyonu geri döndür


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


def save_model(model_name,
               W,
               scaler_params=None,
               model_type='PLAN',
               test_acc=None,
               model_path='',
               activation_potentiation=['linear'],
               weights_type='npy',
               weights_format='raw',
               show_architecture=None
               ):

    """
    Function to save a potentiation learning artificial neural network model.

    Arguments:
    model_name (str): Name of the model.
    model_type (str): Type of the model. default: 'PLAN'
    test_acc (float): Test accuracy of the model. default: None
    weights_type (str): Type of weights to save (options: 'txt', 'pkl', 'npy', 'mat'). default: 'npy'
    WeightFormat (str): Format of the weights (options: 'f', 'raw'). default: 'raw'
    model_path (str): Path where the model will be saved. For example: C:/Users/beydili/Desktop/denemePLAN/ default: ''
    scaler_params (list[num, num]): standard scaler params list: mean,std. If not used standard scaler then be: None.
    W: Weights of the model.
    activation_potentiation (list): For deeper PLAN networks, activation function parameters. For more information please run this code: plan.activations_list() default: ['linear']
    show_architecture (str): It draws model architecture. Takes 2 value='basic' or 'detailed'. Default: None(not drawing)

    Returns:
    str: Message indicating if the model was saved successfully or encountered an error.
    """

    class_count = W.shape[0]

    if test_acc != None:
        test_acc= float(test_acc)

    if weights_type != 'txt' and weights_type != 'npy' and weights_type != 'mat' and weights_type != 'pkl':
        print(Fore.RED + "ERROR110: Save Weight type (File Extension) Type must be 'txt' or 'npy' or 'mat' or 'pkl' from: save_model" + Style.RESET_ALL)
        sys.exit()

    if weights_format != 'd' and weights_format != 'f' and weights_format != 'raw':
        print(Fore.RED + "ERROR111: Weight Format Type must be 'd' or 'f' or 'raw' from: save_model" + Style.RESET_ALL)
        sys.exit()

    NeuronCount = 0
    SynapseCount = 0


    try:
            NeuronCount += np.shape(W)[0] + np.shape(W)[1]
            SynapseCount += np.shape(W)[0] * np.shape(W)[1]
    except:

        print(Fore.RED + "ERROR: Weight matrices has a problem from: save_model" + Style.RESET_ALL)
        sys.exit()

    if scaler_params != None:

        if len(scaler_params) > len(activation_potentiation):

            activation_potentiation += ['']

        elif len(activation_potentiation) > len(scaler_params):

            for i in range(len(activation_potentiation) - len(scaler_params)):

                scaler_params.append(' ')

    data = {'MODEL NAME': model_name,
            'MODEL TYPE': model_type,
            'CLASS COUNT': class_count,
            'NEURON COUNT': NeuronCount,
            'SYNAPSE COUNT': SynapseCount,
            'TEST ACCURACY': test_acc,
            'SAVE DATE': datetime.now(),
            'WEIGHTS TYPE': weights_type,
            'WEIGHTS FORMAT': weights_format,
            'MODEL PATH': model_path,
            'STANDARD SCALER': scaler_params,
            'ACTIVATION POTENTIATION': activation_potentiation
            }
    try:

        df = pd.DataFrame(data)
        df.to_pickle(model_path + model_name + '.pkl')

    except:

        print(Fore.RED + "ERROR: Model log not saved probably model_path incorrect. Check the log parameters from: save_model" + Style.RESET_ALL)
        sys.exit()

    try:

        if weights_type == 'txt' and weights_format == 'f':

                np.savetxt(model_path + model_name + '_weights.txt',  W, fmt='%f')

        if weights_type == 'txt' and weights_format == 'raw':

                np.savetxt(model_path + model_name + '_weights.txt',  W)

        ###

        
        if weights_type == 'pkl' and weights_format == 'f':

            with open(model_path + model_name + '_weights.pkl', 'wb') as f:
                pickle.dump(W.astype(float), f)

        if weights_type == 'pkl' and weights_format =='raw':
        
            with open(model_path + model_name + '_weights.pkl', 'wb') as f:
                pickle.dump(W, f)

        ###

        if weights_type == 'npy' and weights_format == 'f':

                np.save(model_path + model_name + '_weights.npy',  W, W.astype(float))

        if weights_type == 'npy' and weights_format == 'raw':

                np.save(model_path + model_name + '_weights.npy',  W)

        ###

        if weights_type == 'mat' and weights_format == 'f':

                w = {'w': W.astype(float)}
                io.savemat(model_path + model_name + '_weights.mat', w)

        if weights_type == 'mat' and weights_format == 'raw':
                
                w = {'w': W}
                io.savemat(model_path + model_name + '_weights.mat', w)

    except:

        print(Fore.RED + "ERROR: Model Weights not saved. Check the Weight parameters. SaveFilePath expl: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: save_model" + Style.RESET_ALL)
        sys.exit()

    print(df)
    message = (
        Fore.GREEN + "Model Saved Successfully\n" +
        Fore.MAGENTA + "Don't forget, if you want to load model: model log file and weight files must be in the same directory." +
        Style.RESET_ALL
    )

    if show_architecture is not None:
        draw_model_architecture(model_name=model_name, model_path=model_path, style=show_architecture)


    return print(message)

def draw_model_architecture(model_name, model_path='', style='basic'):
    """
    Visualizes the architecture of a neural network model.

    Parameters
    ----------
    model_name : str
        The name of the model to be visualized, which will be displayed in the title or label.
    
    model_path : str
        The file path to the model, from which the architecture is loaded. Default is ''
    
    style : str, optional
        The style of the visualization. 
        Options:
            - 'basic': Displays a simplified view of the model architecture.
            - 'detailed': Shows a more comprehensive view, including layer details and parameters.
        Default is 'basic'.
    
    Returns
    -------
    None
        Draws and displays the architecture of the specified model.

        
    Examples
    --------
    >>> draw_model_architecture("MyModel", "path/to/model", style='detailed')
    """
    model = load_model(model_name=model_name, model_path=model_path)
    
    W = model[get_weights()]
    activation_potentiation = model[get_act_pot()]
    scaler_params = model[get_scaler()]

    text_1 = f"Input Shape:\n{W.shape[1]}"
    text_2 = f"Output Shape:\n{W.shape[0]}"

    if scaler_params is None:
        bottom_left_text = 'Standard Scaler=No'
    else:
        bottom_left_text = 'Standard Scaler=Yes'

    if len(activation_potentiation) != 1 or (len(activation_potentiation) == 1 and activation_potentiation[0] != 'linear'):

        bottom_left_text_1 = f'Aggregation Layers(Aggregates All Conversions)={len(activation_potentiation)}'

    else:

        bottom_left_text_1 = 'Aggregation Layers(Aggregates All Conversions)=0'

    bottom_left_text_2 = 'Potentiation Layer(Fully Connected)=1'

    if scaler_params is None:
        bottom_left_text = 'Standard Scaler=No'
    else:
        bottom_left_text = 'Standard Scaler=Yes'

    num_middle_axes = len(activation_potentiation)

    if style == 'detailed':

        col = 1

    elif style == 'basic':
    
        col = 2

    fig, axes = plt.subplots(1, num_middle_axes + col, figsize=(5 * (num_middle_axes + 2), 5))

    fig.suptitle("Model Architecture", fontsize=16, fontweight='bold')

    for i, activation in enumerate(activation_potentiation):
        x = np.linspace(-100, 100, 100)
        translated_x_train = draw_activations(x, activation)
        y = translated_x_train

        axes[i].plot(x, y, color='b', markersize=6, linewidth=2, label='Activations Over Depth')
        axes[i].set_title(activation_potentiation[i])

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        

        if i < num_middle_axes - 1:
            axes[i].annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), 
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
    
    if style == 'detailed':
    
        G = nx.Graph()
        draw_neural_web(W=W, ax=axes[num_middle_axes], G=G)

    elif style == 'basic':
    
        circle1 = plt.Circle((0.5, 0.5), 0.4, color='skyblue', ec='black', lw=1.5)
        axes[num_middle_axes].add_patch(circle1)
        axes[num_middle_axes].text(0.5, 0.5, text_1, ha='center', va='center', fontsize=12)
        axes[num_middle_axes].set_xlim(0, 1)
        axes[num_middle_axes].set_ylim(0, 1)
        axes[num_middle_axes].axis('off') 

        circle2 = plt.Circle((0.5, 0.5), 0.4, color='lightcoral', ec='black', lw=1.5)
        axes[-1].add_patch(circle2)
        axes[-1].text(0.5, 0.5, text_2, ha='center', va='center', fontsize=12)
        axes[-1].set_xlim(0, 1)
        axes[-1].set_ylim(0, 1)
        axes[-1].axis('off') 
  
     
    fig.text(0.01, 0, bottom_left_text, ha='left', va='bottom', fontsize=10)
    fig.text(0.01, 0.04, bottom_left_text_1, ha='left', va='bottom', fontsize=10)
    fig.text(0.01, 0.08, bottom_left_text_2, ha='left', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


def load_model(model_name,
               model_path,
               ):
    """
   Function to load a potentiation learning model.

   Arguments:
   model_name (str): Name of the model.
   model_path (str): Path where the model is saved.

   Returns:
   lists: W(list[num]), activation_potentiation, DataFrame of the model
    """
    np.set_printoptions(threshold=np.Infinity)

    try:

         df = pd.read_pickle(model_path + model_name + '.pkl')

    except:

        print(Fore.RED + "ERROR: Model Path error. acceptable form: 'C:/Users/hasancanbeydili/Desktop/denemePLAN/' from: load_model" + Style.RESET_ALL)

        sys.exit()

    activation_potentiation = list(df['ACTIVATION POTENTIATION'])
    activation_potentiation = [x for x in activation_potentiation if not (isinstance(x, float) and np.isnan(x))]
    activation_potentiation = [item for item in activation_potentiation if item != ''] 

    scaler_params = df['STANDARD SCALER'].tolist()
    
    try:
        if scaler_params[0] == None:
            scaler_params = scaler_params[0]

    except:
        scaler_params = [item for item in scaler_params if isinstance(item, np.ndarray)]

     
    model_name = str(df['MODEL NAME'].iloc[0])
    WeightType = str(df['WEIGHTS TYPE'].iloc[0])

    if WeightType == 'txt':
            W = np.loadtxt(model_path + model_name + '_weights.txt')
    elif WeightType == 'npy':
            W = np.load(model_path + model_name + '_weights.npy')
    elif WeightType == 'mat':
            W = sio.loadmat(model_path + model_name + '_weights.mat')
    elif WeightType == 'pkl':
        with open(model_path + model_name + '_weights.pkl', 'rb') as f:
            W = pickle.load(f)
    else:

        raise ValueError(
            Fore.RED + "Incorrect weight type value. Value must be 'txt', 'npy', 'pkl' or 'mat' from: load_model." + Style.RESET_ALL)
        
    if WeightType == 'mat':
        W = W['w']

    return W, None, None, activation_potentiation, scaler_params


def predict_model_ssd(Input, model_name, model_path):

    """
    Function to make a prediction using a divided potentiation learning artificial neural network (PLAN).

    Arguments:
    Input (list or ndarray): Input data for the model (single vector or single matrix).
    model_name (str): Name of the model.
    Returns:
    ndarray: Output from the model.
    """
    model = load_model(model_name, model_path)
    
    activation_potentiation = model[get_act_pot()]
    scaler_params = model[get_scaler()]
    W = model[get_weights()]

    Input = standard_scaler(None, Input, scaler_params)


    Wc = np.copy(W)

    neural_layer = Input
    neural_layer = np.array(neural_layer)
    neural_layer = neural_layer.ravel()

    
    neural_layer = fex(neural_layer, W, is_training=False, Class='?', activation_potentiation=activation_potentiation)

    W = np.copy(Wc)
    return neural_layer


def predict_model_ram(Input, W, scaler_params=None, activation_potentiation=['linear']):

    """
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

    Input = standard_scaler(None, Input, scaler_params)

    Wc = np.copy(W)
    
    try:
        
        neural_layer = Input
        neural_layer = np.array(neural_layer)
        neural_layer = neural_layer.ravel()

        neural_layer = fex(neural_layer, W, is_training=False, Class='?', activation_potentiation=activation_potentiation)

        W = np.copy(Wc)
        return neural_layer
    
    except:
        print(Fore.RED + "ERROR: Unexpected input or wrong model parameters from: predict_model_ram." + Style.RESET_ALL)
        sys.exit()


def auto_balancer(x_train, y_train):

    """
   Function to balance the training data across different classes.

   Arguments:
   x_train (list): Input data for training.
   y_train (list): Labels corresponding to the input data.

   Returns:
   tuple: A tuple containing balanced input data and labels.
    """
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
        for i in tqdm(range(class_count),leave=False, ascii="▱▰",
            bar_format= bar_format, desc='Balancing Data',ncols=70):
            if len(ClassIndices[i]) > MinCount:
                SelectedIndices = np.random.choice(
                    ClassIndices[i], MinCount, replace=False)
            else:
                SelectedIndices = ClassIndices[i]
            BalancedIndices.extend(SelectedIndices)

        BalancedInputs = [x_train[idx] for idx in BalancedIndices]
        BalancedLabels = [y_train[idx] for idx in BalancedIndices]

        permutation = np.random.permutation(len(BalancedInputs))
        BalancedInputs = np.array(BalancedInputs)[permutation]
        BalancedLabels = np.array(BalancedLabels)[permutation]

        print(Fore.GREEN + "Data Succesfully Balanced from: " + str(len(x_train)
                                                                                 ) + " to: " + str(len(BalancedInputs)) + ". from: auto_balancer " + Style.RESET_ALL)
    except:
        print(Fore.RED + "ERROR: Inputs and labels must be same length check parameters")
        sys.exit()

    return np.array(BalancedInputs), np.array(BalancedLabels)


def synthetic_augmentation(x_train, y_train):
    """
    Generates synthetic examples to balance classes with fewer examples.

    Arguments:
    x -- Input dataset (examples) - array format
    y -- Class labels (one-hot encoded) - array format

    Returns:
    x_balanced -- Balanced input dataset (array format)
    y_balanced -- Balanced class labels (one-hot encoded, array format)
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


    for class_label in tqdm(range(class_count), leave=False, ascii="▱▰",
            bar_format=bar_format,desc='Augmenting Data',ncols= 52):
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


def standard_scaler(x_train=None, x_test=None, scaler_params=None):
    """
    Standardizes training and test datasets. x_test may be None.

    Args:
        train_data: numpy.ndarray
        test_data: numpy.ndarray (optional)
        scaler_params (optional for using model)

    Returns:
        list:
        Scaler parameters: mean and std
        tuple
        Standardized training and test datasets
    """

    try:

        x_train = x_train.tolist()
        x_test = x_test.tolist()

    except:

        pass

    if x_train != None and scaler_params == None and x_test != None:

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        
        train_data_scaled = (x_train - mean) / std
        test_data_scaled = (x_test - mean) / std

        train_data_scaled = np.nan_to_num(train_data_scaled, nan=0)
        test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)

        scaler_params = [mean, std]

        return scaler_params, train_data_scaled, test_data_scaled
    
    try:
        if scaler_params == None and x_train == None and x_test != None:
            
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)
            train_data_scaled = (x_train - mean) / std
            
            train_data_scaled = np.nan_to_num(train_data_scaled, nan=0)
            
            scaler_params = [mean, std]
            
            return scaler_params, train_data_scaled
    except:

        # this model is not scaled

        return x_test
                
    if scaler_params != None:

        try:

            test_data_scaled = (x_test - scaler_params[0]) / scaler_params[1]
            test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)

        except:

            test_data_scaled = (x_test - scaler_params[0]) / scaler_params[1]
            test_data_scaled = np.nan_to_num(test_data_scaled, nan=0)
        
        return test_data_scaled


def encode_one_hot(y_train, y_test=None, summary=False):
    """
    Performs one-hot encoding on y_train and y_test data.

    Args:
        y_train (numpy.ndarray): Train label data.
        y_test (numpy.ndarray): Test label data. (optional).
        summary (bool): If True, prints the class-to-index mapping. Default: False

    Returns:
        tuple: One-hot encoded y_train ve (eğer varsa) y_test verileri.
    """
    classes = np.unique(y_train)
    class_count = len(classes)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    if summary:
        print("Class-to-index mapping:")
        for cls, idx in class_to_index.items():
            print(f"  {idx}: {cls}")

    y_train_encoded = np.zeros((y_train.shape[0], class_count))
    for i, label in enumerate(y_train):
        y_train_encoded[i, class_to_index[label]] = 1

    if y_test is not None:
        y_test_encoded = np.zeros((y_test.shape[0], class_count))
        for i, label in enumerate(y_test):
            y_test_encoded[i, class_to_index[label]] = 1
        return y_train_encoded, y_test_encoded

    return y_train_encoded


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

    try:

        grid_full = np.zeros((grid.shape[0], x_test.shape[1]))
        grid_full[:, feature_indices] = grid
        
        Z = [None] * len(grid_full)

        predict_progress = tqdm(total=len(grid_full),leave=False, ascii="▱▰",
            bar_format=bar_format,desc="Predicts For Decision Boundary",ncols= 65)

        for i in range(len(grid_full)):

            Z[i] = np.argmax(predict_model_ram(grid_full[i], W=W, activation_potentiation=activation_potentiation))
            predict_progress.update(1)

        predict_progress.close()

        Z = np.array(Z)
        Z = Z.reshape(xx.shape)

        axs[1,1].contourf(xx, yy, Z, alpha=0.8)
        axs[1,1].scatter(x_test[:, feature_indices[0]], x_test[:, feature_indices[1]], c=decode_one_hot(y_test), edgecolors='k', marker='o', s=20, alpha=0.9)
        axs[1,1].set_xlabel(f'Feature {0 + 1}')
        axs[1,1].set_ylabel(f'Feature {1 + 1}')
        axs[1,1].set_title('Decision Boundary')

    except:
        pass

    plt.show()


def plot_decision_boundary(x, y, activation_potentiation, W, artist=None, ax=None):
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

    if ax is None:

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=decode_one_hot(y), edgecolors='k', marker='o', s=20, alpha=0.9)
        plt.xlabel(f'Feature {0 + 1}')
        plt.ylabel(f'Feature {1 + 1}')
        plt.title('Decision Boundary')

        plt.show()

    else:

        try:
            art1_1 = ax[1, 0].contourf(xx, yy, Z, alpha=0.8)
            art1_2 = ax[1, 0].scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=decode_one_hot(y), edgecolors='k', marker='o', s=20, alpha=0.9)
            ax[1, 0].set_xlabel(f'Feature {0 + 1}')
            ax[1, 0].set_ylabel(f'Feature {1 + 1}')
            ax[1, 0].set_title('Decision Boundary')

            return art1_1, art1_2
        
        except:

            art1_1 = ax[0].contourf(xx, yy, Z, alpha=0.8)
            art1_2 = ax[0].scatter(x[:, feature_indices[0]], x[:, feature_indices[1]], c=decode_one_hot(y), edgecolors='k', marker='o', s=20, alpha=0.9)
            ax[0].set_xlabel(f'Feature {0 + 1}')
            ax[0].set_ylabel(f'Feature {1 + 1}')
            ax[0].set_title('Decision Boundary')


            return art1_1, art1_2

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
        pass

    classes = np.arange(y_train.shape[1])
    class_count = len(classes)
    
    x_balanced = []
    y_balanced = []

    for class_label in tqdm(range(class_count),leave=False, ascii="▱▰",
            bar_format=bar_format,desc='Augmenting Data',ncols= 52):
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