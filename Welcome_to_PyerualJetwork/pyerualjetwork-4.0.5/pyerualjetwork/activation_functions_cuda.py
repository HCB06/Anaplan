import cupy as cp

# ACTIVATION FUNCTIONS ----

def all_activations():
    
    activations_list = ['linear', 'sigmoid', 'relu', 'tanh', 'circular', 'spiral', 'swish', 'sin_plus', 'mod_circular', 'tanh_circular', 'leaky_relu', 'softplus', 'elu', 'gelu', 'selu', 'sinakt', 'p_squared', 'sglu', 'dlrelu', 'exsig', 'acos', 'gla', 'srelu', 'qelu', 'isra', 'waveakt', 'arctan', 'bent_identity', 'sech', 'softsign', 'pwl', 'cubic', 'gaussian', 'sine', 'tanh_square', 'mod_sigmoid', 'quartic', 'square_quartic', 'cubic_quadratic', 'exp_cubic', 'sine_square', 'logarithmic', 'scaled_cubic', 'sine_offset']
    
    return activations_list

def spiral_activation(x):
    if x.ndim == 1:
        r = cp.sqrt(cp.sum(x**2)) 
        theta = cp.arctan2(x[1], x[0]) 

        spiral_x = r * cp.cos(theta + r) 
        spiral_y = r * cp.sin(theta + r) 

        spiral_output = cp.array([spiral_x, spiral_y])
    else:
        r = cp.sqrt(cp.sum(x**2, axis=-1))
        theta = cp.arctan2(x[:, 1], x[:, 0]) 

        spiral_x = r * cp.cos(theta + r) 
        spiral_y = r * cp.sin(theta + r) 

        spiral_output = cp.stack((spiral_x, spiral_y), axis=-1)

    return spiral_output


def Softmax(x):
    """Optimized Softmax function"""
    return cp.array(cp.exp(x - cp.max(x, axis=-1, keepdims=True)) / cp.sum(cp.exp(x - cp.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True))

def Sigmoid(x):
    """Optimized Sigmoid function"""
    return 1 / (1 + cp.exp(-x))

def Relu(x):
    """Optimized ReLU function"""
    return cp.maximum(0, x)

def tanh(x):
    """Optimized Tanh function"""
    return cp.tanh(x)

def swish(x):
    """Optimized Swish function"""
    return x * Sigmoid(x)

def sin_plus(x):
    """Optimized SinPlus function"""
    return (cp.sin(x) + 1) / 2

def modular_circular_activation(x, period=2*cp.pi):
    """Optimized Modular Circular Activation function"""
    return cp.mod(x, period) / period

def tanh_circular_activation(x):
    """Optimized Tanh Circular Activation function"""
    return (cp.tanh(x) + 1) / 2

def leaky_relu(x, alpha=0.01):
    """Optimized Leaky ReLU function"""
    return cp.where(x > 0, x, alpha * x)

def softplus(x):
    """Optimized Softplus function"""
    return cp.log1p(cp.exp(x))

def elu(x, alpha=1.0):
    """Optimized ELU function"""
    return cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

def gelu(x):
    """Optimized GELU function"""
    return 0.5 * x * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))

def selu(x, lambda_=1.0507, alpha=1.6733):
    """Optimized SELU function"""
    return lambda_ * cp.where(x > 0, x, alpha * (cp.exp(x) - 1))

def sinakt(x):
    """Optimized SinAkt function"""
    return cp.sin(x) + cp.cos(x)

def p_squared(x, alpha=1.0, beta=0.0):
    """Optimized P-squared function"""
    return alpha * x**2 + beta * x

def sglu(x, alpha=1.0):
    """Optimized SGU function"""
    return cp.array(cp.exp(alpha * x)) * x

def dlrelu(x):
    """Optimized Double Leaky ReLU (DLReLU) function"""
    return cp.maximum(0.01 * x, x) + cp.minimum(0.01 * x, 0.1 * x)

def exsig(x):
    """Optimized Exponential Sigmoid (ExSig) function"""
    return 1 / (1 + cp.exp(-x**2))

def acos(x, alpha=1.0, beta=0.0):
    """Optimized Adaptive Cosine Activation (ACos) function"""
    return cp.cos(alpha * x + beta)

def gla(x, alpha=1.0, mu=0.0):
    """Optimized Gaussian-like Activation (GLA) function"""
    return cp.exp(-alpha * (x - mu)**2)

def srelu(x):
    """Optimized Swish ReLU (SReLU) function"""
    return x * (1 / (1 + cp.exp(-x))) + cp.maximum(0, x)

def qelu(x):
    """Optimized Quadratic Exponential Linear Unit (QELU) function"""
    return x**2 * cp.exp(x) - 1

def isra(x):
    """Optimized Inverse Square Root Activation (ISRA) function"""
    return x / cp.sqrt(cp.abs(x) + 1)

def waveakt(x, alpha=1.0, beta=2.0, gamma=3.0):
    """Optimized Wave Activation function"""
    return cp.sin(alpha * x) * cp.cos(beta * x) * cp.sin(gamma * x)

def arctan(x):
    """Optimized Arctan function"""
    return cp.arctan(x)

def bent_identity(x):
    """Optimized Bent Identity function"""
    return (cp.sqrt(x**2 + 1) - 1) / 2 + x

def circular_activation(x, scale=2.0, frequency=1.0, shift=0.0):
    """Optimized Circular Activation function"""
    n_features = x.shape[0]
    circular_output = cp.zeros_like(x)
    
    r = cp.sqrt(cp.sum(x**2))
    for i in range(n_features):
        theta = 2 * cp.pi * (i / n_features) + shift
        circular_x = r * cp.cos(theta + frequency * r) * scale
        circular_y = r * cp.sin(theta + frequency * r) * scale
        
        circular_output[i] = circular_x if i % 2 == 0 else circular_y
    
    return circular_output

def sech(x):
    """Optimized Sech function"""
    return 2 / (cp.exp(x) + cp.exp(-x))

def softsign(x):
    """Optimized Softsign function"""
    return x / (1 + cp.abs(x))

def pwl(x, alpha=0.5, beta=1.5):
    """Optimized Piecewise Linear function (PWL)"""
    return cp.where(x <= 0, alpha * x, beta * x)

def cubic(x):
    """Optimized Cubic function"""
    return x**3

def gaussian(x, alpha=1.0, mu=0.0):
    """Optimized Gaussian function"""
    return cp.exp(-alpha * (x - mu)**2)

def sine(x, alpha=1.0):
    """Optimized Sine function"""
    return cp.sin(alpha * x)

def tanh_square(x):
    """Optimized Tanh Square function"""
    return cp.tanh(x)**2

def mod_sigmoid(x, alpha=1.0, beta=0.0):
    """Optimized Modified Sigmoid function"""
    return 1 / (1 + cp.exp(-alpha * x + beta))

def quartic(x):
    """Optimized Quartic function"""
    return x**4

def square_quartic(x):
    """Optimized Square Quartic function"""
    return (x**2)**2

def cubic_quadratic(x):
    """Optimized Cubic Quadratic function"""
    return x**3 * (x**2)

def exp_cubic(x):
    """Optimized Exponential Cubic function"""
    return cp.exp(x**3)

def sine_square(x):
    """Optimized Sine Square function"""
    return cp.sin(x)**2

def logarithmic(x):
    """Optimized Logarithmic function"""
    return cp.log(x**2 + 1)

def scaled_cubic(x, alpha=1.0):
    """Optimized Scaled Cubic function"""
    return alpha * x**3

def sine_offset(x, beta=0.0):
    """Optimized Sine Offset function"""
    return cp.sin(x + beta)

def apply_activation(Input, activation_list):
   """
    Applies a sequence of activation functions to the input.
    
    Args:
        Input (numpy.ndarray): The input to apply activations to.
        activation_list (list): A list of activation function names to apply.
    
    Returns:
        numpy.ndarray: The input after all activations have been applied.
   """
    
   origin_input = cp.copy(Input)

   for i in range(len(activation_list)):

      if activation_list[i] == 'sigmoid':
         Input += Sigmoid(origin_input)

      elif activation_list[i] == 'swish':
         Input += swish(origin_input)

      elif activation_list[i] == 'mod_circular':
         Input += modular_circular_activation(origin_input)

      elif activation_list[i] == 'tanh_circular':
         Input += tanh_circular_activation(origin_input)

      elif activation_list[i] == 'leaky_relu':
         Input += leaky_relu(origin_input)

      elif activation_list[i] == 'relu':
         Input += Relu(origin_input)

      elif activation_list[i] == 'softplus':
         Input += softplus(origin_input)

      elif activation_list[i] == 'elu':
         Input += elu(origin_input)

      elif activation_list[i] == 'gelu':
         Input += gelu(origin_input)

      elif activation_list[i] == 'selu':
         Input += selu(origin_input)

      elif activation_list[i] == 'tanh':
         Input += tanh(origin_input)

      elif activation_list[i] == 'sinakt':
         Input += sinakt(origin_input)

      elif activation_list[i] == 'p_squared':
         Input += p_squared(origin_input)

      elif activation_list[i] == 'sglu':
         Input += sglu(origin_input, alpha=1.0)

      elif activation_list[i] == 'dlrelu':
         Input += dlrelu(origin_input)

      elif activation_list[i] == 'exsig':
         Input += exsig(origin_input)

      elif activation_list[i] == 'sin_plus':
         Input += sin_plus(origin_input)

      elif activation_list[i] == 'acos':
         Input += acos(origin_input, alpha=1.0, beta=0.0)

      elif activation_list[i] == 'gla':
         Input += gla(origin_input, alpha=1.0, mu=0.0)

      elif activation_list[i] == 'srelu':
         Input += srelu(origin_input)

      elif activation_list[i] == 'qelu':
         Input += qelu(origin_input)

      elif activation_list[i] == 'isra':
         Input += isra(origin_input)

      elif activation_list[i] == 'waveakt':
         Input += waveakt(origin_input)

      elif activation_list[i] == 'arctan':
         Input += arctan(origin_input)

      elif activation_list[i] == 'bent_identity':
         Input += bent_identity(origin_input)

      elif activation_list[i] == 'sech':
         Input += sech(origin_input)

      elif activation_list[i] == 'softsign':
         Input += softsign(origin_input)

      elif activation_list[i] == 'pwl':
         Input += pwl(origin_input)

      elif activation_list[i] == 'cubic':
         Input += cubic(origin_input)

      elif activation_list[i] == 'gaussian':
         Input += gaussian(origin_input)

      elif activation_list[i] == 'sine':
         Input += sine(origin_input)

      elif activation_list[i] == 'tanh_square':
         Input += tanh_square(origin_input)

      elif activation_list[i] == 'mod_sigmoid':
         Input += mod_sigmoid(origin_input)

      elif activation_list[i] == 'linear':
         Input += origin_input

      elif activation_list[i] == 'quartic':
         Input += quartic(origin_input)

      elif activation_list[i] == 'square_quartic':
         Input += square_quartic(origin_input)

      elif activation_list[i] == 'cubic_quadratic':
         Input += cubic_quadratic(origin_input)

      elif activation_list[i] == 'exp_cubic':
         Input += exp_cubic(origin_input)

      elif activation_list[i] == 'sine_square':
         Input += sine_square(origin_input)

      elif activation_list[i] == 'logarithmic':
         Input += logarithmic(origin_input)

      elif activation_list[i] == 'scaled_cubic':
         Input += scaled_cubic(origin_input, 1.0)

      elif activation_list[i] == 'sine_offset':
         Input += sine_offset(origin_input, 1.0)

      elif activation_list[i] == 'spiral':
         Input += spiral_activation(origin_input)

      elif activation_list[i] == 'circular':
         Input += circular_activation(origin_input)

   return Input