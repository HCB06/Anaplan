from tqdm import tqdm
import cupy as cp
from colorama import Fore, Style
import sys
import math
import numpy as np

def encode_one_hot(y_train, y_test=None, summary=False):
    """
    Performs one-hot encoding on y_train and y_test data.

    Args:
        y_train (cupy.ndarray): Train label data.
        y_test (cupy.ndarray): Test label data. (optional).
        summary (bool): If True, prints the class-to-index mapping. Default: False
   
    Returns:
        tuple: One-hot encoded y_train and (if given) y_test.
    """

    if len(y_train) < 256:
        if y_train.dtype != cp.uint8:
            y_train = cp.array(y_train, copy=False).astype(cp.uint8, copy=False)
    elif len(y_train) <= 32767:
        if y_train.dtype != cp.uint16:
            y_train = cp.array(y_train, copy=False).astype(cp.uint16, copy=False)
    else:
        if y_train.dtype != cp.uint32:
            y_train = cp.array(y_train, copy=False).astype(cp.uint32, copy=False)

    if y_test is not None:
        if len(y_test) < 256:
            if y_test.dtype != cp.uint8:
                y_test = cp.array(y_test, copy=False).astype(cp.uint8, copy=False)
        elif len(y_test) <= 32767:
            if y_test.dtype != cp.uint16:
                y_test = cp.array(y_test, copy=False).astype(cp.uint16, copy=False)
        else:
            if y_test.dtype != cp.uint32:
                y_test = cp.array(y_test, copy=False).astype(cp.uint32, copy=False)

    y_train = y_train.get()
    y_test = y_test.get()

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
        return cp.array(y_train_encoded), cp.array(y_test_encoded)

    return cp.array(y_train_encoded)


def decode_one_hot(encoded_data):
    """
    Decodes one-hot encoded data to original categorical labels.

    Args:
        encoded_data (cupy.ndarray): One-hot encoded data with shape (n_samples, n_classes).

    Returns:
        cupy.ndarray: Decoded categorical labels with shape (n_samples,).
    """

    decoded_labels = cp.argmax(encoded_data, axis=1)

    return decoded_labels


def split(X, y, test_size, random_state, dtype=cp.float32):
    """
    Splits the given X (features) and y (labels) data into training and testing subsets.

    Args:
        X (cupy.ndarray): Features data.
        y (cupy.ndarray): Labels data.
        test_size (float or int): Proportion or number of samples for the test subset.
        random_state (int or None): Seed for random state.
        dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        tuple: x_train, x_test, y_train, y_test as ordered training and testing data subsets.
    """
    X = cp.array(X, copy=False).astype(dtype, copy=False)
    if len(y) < 256:
        if y.dtype != cp.uint8:
            y = cp.array(y, copy=False).astype(cp.uint8, copy=False)
    elif len(y) <= 32767:
        if y.dtype != cp.uint16:
            y = cp.array(y, copy=False).astype(cp.uint16, copy=False)
    else:
        if y.dtype != cp.uint32:
            y = cp.array(y, copy=False).astype(cp.uint32, copy=False)

    
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
        cp.random.seed(random_state)

    indices = cp.arange(num_samples)
    cp.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test


def manuel_balancer(x_train, y_train, target_samples_per_class, dtype=cp.float32):
    """
    Generates synthetic examples to balance classes to the specified number of examples per class.

    Arguments:
    x_train -- Input dataset (examples) - cupy array format
    y_train -- Class labels (one-hot encoded) - cupy array format
    target_samples_per_class -- Desired number of samples per class
    dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
    x_balanced -- Balanced input dataset (cupy array format)
    y_balanced -- Balanced class labels (one-hot encoded, cupy array format)
    """
    from .ui import loading_bars

    bar_format = loading_bars()[0]
    
    x_train = cp.array(x_train, copy=False).astype(dtype, copy=False)

    if len(y_train[0]) < 256:
        if y_train.dtype != cp.uint8:
            y_train = cp.array(y_train, copy=False).astype(cp.uint8, copy=False)
    elif len(y_train[0]) <= 32767:
        if y_train.dtype != cp.uint16:
            y_train = cp.array(y_train, copy=False).astype(cp.uint16, copy=False)
    else:
        if y_train.dtype != cp.uint32:
            y_train = cp.array(y_train, copy=False).astype(cp.uint32, copy=False)


    classes = cp.arange(y_train.shape[1])
    class_count = len(classes)
    
    x_balanced = []
    y_balanced = []

    for class_label in tqdm(range(class_count),leave=False, ascii="▱▰",
            bar_format=bar_format,desc='Augmenting Data',ncols= 52):
        class_indices = cp.where(cp.argmax(y_train, axis=1) == class_label)[0]
        num_samples = len(class_indices)
        
        if num_samples > target_samples_per_class:
      
            selected_indices = cp.random.choice(class_indices, target_samples_per_class, replace=False)
            x_balanced.append(x_train[selected_indices])
            y_balanced.append(y_train[selected_indices])
            
        else:
            
            x_balanced.append(x_train[class_indices])
            y_balanced.append(y_train[class_indices])

            if num_samples < target_samples_per_class:
                
                samples_to_add = target_samples_per_class - num_samples
                additional_samples = cp.zeros((samples_to_add, x_train.shape[1]))
                additional_labels = cp.zeros((samples_to_add, y_train.shape[1]))
                
                for i in range(samples_to_add):

                    random_indices = cp.random.choice(class_indices, 2, replace=False)
                    sample1 = x_train[random_indices[0]]
                    sample2 = x_train[random_indices[1]]

                    
                    synthetic_sample = sample1 + (sample2 - sample1) * cp.random.rand()

                    additional_samples[i] = synthetic_sample
                    additional_labels[i] = y_train[class_indices[0]]
                    
                    
                x_balanced.append(additional_samples)
                y_balanced.append(additional_labels)
    
    x_balanced = cp.vstack(x_balanced)
    y_balanced = cp.vstack(y_balanced)

    return x_balanced, y_balanced


def auto_balancer(x_train, y_train, dtype=cp.float32):

    """
   Function to balance the training data across different classes.

   Arguments:
   x_train (list): Input data for training.
   
   y_train (list): Labels corresponding to the input data.
   
   dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

   Returns:
   tuple: A tuple containing balanced input data and labels.
    """
    from .ui import loading_bars

    x_train = cp.array(x_train, copy=False).astype(dtype, copy=False)

    if len(y_train[0]) < 256:
        if y_train.dtype != cp.uint8:
            y_train = cp.array(y_train, copy=False).astype(cp.uint8, copy=False)
    elif len(y_train[0]) <= 32767:
        if y_train.dtype != cp.uint16:
            y_train = cp.array(y_train, copy=False).astype(cp.uint16, copy=False)
    else:
        if y_train.dtype != cp.uint32:
            y_train = cp.array(y_train, copy=False).astype(cp.uint32, copy=False)

    bar_format = loading_bars()[0]
    
    classes = cp.arange(y_train.shape[1])
    class_count = len(classes)
    
    try:
        ClassIndices = {i: cp.where(cp.array(y_train)[:, i] == 1)[
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
                SelectedIndices = cp.random.choice(
                    ClassIndices[i], MinCount, replace=False)
            else:
                SelectedIndices = ClassIndices[i]
            BalancedIndices.extend(SelectedIndices)

        BalancedInputs = [x_train[idx] for idx in BalancedIndices]
        BalancedLabels = [y_train[idx] for idx in BalancedIndices]

        permutation = cp.random.permutation(len(BalancedInputs))
        BalancedInputs = cp.array(BalancedInputs)[permutation]
        BalancedLabels = cp.array(BalancedLabels)[permutation]

        print(Fore.GREEN + "Data Succesfully Balanced from: " + str(len(x_train)
                                                                                 ) + " to: " + str(len(BalancedInputs)) + ". from: auto_balancer " + Style.RESET_ALL)
    except:
        print(Fore.RED + "ERROR: Inputs and labels must be same length check parameters")
        sys.exit()

    return BalancedInputs, BalancedLabels


def synthetic_augmentation(x_train, y_train, dtype=cp.float32):
    """
    Generates synthetic examples to balance classes with fewer examples using CuPy.
    Arguments:
    
    x_train -- Input dataset (examples) - cupy array format
    
    y_train -- Class labels (one-hot encoded) - cupy array format
    
    dtype (cupy.dtype): Data type for the arrays. np.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)
    
    Returns:
    x_train_balanced -- Balanced input dataset (cupy array format)
    y_train_balanced -- Balanced class labels (one-hot encoded, cupy array format)
    """
    from .ui import loading_bars

    bar_format = loading_bars()[0]
    
    x = x_train.astype(dtype, copy=False)

    if len(y_train[0]) < 256:
        if y_train.dtype != cp.uint8:
            y_train = cp.array(y_train, copy=False).astype(cp.uint8, copy=False)
    elif len(y_train[0]) <= 32767:
        if y_train.dtype != cp.uint16:
            y_train = cp.array(y_train, copy=False).astype(cp.uint16, copy=False)
    else:
        if y_train.dtype != cp.uint32:
            y_train = cp.array(y_train, copy=False).astype(cp.uint32, copy=False)

    y = y_train

    classes = cp.arange(y_train.shape[1])
    class_count = len(classes)
    class_distribution = {i: 0 for i in range(class_count)}
    
    y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else y
    for label in y_cpu:
        class_distribution[cp.argmax(label).item()] += 1
        
    max_class_count = max(class_distribution.values())
    x_balanced = list(x)
    y_balanced = list(y)
    
    for class_label in tqdm(range(class_count), leave=False, ascii="▱▰",
            bar_format=bar_format, desc='Augmenting Data', ncols=52):
        class_indices = [i for i, label in enumerate(y) if cp.argmax(label) == class_label]
        num_samples = len(class_indices)
        
        if num_samples < max_class_count:
            while num_samples < max_class_count:
                random_indices = cp.random.choice(
                    cp.array(class_indices), 2, replace=False)
                sample1 = x[random_indices[0]]
                sample2 = x[random_indices[1]]
                
                synthetic_sample = sample1 + \
                    (sample2 - sample1) * cp.random.rand()
                
                x_balanced.append(synthetic_sample)
                y_balanced.append(y[class_indices[0]])
                num_samples += 1
    
    x_balanced = cp.array(x_balanced)
    y_balanced = cp.array(y_balanced)
    
    return x_balanced, y_balanced

def standard_scaler(x_train=None, x_test=None, scaler_params=None, dtype=cp.float32):
    """
    Standardizes training and test datasets. x_test may be None.

    Args:
        x_train: cupy.ndarray
        
        x_test: cupy.ndarray (optional)
        
        scaler_params (optional for using model)
        
        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        list:
        Scaler parameters: mean and std
        tuple
        Standardized training and test datasets
    """
    if x_train is not None and scaler_params is None and x_test is not None:
        x_train = x_train.astype(dtype, copy=False)
        x_test = x_test.astype(dtype, copy=False)

        mean = cp.mean(x_train, axis=0)
        std = cp.std(x_train, axis=0)
        
        train_data_scaled = (x_train - mean) / std
        test_data_scaled = (x_test - mean) / std

        train_data_scaled = cp.nan_to_num(train_data_scaled, nan=0)
        test_data_scaled = cp.nan_to_num(test_data_scaled, nan=0)

        scaler_params = [mean, std]

        return scaler_params, train_data_scaled, test_data_scaled
    
    if scaler_params is None and x_train is None and x_test is not None:
        return x_test.astype(dtype, copy=False)  # sample data not scaled
            
    if scaler_params is not None:
        x_test = x_test.astype(dtype, copy=False)
        scaled_data = (x_test - scaler_params[0]) / scaler_params[1]
        scaled_data = cp.nan_to_num(scaled_data, nan=0)

        return scaled_data  # sample data scaled
    
    
def normalization(
    Input,  # num: Input data to be normalized.
    dtype=cp.float32
):
    """
    Normalizes the input data using maximum absolute scaling.

    Args:
        Input (num): Input data to be normalized.

        dtype (cupy.dtype): Data type for the arrays. cp.float32 by default. Example: cp.float64 or cp.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

    Returns:
        (num) Scaled input data after normalization.
    """

    MaxAbs = cp.max(cp.abs(Input.astype(dtype, copy=False)))
    return (Input / MaxAbs)


def find_closest_factors(a):

    root = int(math.sqrt(a))
    
    for i in range(root, 0, -1):
        if a % i == 0:
            j = a // i
            return i, j
        

def batcher(x_test, y_test, batch_size=1):
    y_labels = cp.argmax(y_test, axis=1)  # Sınıf etiketlerini belirle
    
    unique_labels = cp.unique(y_labels)  # Tüm sınıfları bul
    total_samples = sum(
        int(cp.sum(y_labels == class_label) * batch_size) for class_label in unique_labels
    )
    
    sampled_x = cp.empty((total_samples, x_test.shape[1]), dtype=x_test.dtype)
    sampled_y = cp.empty((total_samples, y_test.shape[1]), dtype=y_test.dtype)
    
    offset = 0
    for class_label in unique_labels:
        # Sınıfa ait indeksleri bulun
        class_indices = cp.where(y_labels == class_label)[0]
        
        # Örnek sayısını belirle
        num_samples = int(len(class_indices) * batch_size)
        
        # Rastgele örnek seç
        sampled_indices = cp.random.choice(class_indices, num_samples, replace=False)
        
        # Veriyi sampled dizilerine yaz
        sampled_x[offset:offset + num_samples] = x_test[sampled_indices]
        sampled_y[offset:offset + num_samples] = y_test[sampled_indices]
        
        # Kaydırmayı güncelle
        offset += num_samples
    
    return sampled_x, sampled_y