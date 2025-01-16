import psutil
import numpy as np
import cupy as cp
import gc
import logging

def get_available_memory():
    """
    The function `get_available_memory` returns the amount of available memory in the system using the
    `psutil` library.
    :return: The function `get_available_memory()` returns the amount of available memory in bytes on
    the system.
    """
    memory = psutil.virtual_memory().available
    return memory

def get_optimal_batch_size_for_cpu(x, data_size_bytes, available_memory):
    """
    The function calculates the optimal batch size for a given data size and available memory based on
    the size of each element.
    
    :param x: `x` is a NumPy array representing the input data for which we want to determine the
    optimal batch size for processing on the CPU
    :param data_size_bytes: The `data_size_bytes` parameter represents the size of the data in bytes
    that you want to process in batches
    :param available_memory: The `available_memory` parameter represents the total memory available on
    the CPU in bytes. This function calculates the optimal batch size for processing data based on the
    provided parameters. Let me know if you need any further assistance or explanation!
    :return: the optimal batch size for a given array `x` based on the available memory and the size of
    each element in bytes.
    """
    safe_memory = available_memory * 0.25
    element_size = data_size_bytes / x.size
    return int(safe_memory / (element_size * 2))

def transfer_to_cpu(x, dtype=np.float32):
    """
    The `transfer_to_cpu` function converts data to a specified data type on the CPU, handling memory constraints
    by batching the conversion process.
    
    :param x: The `x` parameter in the `transfer_to_cpu` function is the input data that you want to transfer to
    the CPU. It can be either a NumPy array or any other data structure that supports the `get` method
    for retrieving the data
    :param dtype: The `dtype` parameter in the `transfer_to_cpu` function specifies the data type to which the
    input array `x` should be converted before moving it to the CPU. By default, it is set to
    `np.float32`, which is a 32-bit floating-point number data type in NumPy
    :return: The `transfer_to_cpu` function returns the processed data in NumPy array format with the specified
    data type (`dtype`). If the input `x` is already a NumPy array with the same data type as specified,
    it returns `x` as is. Otherwise, it converts the input data to the specified data type and returns
    the processed NumPy array.
    """
    try:
        if isinstance(x, np.ndarray):
            return x.astype(dtype) if x.dtype != dtype else x
            
        data_size = x.nbytes
        available_memory = get_available_memory()
        
        logging.debug(f"Data size: {data_size/1e6:.2f}MB, Available memory: {available_memory/1e6:.2f}MB")
        
        if data_size <= available_memory * 0.25:
            final_result = np.array(x.get(), dtype=dtype, copy=False)
            del x
            cp.get_default_memory_pool().free_all_blocks()
            return final_result
            
        batch_size = get_optimal_batch_size_for_cpu(x, data_size, available_memory)
        logging.debug(f"Using batch size: {batch_size}")
        
        result = []
        total_batches = (x.size + batch_size - 1) // batch_size
        
        for i in range(0, x.size, batch_size):
            try:
                chunk = x[i:i + batch_size]
                result.append(np.array(chunk.get(), dtype=dtype))
                del chunk
                
                if i > 0 and i % (batch_size * 10) == 0:
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                raise
                
        final_result = np.concatenate(result)
        del x
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        return final_result
        
    except Exception as e:
        logging.error(f"Error in transfer_to_cpu: {str(e)}")
        raise

def get_optimal_batch_size_for_gpu(x, data_size_bytes):
    """
    The function calculates the optimal batch size for a GPU based on available memory and data size.
    
    :param x: A list or array containing the data elements that will be processed on the GPU
    :param data_size_bytes: The `data_size_bytes` parameter represents the total size of the data in
    bytes that you want to process on the GPU. This could be the size of a single batch of data or the
    total size of the dataset, depending on how you are structuring your computations
    :return: the optimal batch size that can be used for processing the given data on the GPU, based on
    the available free memory on the GPU and the size of the data elements.
    """
    free_memory = cp.get_default_memory_pool().free_bytes()
    device_memory = cp.cuda.runtime.memGetInfo()[0]
    safe_memory = min(free_memory, device_memory) * 0.25
    
    element_size = data_size_bytes / len(x)
    return int(safe_memory / (element_size * 2))


def transfer_to_gpu(x, dtype=cp.float32):
    """
    The `transfer_to_gpu` function in Python converts input data to GPU arrays, optimizing memory usage by
    batching and handling out-of-memory errors.
    
    :param x: The `x` parameter in the `transfer_to_gpu` function is the input data that you want to transfer to
    the GPU for processing. It can be either a NumPy array or a CuPy array. If it's a NumPy array, the
    function will convert it to a CuPy array and
    :param dtype: The `dtype` parameter in the `transfer_to_gpu` function specifies the data type to which the
    input array `x` should be converted when moving it to the GPU. By default, it is set to
    `cp.float32`, which is a 32-bit floating-point data type provided by the Cu
    :return: The `transfer_to_gpu` function returns the input data `x` converted to a GPU array of type `dtype`
    (default is `cp.float32`). If the input `x` is already a GPU array with the same dtype, it returns
    `x` as is. If the data size of `x` exceeds 25% of the free GPU memory, it processes the data in
    batches to
    """
    
    try:
        if isinstance(x, cp.ndarray):
            return x.astype(dtype) if x.dtype != dtype else x
            
        data_size = x.nbytes
        free_gpu_memory = cp.cuda.runtime.memGetInfo()[0]
        
        logging.debug(f"Data size: {data_size/1e6:.2f}MB, Free GPU memory: {free_gpu_memory/1e6:.2f}MB")
        
        if data_size <= free_gpu_memory * 0.25:
            new_x = cp.array(x, dtype=dtype, copy=False)
            del x
            gc.collect()
            return new_x
            
        batch_size = get_optimal_batch_size_for_gpu(x, data_size)
        logging.debug(f"Using batch size: {batch_size}")
        
        result = []
        total_batches = (len(x) + batch_size - 1) // batch_size
        
        for i in range(0, len(x), batch_size):
            try:
                chunk = x[i:i + batch_size]
                gpu_chunk = cp.array(chunk, dtype=dtype)
                result.append(gpu_chunk)
                
                del chunk

                if i > 0 and i % (batch_size * 5) == 0:
                    pool = cp.get_default_memory_pool()
                    if pool.used_bytes() > free_gpu_memory * 0.75:
                        pool.free_all_blocks()
                    gc.collect()
                    
            except cp.cuda.memory.OutOfMemoryError:
                logging.error(f"GPU out of memory at batch {i//batch_size + 1}/{total_batches}")
                cp.get_default_memory_pool().free_all_blocks()
                batch_size = max(batch_size // 2, 1)
                continue
                
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {str(e)}")
                raise
                
        try:
            final_result = cp.concatenate(result)
            del result
            del x
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            return final_result
            
        except Exception as e:
            logging.error(f"Error concatenating results: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Error in transfer_to_gpu: {str(e)}")
        raise
