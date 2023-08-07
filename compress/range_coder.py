import constriction
import numpy as np
from common.tools import get_np_size

def compress_matrix_flatten(matrix):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    if unique.shape[0] == 1:
        return unique.astype(np.uint8), unique
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    probabilities_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(message, probabilities_model)
    compressed = encoder.get_compressed()
    return compressed, unique

def compress_matrix_flatten_new(matrix):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    if unique.shape[0] == 1:
        return unique.astype(np.uint8), unique, 0
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    probabilities_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(message, probabilities_model)
    compressed = encoder.get_compressed()
    return compressed, unique, probabilities

if __name__ == '__main__':
    matrix = np.random.randint(low=0, high=255, size=(128, 128)).astype(np.float32)
    compressed, symtable = compress_matrix_flatten(matrix)
    print(get_np_size(compressed), get_np_size(symtable))
