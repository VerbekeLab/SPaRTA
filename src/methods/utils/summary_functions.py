import numpy as np

def summary_functions(piece, aggregation):
    if aggregation == 'sum':
        return piece.sum()
    elif aggregation == 'mean':
        return piece.mean()
    elif aggregation == 'max':
        return piece.max()
    elif aggregation == 'min':
        return piece.min()
    elif aggregation == 'std':
        return piece.std()
    elif aggregation == 'median':
        return np.median(piece)
    elif aggregation == 'iqr':
        q75, q25 = np.percentile(piece, [75 ,25])
        return q75 - q25
    else:
        raise ValueError("Unsupported aggregation method")

def summary_00_function(adj_full, size_0, aggregation='sum'):
    piece_00 = adj_full[:size_0, :size_0]
    return summary_functions(piece_00, aggregation)

def summary_01_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_01 = adj_full[:size_0, size_0:size_0 + size_1]
    return summary_functions(piece_01, aggregation)

def summary_02_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_02 = adj_full[:size_0, size_0 + size_1:]
    return summary_functions(piece_02, aggregation)

def summary_10_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_10 = adj_full[size_0:size_0 + size_1, :size_0]
    return summary_functions(piece_10, aggregation)

def summary_11_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_11 = adj_full[size_0:size_0 + size_1, size_0:size_0 + size_1]
    return summary_functions(piece_11, aggregation)

def summary_12_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_12 = adj_full[size_0:size_0 + size_1, size_0 + size_1:]
    return summary_functions(piece_12, aggregation)

def summary_20_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_20 = adj_full[size_0 + size_1:, :size_0]
    return summary_functions(piece_20, aggregation)

def summary_21_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_21 = adj_full[size_0 + size_1:, size_0:size_0 + size_1]
    return summary_functions(piece_21, aggregation)

def summary_22_function(adj_full, size_0, size_1, aggregation='sum'):
    piece_22 = adj_full[size_0 + size_1:, size_0 + size_1:]
    return summary_functions(piece_22, aggregation)