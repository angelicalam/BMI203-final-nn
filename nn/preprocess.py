# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    np.random.seed(42)
    # Given that there are far more positives than negatives, sample
    # positive sequences with replacement and negatives without. 
    negative = seqs[labels==0]
    positive = seqs[labels==1]
    idx_negative = np.random.choice(range(len(negative)), 500, replace=False)
    sampled_seqs = negative[idx_negative]
    sampled_labels = np.zeros(500)
    idx_positive = np.random.choice(range(len(positive)), 500, replace=True)
    sampled_seqs = np.concatenate((sampled_seqs, positive[idx_positive]))
    sampled_labels = np.concatenate((sampled_labels, np.ones(500)))
    # Scramble sampled seqs and labels
    idx = np.random.choice(range(len(sampled_seqs)), 1000, replace=False)
    sampled_seqs = sampled_seqs[idx]
    sampled_labels = sampled_labels[idx]
    
    return sampled_seqs, sampled_labels
    

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    code = {'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1] }
    encodings = []
    for s in seq_arr:
        encodings.append( np.concatenate([code[base] for base in s]) )
        
    return np.matrix(encodings)
        
        