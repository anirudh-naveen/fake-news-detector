"""
Word Embeddings Module
Handles loading GloVe embeddings and creating embedding matrix
"""

import numpy as np
import config


def load_glove_embeddings(file_path=config.GLOVE_FILE):
    """
    Load GloVe word embeddings from file.
    
    Args:
        file_path: Path to GloVe embeddings file
    
    Returns:
        dict: Dictionary mapping words to their embedding vectors
    """
    print(f"\nLoading GloVe embeddings from {file_path}...")
    embedding_index = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    
    print(f"Loaded {len(embedding_index)} word vectors from GloVe")
    return embedding_index


def create_embedding_matrix(word_index, embedding_index, vocab_size, 
                            embedding_dim=config.EMBEDDING_DIM):
    """
    Create embedding matrix for vocabulary using GloVe embeddings.
    
    Args:
        word_index: Tokenizer word index dictionary
        embedding_index: GloVe embeddings dictionary
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
    
    Returns:
        numpy.ndarray: Embedding matrix of shape (vocab_size + 1, embedding_dim)
    """
    print("\nCreating embedding matrix...")
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    
    words_found = 0
    for word, i in word_index.items():
        if i <= vocab_size:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                words_found += 1
    
    print(f"Created embedding matrix: {embedding_matrix.shape}")
    print(f"Found GloVe vectors for {words_found}/{vocab_size} words in vocabulary")
    
    return embedding_matrix

