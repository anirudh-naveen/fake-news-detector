"""
Model Builder Module
Defines and compiles the neural network architecture
"""

import tensorflow as tf
import config


def build_model(vocab_size, embedding_matrix, embedding_dim=config.EMBEDDING_DIM, 
                max_length=config.MAX_LENGTH):
    """
    Build the LSTM-based fake news detection model.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained embedding matrix
        embedding_dim: Dimension of word embeddings
        max_length: Maximum sequence length
    
    Returns:
        Compiled TensorFlow model
    """
    print("\n=== Building Model Architecture ===")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size + 1, 
            embedding_dim, 
            input_length=max_length, 
            weights=[embedding_matrix], 
            trainable=False
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model

