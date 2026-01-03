"""
Data Preprocessing Module
Handles data loading, cleaning, encoding, and tokenization
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config


def load_and_clean_data(file_path=config.DATA_FILE):
    """
    Load and clean the dataset.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame: Cleaned dataset
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    data = data.drop(["Unnamed: 0"], axis=1)
    print(f"Dataset loaded: {data.shape}")
    print(data.head(5))
    return data


def encode_labels(data):
    """
    Encode categorical labels to numerical values.
    
    Args:
        data: DataFrame with 'label' column
    
    Returns:
        DataFrame: Data with encoded labels
        LabelEncoder: Fitted encoder
    """
    print("\nEncoding labels...")
    le = preprocessing.LabelEncoder()
    le.fit(data['label'])
    data['label'] = le.transform(data['label'])
    print("Labels encoded")
    return data, le


def prepare_data(data, training_size=config.TRAINING_SIZE, use_text=True, max_words=None):
    """
    Extract titles, texts, and labels from dataset.
    
    Args:
        data: DataFrame with 'title', 'text', and 'label' columns
        training_size: Number of samples to use
        use_text: If True, use article text; if False, use only titles
        max_words: Maximum words to use from text (None = use all, or specify number)
    
    Returns:
        tuple: (texts, labels) - texts can be titles or article text
    """
    print(f"\nPreparing {training_size} samples...")
    texts = []
    labels = []
    
    if max_words is None:
        max_words = config.MAX_LENGTH
    
    for x in range(training_size):
        if use_text and 'text' in data.columns:
            # Use article text (truncate to max_words)
            article_text = str(data['text'][x])
            words = article_text.split()[:max_words]
            texts.append(" ".join(words))
        else:
            # Use only title
            texts.append(str(data['title'][x]))
        
        labels.append(data['label'][x])
    
    print(f"Prepared {len(texts)} samples")
    if use_text:
        avg_length = sum(len(t.split()) for t in texts) / len(texts)
        print(f"Average text length: {avg_length:.1f} words")
    return texts, labels


def tokenize_and_pad(texts, padding_type=config.PADDING_TYPE, 
                     trunc_type=config.TRUNC_TYPE, max_length=config.MAX_LENGTH):
    """
    Tokenize text and pad sequences.
    
    Args:
        texts: List of text strings (titles or article text)
        padding_type: Padding type ('pre' or 'post')
        trunc_type: Truncation type ('pre' or 'post')
        max_length: Maximum sequence length
    
    Returns:
        tuple: (tokenizer, padded_sequences, vocab_size)
    """
    print("\nTokenizing text...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Padded sequences shape: {padded.shape}")
    print(f"Max sequence length: {max_length}")
    
    return tokenizer, padded, vocab_size


def split_data(padded_sequences, labels, test_portion=config.TEST_PORTION, 
               training_size=config.TRAINING_SIZE):
    """
    Split data into training and testing sets.
    
    Args:
        padded_sequences: Padded sequences
        labels: List of labels
        test_portion: Fraction of data for testing
        training_size: Total number of samples
    
    Returns:
        tuple: (training_sequences, test_sequences, training_labels, test_labels)
    """
    print("\nSplitting data...")
    split = int(test_portion * training_size)
    
    training_sequences = padded_sequences[split:training_size]
    test_sequences = padded_sequences[0:split]
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]
    
    # Convert to numpy arrays
    training_sequences = np.array(training_sequences)
    test_sequences = np.array(test_sequences)
    
    print(f"Training samples: {len(training_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    
    return training_sequences, test_sequences, training_labels, test_labels

