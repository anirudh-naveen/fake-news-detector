"""
Configuration file for Fake News Detection Model
Contains all hyperparameters and settings
"""

# Model hyperparameters
EMBEDDING_DIM = 50
MAX_LENGTH = 500
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
OOV_TOK = "<OOV>"

# Training parameters
TRAINING_SIZE = 3000
TEST_PORTION = 0.1
EPOCHS = 50

# File paths
DATA_FILE = 'news.csv'
GLOVE_FILE = 'glove.6B.50d.txt'
MODEL_FILE = 'fake_news_model.h5'
TOKENIZER_FILE = 'tokenizer.pickle'
HISTORY_PLOT_FILE = 'training_history.png'

