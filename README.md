# Fake News Detection Model

A deep learning model built with TensorFlow and Keras to detect fake news articles using LSTM neural networks and GloVe word embeddings.

# Project Overview

This project implements a fake news detection system using:
- **Deep Learning**: LSTM (Long Short-Term Memory) networks with CNN layers
- **Word Embeddings**: Pre-trained GloVe (Global Vectors for Word Representation) embeddings
- **Architecture**: Embedding → Conv1D → MaxPooling → LSTM → Dense layers
- **Accuracy**: Achieves high accuracy in classifying news as REAL or FAKE

# Features

- Pre-trained GloVe word embeddings for better text understanding
- CNN-LSTM hybrid architecture for feature extraction and sequence modeling
- Comprehensive model evaluation with confusion matrix and classification report
- Model persistence (save/load trained model)
- Training visualization (accuracy and loss plots)
- Easy-to-use prediction function for new articles

# Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Model evaluation metrics
- **Matplotlib**: Visualization
- **GloVe Embeddings**: Pre-trained word vectors (50-dimensional)

# Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd fake-news-detector
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download GloVe embeddings:
```bash
wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
```

**Note**: For macOS with Apple Silicon (M1/M2/M3), you may need to install TensorFlow-Metal:
```bash
pip install tensorflow-macos tensorflow-metal
```

# Usage

## Training the Model

Run the training script to train the model on the dataset:

```bash
python detector.py
```

This will:
1. Load and preprocess the news dataset
2. Load GloVe word embeddings
3. Build and compile the LSTM model
4. Train the model for 50 epochs
5. Evaluate the model performance
6. Save the trained model and tokenizer
7. Generate training history visualization

**Expected Output:**
- Model summary showing architecture
- Training progress for each epoch
- Final training and test accuracy
- Confusion matrix and classification report
- Saved files: `fake_news_model.h5`, `tokenizer.pickle`, `training_history.png`

## Making Predictions

Use the prediction script to classify new news articles:

```bash
python predict.py
```

Or use the prediction function in your code:

```python
from predict import predict_news, load_model

# Load the saved model and tokenizer
model, tokenizer = load_model()

# Predict on new text
text = "Your news article text here"
result, confidence = predict_news(text, model, tokenizer)
print(f"Prediction: {result} (Confidence: {confidence:.4f})")
```

# Model Architecture

```
Input (Text) 
  ↓
Tokenization & Padding
  ↓
Embedding Layer (GloVe 50d, non-trainable)
  ↓
Dropout (0.2)
  ↓
Conv1D (64 filters, kernel size 5)
  ↓
MaxPooling1D (pool size 4)
  ↓
LSTM (64 units)
  ↓
Dense (1 unit, sigmoid activation)
  ↓
Output (REAL/FAKE)
```

# Model Performance

The model is trained on 3,000 news articles with:
- **Training Set**: 2,700 samples
- **Test Set**: 300 samples
- **Epochs**: 50
- **Embedding Dimension**: 50
- **Max Sequence Length**: 54

Performance metrics are displayed after training, including:
- Training and validation accuracy
- Confusion matrix
- Precision, Recall, and F1-score

# Project Structure

```
fake-news-detector/
├── detector.py              # Main training script
├── predict.py               # Prediction script for new articles
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── news.csv                # Dataset (not included in repo)
├── glove.6B.50d.txt        # GloVe word embeddings
├── fake_news_model.h5      # Saved trained model (generated)
├── tokenizer.pickle        # Saved tokenizer (generated)
└── training_history.png    # Training visualization (generated)
```

# Configuration

You can modify these parameters in `detector.py`:

- `training_size`: Number of samples to use for training (default: 3000)
- `test_portion`: Fraction of data for testing (default: 0.1)
- `embedding_dim`: GloVe embedding dimension (default: 50)
- `max_length`: Maximum sequence length (default: 54)
- `epochs`: Number of training epochs (default: 50)

# Dataset

The model uses a fake news dataset with the following structure:
- `title`: News article title
- `text`: Full article text
- `label`: Classification label (FAKE/REAL)

**Note**: The dataset file (`news.csv`) is not included in this repository. You'll need to obtain it separately.

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is open source and available under the MIT License.

# Acknowledgments

- GloVe embeddings by Stanford NLP Group
- TensorFlow/Keras documentation
- GeeksforGeeks tutorials for guidance
