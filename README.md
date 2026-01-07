# Fake News Detection Model

A deep learning model built with TensorFlow and Keras to detect fake news articles using LSTM neural networks and GloVe word embeddings. The model processes full article text and achieves 90%+ accuracy in classifying news as REAL or FAKE.

# Project Overview

This project implements a fake news detection system using:
- **Deep Learning**: Dual-layer LSTM networks with CNN layers for feature extraction
- **Word Embeddings**: Pre-trained GloVe (Global Vectors for Word Representation) embeddings
- **Architecture**: Embedding → Conv1D → MaxPooling → Dual LSTM → Dense layers
- **Accuracy**: Achieves 90.33% test accuracy in classifying news as REAL or FAKE
- **Data Collection**: Includes web scraping tools for expanding the training dataset

# Features

- **Web Interface**: Interactive Streamlit web app for real-time predictions
- **Pre-trained Embeddings**: GloVe word embeddings for better text understanding
- **Deep Learning**: Dual-layer CNN-LSTM hybrid architecture for feature extraction and sequence modeling
- **Full Article Processing**: Uses first 500 words of article text (not just headlines)
- **Model Evaluation**: Comprehensive metrics with confusion matrix and classification report
- **Model Persistence**: Save/load trained model for reuse
- **Visualization**: Training history plots and prediction confidence scores
- **Data Collection**: Web scraping tools for expanding training dataset
- **Docker Support**: Containerized deployment for easy setup
- **API Ready**: Python functions for integration into other applications
- **Modular Architecture**: Clean, organized codebase with separate modules for each component

# Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Model evaluation metrics
- **Matplotlib**: Visualization
- **Streamlit**: Web application framework
- **GloVe Embeddings**: Pre-trained word vectors (50-dimensional)
- **python-dotenv**: Environment variable management
- **Docker**: Containerization (optional)

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

3. Set up environment variables (for web scraping):
```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API keys
# NEWS_API_KEY=your_api_key_here
```

4. Download GloVe embeddings:
```bash
wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
```

**Note**: For macOS with Apple Silicon (M1/M2/M3), you may need to install TensorFlow-Metal:
```bash
pip install tensorflow-macos tensorflow-metal
```

**Note**: The `.env` file is not tracked by git for security. Make sure to create your own `.env` file with your API keys.

# Usage

## Training the Model

Run the training script to train the model on the dataset:

```bash
python detector.py
```

This will:
1. Load and preprocess the news dataset (uses article text, first 500 words)
2. Load GloVe word embeddings
3. Build and compile the dual-layer LSTM model
4. Train the model for 50 epochs
5. Evaluate the model performance
6. Save the trained model and tokenizer
7. Generate training history visualization

**Expected Output:**
- Model summary showing architecture
- Training progress for each epoch
- Final training accuracy (~99%) and test accuracy (~90%)
- Confusion matrix and classification report
- Saved files: `fake_news_model.h5`, `tokenizer.pickle`, `training_history.png`

**Configuration**: You can adjust training parameters in `config.py`:
- `TRAINING_SIZE`: Number of samples to use (default: 3000, can use up to 6335)
- `MAX_LENGTH`: Maximum sequence length in words (default: 500)
- `EPOCHS`: Number of training epochs (default: 50)

## Making Predictions

### Option 1: Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and enter news text to get instant predictions!

### Option 2: Command Line

Use the prediction script to classify new news articles:

```bash
python predict.py
```

### Option 3: Python API

Use the prediction function in your code:

```python
from predict import predict_news, load_model

# Load the saved model and tokenizer
model, tokenizer = load_model()

# Predict on new text
text = "Your news article text here"
result, confidence = predict_news(text, model, tokenizer)
print(f"Prediction: {result} (Confidence: {confidence:.4f})")
```

### Option 4: Docker

Run using Docker:

```bash
# Build and run
docker-compose up --build

# Or using Docker directly
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```

## Collecting Additional Training Data

The project includes web scraping tools to expand the training dataset:

### Using NewsAPI Scraper

1. Get a free API key from [NewsAPI.org](https://newsapi.org/)
2. Add it to your `.env` file:
   ```
   NEWS_API_KEY=your_api_key_here
   ```
3. Run the scraper:
   ```bash
   python newscraper.py
   ```
4. This will scrape real news articles from reliable sources (BBC, Reuters, AP)
5. Combine with existing dataset:
   ```python
   import pandas as pd
   existing = pd.read_csv('news.csv')
   scraped = pd.read_csv('scraped_real_news.csv')
   combined = pd.concat([existing, scraped], ignore_index=True)
   combined = combined.sample(frac=1).reset_index(drop=True)
   combined.to_csv('combined_news.csv', index=False)
   ```
6. Update `config.py` to use the combined dataset:
   ```python
   DATA_FILE = 'combined_news.csv'
   TRAINING_SIZE = len(combined)  # Or specific number
   ```

# Model Architecture

```
Input (Text, up to 500 words)
  ↓
Tokenization & Padding (to 500 words)
  ↓
Embedding Layer (GloVe 50d, non-trainable)
  ↓
Dropout (0.3)
  ↓
Conv1D (128 filters, kernel size 5)
  ↓
MaxPooling1D (pool size 4)
  ↓
LSTM (128 units, return_sequences=True)
  ↓
Dropout (0.3)
  ↓
LSTM (64 units)
  ↓
Dropout (0.3)
  ↓
Dense (32 units, ReLU activation)
  ↓
Dense (1 unit, sigmoid activation)
  ↓
Output (REAL/FAKE)
```

**Key Architecture Features:**
- Dual-layer LSTM for better sequence understanding
- Increased Conv1D filters (128) for better feature extraction
- Multiple dropout layers (0.3) for regularization
- Intermediate dense layer for feature refinement

# Model Performance

The model is trained on news articles with the following configuration:
- **Training Set**: 2,700 samples (configurable, up to 6,335 available)
- **Test Set**: 300 samples (10% of training size)
- **Epochs**: 50
- **Embedding Dimension**: 50
- **Max Sequence Length**: 500 words (uses full article text, not just headlines)
- **Input**: First 500 words of article text

**Current Performance:**
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~90.33%
- **Precision**: 0.91 (macro average)
- **Recall**: 0.91 (macro average)
- **F1-Score**: 0.90

Performance metrics are displayed after training, including:
- Training and validation accuracy
- Confusion matrix
- Precision, Recall, and F1-score per class

**Note**: To improve accuracy further, you can:
- Increase `TRAINING_SIZE` to use all available data (6,335 samples)
- Add more training data using the web scraping tools
- Fine-tune hyperparameters in `config.py`

# Project Structure

```
fake-news-detector/
├── detector.py              # Main training script (orchestrates pipeline)
├── predict.py               # Prediction script for new articles
├── app.py                   # Streamlit web application
├── newscraper.py            # Web scraping tool for data collection
├── config.py                # Configuration file (hyperparameters)
├── data_preprocessing.py    # Data loading and preprocessing module
├── embeddings.py            # GloVe embedding loading module
├── model_builder.py         # Model architecture definition
├── trainer.py               # Training and evaluation module
├── visualizer.py            # Training visualization module
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore file
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── tests/                   # Unit tests directory
├── news.csv                 # Dataset (not included in repo)
├── glove.6B.50d.txt         # GloVe word embeddings
├── fake_news_model.h5       # Saved trained model (generated)
├── tokenizer.pickle         # Saved tokenizer (generated)
└── training_history.png     # Training visualization (generated)
```

# Configuration

All configuration parameters are centralized in `config.py`:

**Model Hyperparameters:**
- `EMBEDDING_DIM`: GloVe embedding dimension (default: 50)
- `MAX_LENGTH`: Maximum sequence length in words (default: 500)
- `PADDING_TYPE`: Padding type for sequences (default: 'post')
- `TRUNC_TYPE`: Truncation type for sequences (default: 'post')

**Training Parameters:**
- `TRAINING_SIZE`: Number of samples to use for training (default: 3000, max: 6335)
- `TEST_PORTION`: Fraction of data for testing (default: 0.1)
- `EPOCHS`: Number of training epochs (default: 50)

**File Paths:**
- `DATA_FILE`: Path to training dataset CSV file
- `GLOVE_FILE`: Path to GloVe embeddings file
- `MODEL_FILE`: Path to save trained model
- `TOKENIZER_FILE`: Path to save tokenizer

**Environment Variables:**
- `NEWS_API_KEY`: API key for NewsAPI (for web scraping, stored in `.env` file)

# Dataset

The model uses a fake news dataset with the following structure:
- `title`: News article title
- `text`: Full article text
- `label`: Classification label (FAKE/REAL)

**Dataset Details:**
- Total available samples: 6,335+
- Default training size: configurable
- Uses first 500 words of article text for training
- Labels are encoded: FAKE=0, REAL=1

**Expanding the Dataset:**
- Use all available data: Set `TRAINING_SIZE = 6335` in `config.py`
- Add scraped data: Use `newscraper.py` to collect more articles
- Combine datasets: Merge multiple CSV files with same structure

**Note**: The dataset file (`news.csv`) is not included in this repository. You'll need to obtain it separately here: https://www.geeksforgeeks.org/nlp/fake-news-detection-model-using-tensorflow-in-python/. The repository includes tools to scrape additional training data.

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is open source and available under the MIT License.

# Acknowledgments

- GloVe embeddings by Stanford NLP Group
- TensorFlow/Keras documentation
- GeeksforGeeks tutorials for guidance
- NewsAPI.org for news data API
