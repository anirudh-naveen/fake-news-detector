import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config


def load_model():
    """
    Load the saved model and tokenizer.
    
    Returns:
        model: Loaded TensorFlow model
        tokenizer: Loaded tokenizer
    """
    try:
        # Load the model
        print("Loading model...")
        model = tf.keras.models.load_model(config.MODEL_FILE)
        print("Model loaded successfully")
        
        # Load the tokenizer
        print("Loading tokenizer...")
        with open(config.TOKENIZER_FILE, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully\n")
        
        return model, tokenizer
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first by running: python detector.py")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def predict_news(text, model, tokenizer, max_length=config.MAX_LENGTH):
    """
    Predict if a news article is fake or real.
    
    Args:
        text (str): News article text or title to classify
        model: Trained TensorFlow model
        tokenizer: Fitted tokenizer
        max_length (int): Maximum sequence length (default: 54)
    
    Returns:
        tuple: (prediction_label, confidence_score)
            - prediction_label: "REAL" or "FAKE"
            - confidence_score: Float between 0 and 1
    """
    if model is None or tokenizer is None:
        return None, None
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad sequences
    sequences = pad_sequences(
        sequences, 
        maxlen=max_length, 
        padding=config.PADDING_TYPE, 
        truncating=config.TRUNC_TYPE
    )
    
    # Make prediction
    prediction = model.predict(sequences, verbose=0)[0][0]
    
    # Determine label
    label = "REAL" if prediction >= 0.5 else "FAKE"
    
    return label, prediction


def predict_multiple(texts, model, tokenizer):
    """
    Predict on multiple news articles at once.
    
    Args:
        texts (list): List of news article texts
        model: Trained TensorFlow model
        tokenizer: Fitted tokenizer
    
    Returns:
        list: List of tuples (text, label, confidence)
    """
    results = []
    for text in texts:
        label, confidence = predict_news(text, model, tokenizer)
        results.append((text, label, confidence))
    return results


def main():
    """Main function for interactive prediction."""
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return
    
    print("=" * 60)
    print("Fake News Detection - Interactive Mode")
    print("=" * 60)
    print("Enter news article titles or text to classify.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user input
        user_input = input("Enter news text (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        # Make prediction
        label, confidence = predict_news(user_input, model, tokenizer)
        
        if label:
            print(f"\n{'='*60}")
            print(f"Text: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"{'='*60}\n")
        else:
            print("Error making prediction.\n")


if __name__ == "__main__":
    # Example usage
    print("Fake News Detection - Prediction Script\n")
    
    # Load model
    model, tokenizer = load_model()
    
    if model and tokenizer:
        # Example predictions
        print("=" * 60)
        print("Example Predictions:")
        print("=" * 60)
        
        test_cases = [
            "Karry to go to France in gesture of sympathy",
            "Scientists discover new planet with potential for life",
            "Breaking: Major breakthrough in cancer research announced today",
            "You won't believe what happened next! Click here now!"
        ]
        
        for text in test_cases:
            label, confidence = predict_news(text, model, tokenizer)
            print(f"\nText: {text}")
            print(f"Prediction: {label} (Confidence: {confidence:.4f})")
        
        print("\n" + "=" * 60)
        print("Starting interactive mode...")
        print("=" * 60 + "\n")
        
        # Start interactive mode
        main()
    else:
        print("\nPlease train the model first by running: python detector.py")

