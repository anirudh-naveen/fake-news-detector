"""
Training Module
Handles model training, evaluation, and saving
"""

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import config


def train_model(model, training_sequences, training_labels, 
                test_sequences, test_labels, epochs=config.EPOCHS):
    """
    Train the model.
    
    Args:
        model: Compiled TensorFlow model
        training_sequences: Training input sequences
        training_labels: Training labels
        test_sequences: Test input sequences
        test_labels: Test labels
        epochs: Number of training epochs
    
    Returns:
        Training history object
    """
    print(f"\n=== Training Model ===")
    print(f"Training on {len(training_sequences)} samples, validating on {len(test_sequences)} samples")
    
    history = model.fit(
        training_sequences, 
        np.array(training_labels), 
        epochs=epochs, 
        validation_data=(test_sequences, np.array(test_labels)), 
        verbose=2
    )
    
    print("Training completed")
    return history


def evaluate_model(model, training_sequences, training_labels, 
                   test_sequences, test_labels):
    """
    Evaluate the model and print metrics.
    
    Args:
        model: Trained TensorFlow model
        training_sequences: Training input sequences
        training_labels: Training labels
        test_sequences: Test input sequences
        test_labels: Test labels
    """
    print("\n=== Evaluating Model ===")
    
    # Get predictions
    train_pred = (model.predict(training_sequences, verbose=0) >= 0.5).astype(int)
    test_pred = (model.predict(test_sequences, verbose=0) >= 0.5).astype(int)
    
    # Calculate metrics
    train_accuracy = accuracy_score(training_labels, train_pred)
    test_accuracy = accuracy_score(test_labels, test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_pred))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred))
    
    return train_accuracy, test_accuracy


def save_model(model, tokenizer, model_file=config.MODEL_FILE, 
               tokenizer_file=config.TOKENIZER_FILE):
    """
    Save the trained model and tokenizer.
    
    Args:
        model: Trained TensorFlow model
        tokenizer: Fitted tokenizer
        model_file: Path to save model
        tokenizer_file: Path to save tokenizer
    """
    print("\n=== Saving Model ===")
    
    # Save model
    model.save(model_file)
    print(f"Model saved to {model_file}")
    
    # Save tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_file}")

