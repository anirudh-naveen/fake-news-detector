"""
Visualization Module
Creates plots for training history
"""

import matplotlib.pyplot as plt
import config


def plot_training_history(history, save_path=config.HISTORY_PLOT_FILE):
    """
    Plot and save training history (accuracy and loss).
    
    Args:
        history: Training history object from model.fit()
        save_path: Path to save the plot
    """
    print("\n=== Generating Training Visualization ===")
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

