"""
Fake News Detection - Main Training Script
Orchestrates the entire training pipeline
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import modules
import config
import data_preprocessing as prep
import embeddings
import model_builder
import trainer
import visualizer


def main():
    """Main function to run the complete training pipeline."""
    
    # Step 1: Load and preprocess data
    data = prep.load_and_clean_data()
    data, label_encoder = prep.encode_labels(data)
    
    # Step 2: Prepare data for training
    titles, texts, labels = prep.prepare_data(data, config.TRAINING_SIZE)
    
    # Step 3: Tokenize and pad sequences
    tokenizer, padded_sequences, vocab_size = prep.tokenize_and_pad(titles)
    
    # Step 4: Split data
    training_sequences, test_sequences, training_labels, test_labels = prep.split_data(
        padded_sequences, labels, config.TEST_PORTION, config.TRAINING_SIZE
    )
    
    # Step 5: Load GloVe embeddings
    embedding_index = embeddings.load_glove_embeddings()
    
    # Step 6: Create embedding matrix
    embedding_matrix = embeddings.create_embedding_matrix(
        tokenizer.word_index, embedding_index, vocab_size, config.EMBEDDING_DIM
    )
    
    # Step 7: Build model
    model = model_builder.build_model(vocab_size, embedding_matrix)
    
    # Step 8: Train model
    history = trainer.train_model(
        model, training_sequences, training_labels, 
        test_sequences, test_labels, config.EPOCHS
    )
    
    # Step 9: Evaluate model
    trainer.evaluate_model(
        model, training_sequences, training_labels, 
        test_sequences, test_labels
    )
    
    # Step 10: Save model and tokenizer
    trainer.save_model(model, tokenizer)
    
    # Step 11: Visualize training history
    visualizer.plot_training_history(history)
    
    # Step 12: Test with sample
    print("\n=== Testing with Sample Text ===")
    test_text = "Karry to go to France in gesture of sympathy"
    print(f"Testing with: '{test_text}'")
    
    sequences = tokenizer.texts_to_sequences([test_text])
    sequences = pad_sequences(
        sequences, 
        maxlen=config.MAX_LENGTH, 
        padding=config.PADDING_TYPE, 
        truncating=config.TRUNC_TYPE
    )
    prediction = model.predict(sequences, verbose=0)[0][0]
    
    print(f"Model confidence: {prediction:.4f}")
    if prediction >= 0.5:
        print("Result: This news is REAL")
    else:
        print("Result: This news is FAKE")
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
