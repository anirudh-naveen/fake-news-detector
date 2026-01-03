"""
Fake News Detection - Main Training Script
Orchestrates the entire training pipeline using modular components
"""

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
    
    print("="*60)
    print("Fake News Detection Model - Training Pipeline")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    data = prep.load_and_clean_data()
    data, label_encoder = prep.encode_labels(data)
    
    # Step 2: Prepare data for training
    print("\n[Step 2] Preparing data for training...")
    titles, texts, labels = prep.prepare_data(data, config.TRAINING_SIZE)
    
    # Step 3: Tokenize and pad sequences
    print("\n[Step 3] Tokenizing and padding sequences...")
    tokenizer, padded_sequences, vocab_size = prep.tokenize_and_pad(titles)
    
    # Step 4: Split data
    print("\n[Step 4] Splitting data into train/test sets...")
    training_sequences, test_sequences, training_labels, test_labels = prep.split_data(
        padded_sequences, labels, config.TEST_PORTION, config.TRAINING_SIZE
    )
    
    # Step 5: Load GloVe embeddings
    print("\n[Step 5] Loading GloVe word embeddings...")
    embedding_index = embeddings.load_glove_embeddings()
    
    # Step 6: Create embedding matrix
    print("\n[Step 6] Creating embedding matrix...")
    embedding_matrix = embeddings.create_embedding_matrix(
        tokenizer.word_index, embedding_index, vocab_size, config.EMBEDDING_DIM
    )
    
    # Step 7: Build model
    print("\n[Step 7] Building model architecture...")
    model = model_builder.build_model(vocab_size, embedding_matrix)
    
    # Step 8: Train model
    print("\n[Step 8] Training model...")
    history = trainer.train_model(
        model, training_sequences, training_labels, 
        test_sequences, test_labels, config.EPOCHS
    )
    
    # Step 9: Evaluate model
    print("\n[Step 9] Evaluating model performance...")
    trainer.evaluate_model(
        model, training_sequences, training_labels, 
        test_sequences, test_labels
    )
    
    # Step 10: Save model and tokenizer
    print("\n[Step 10] Saving model and tokenizer...")
    trainer.save_model(model, tokenizer)
    
    # Step 11: Visualize training history
    print("\n[Step 11] Generating training visualization...")
    visualizer.plot_training_history(history)
    
    # Step 12: Test with sample
    print("\n[Step 12] Testing with sample text...")
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
    print("✓ Training pipeline completed successfully!")
    print("="*60)
    print(f"\nSaved files:")
    print(f"  - Model: {config.MODEL_FILE}")
    print(f"  - Tokenizer: {config.TOKENIZER_FILE}")
    print(f"  - Training plot: {config.HISTORY_PLOT_FILE}")


if __name__ == "__main__":
    main()
