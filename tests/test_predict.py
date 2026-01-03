"""
Unit tests for prediction functionality
Run with: pytest tests/
"""

import pytest
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config
from predict import predict_news


class TestPredictions:
    """Test prediction functionality"""
    
    def test_predict_news_format(self):
        """Test that predict_news returns correct format"""
        # Create a dummy tokenizer and model would be needed for full test
        # This is a template - you'd need to mock the model
        pass
    
    def test_padding_length(self):
        """Test that sequences are padded to correct length"""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(["test text"])
        
        sequences = tokenizer.texts_to_sequences(["short text"])
        padded = pad_sequences(
            sequences,
            maxlen=config.MAX_LENGTH,
            padding=config.PADDING_TYPE,
            truncating=config.TRUNC_TYPE
        )
        
        assert padded.shape[1] == config.MAX_LENGTH
    
    def test_tokenizer_vocab(self):
        """Test tokenizer creates vocabulary"""
        tokenizer = Tokenizer()
        texts = ["hello world", "test text"]
        tokenizer.fit_on_texts(texts)
        
        assert len(tokenizer.word_index) > 0
        assert "hello" in tokenizer.word_index
        assert "world" in tokenizer.word_index


if __name__ == "__main__":
    pytest.main([__file__])

