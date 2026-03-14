import numpy as np
import re
from collections import Counter


class DataLoader:
    def __init__(self, text, window_size=2, min_count=3):
        """
        Initializes the data loader.

        :param text: Raw text string (e.g., loaded from a file).
        :param window_size: Size of the context window.
        :param min_count: Minimum frequency for a word to be included in the vocabulary.
        """
        self.window_size = window_size
        self.min_count = min_count

        # 1. Prepare raw tokens
        raw_tokens = self._clean_and_tokenize(text)

        # 2. Dictionaries mapping words to IDs and vice versa
        self.word_to_id = {}
        self.id_to_word = {}

        # 3. Filter rare words and build the vocabulary
        self.tokens = self._filter_and_build_vocab(raw_tokens)
        self.vocab_size = len(self.word_to_id)

        print(f"Vocabulary built! Unique words: {self.vocab_size}")
        print(f"Total words in text after filtering: {len(self.tokens)}")

    def _clean_and_tokenize(self, text):
        """Converts to lowercase, removes special characters, and splits into words."""
        text = text.lower()
        # Keep only basic latin letters, numbers, and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _filter_and_build_vocab(self, tokens):
        """Counts word frequencies, removes rare words, and creates ID mappings."""
        # Count occurrences of each word
        word_counts = Counter(tokens)

        # Keep only words that appear at least 'min_count' times
        valid_words = {word for word, count in word_counts.items() if count >= self.min_count}

        # Assign unique IDs to valid words
        for i, word in enumerate(valid_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        # Remove rare words from the original text flow completely.
        # This brings valid words closer together in the context window.
        filtered_tokens = [word for word in tokens if word in valid_words]

        return filtered_tokens

    def get_training_pairs(self):
        """
        Generates training pairs for the Skip-gram model.
        Returns a NumPy array of (center_word_id, context_word_id) pairs.
        """
        pairs = []

        # Convert the filtered text into a list of numerical IDs
        token_ids = [self.word_to_id[word] for word in self.tokens]

        for i, center_word_id in enumerate(token_ids):
            # Define window boundaries to avoid index out of bounds
            start = max(0, i - self.window_size)
            end = min(len(token_ids), i + self.window_size + 1)

            # Collect context words
            for j in range(start, end):
                if i != j:  # Do not pair the word with itself
                    context_word_id = token_ids[j]
                    pairs.append((center_word_id, context_word_id))

        # Convert to NumPy array for the model
        return np.array(pairs)