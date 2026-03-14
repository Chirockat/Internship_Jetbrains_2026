import numpy as np
import re
from collections import Counter

class DataLoader:
    def __init__(self, text, window_size=2, min_count=3):
        self.window_size = window_size
        self.min_count = min_count

        raw_tokens = self._clean_and_tokenize(text)

        self.word_to_id = {}
        self.id_to_word = {}

        self.tokens = self._filter_and_build_vocab(raw_tokens)
        self.vocab_size = len(self.word_to_id)

        print(f"Vocabulary built. Unique words: {self.vocab_size}")
        print(f"Total words in text after filtering: {len(self.tokens)}")

    def _clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _filter_and_build_vocab(self, tokens):
        word_counts = Counter(tokens)

        valid_words = {word for word, count in word_counts.items() if count >= self.min_count}

        for i, word in enumerate(valid_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        filtered_tokens = [word for word in tokens if word in valid_words]

        return filtered_tokens

    def get_training_pairs(self):
        pairs = []

        token_ids = [self.word_to_id[word] for word in self.tokens]

        for i, center_word_id in enumerate(token_ids):
            start = max(0, i - self.window_size)
            end = min(len(token_ids), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word_id = token_ids[j]
                    pairs.append((center_word_id, context_word_id))

        return np.array(pairs)