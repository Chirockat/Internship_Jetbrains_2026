import numpy as np


class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.05, num_negative_samples=5):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples

        np.random.seed(42)

        self.W1 = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embedding_dim))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.embedding_dim, self.vocab_size))

    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def train_step(self, center_word_id, context_word_id):
        h = self.W1[center_word_id]
        loss = 0.0

        dW1_center = np.zeros(self.embedding_dim)

        dW2_updates = []

        target_ids = [context_word_id]
        labels = [1]

        while len(target_ids) < self.num_negative_samples + 1:
            neg_id = np.random.randint(0, self.vocab_size)
            if neg_id != context_word_id:
                target_ids.append(neg_id)
                labels.append(0)

        for target_id, label in zip(target_ids, labels):
            w2_vector = self.W2[:, target_id]
            z = np.dot(h, w2_vector)
            p = self.sigmoid(z)

            e = p - label

            if label == 1:
                loss -= np.log(p + 1e-9)
            else:
                loss -= np.log(1 - p + 1e-9)

            dW2_column = e * h
            dW1_center += e * w2_vector

            dW2_updates.append((target_id, dW2_column))

        for target_id, dW2_column in dW2_updates:
            self.W2[:, target_id] -= self.learning_rate * dW2_column

        self.W1[center_word_id] -= self.learning_rate * dW1_center

        return loss