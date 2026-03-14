import numpy as np

class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        self.W1 = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embedding_dim))
        self.W2 = np.random.uniform(-0.1, 0.1, (self.embedding_dim, self.vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, center_word_id):
        self.h = self.W1[center_word_id]
        self.u = np.dot(self.h, self.W2)
        self.y_pred = self.softmax(self.u)
        return self.y_pred

    def backward(self, center_word_id, context_word_id):
        y_true = np.zeros(self.vocab_size)
        y_true[context_word_id] = 1.0

        e = self.y_pred - y_true

        dW2 = np.outer(self.h, e)
        dh = np.dot(self.W2, e)

        self.W2 -= self.learning_rate * dW2
        self.W1[center_word_id] -= self.learning_rate * dh

        loss = -np.log(self.y_pred[context_word_id] + 1e-9)
        return loss