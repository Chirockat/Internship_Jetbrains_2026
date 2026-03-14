import numpy as np


class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.01):
        """
        Inicjalizacja modelu Word2Vec (Skip-gram).

        :param vocab_size: Rozmiar słownika (ile unikalnych słów mamy).
        :param embedding_dim: Liczba wymiarów dla wektora słowa (zwykle od 50 do 300).
        :param learning_rate: Szybkość uczenia (jak duże kroki robimy przy poprawianiu błędów).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # INICJALIZACJA WAG (Początkowo są to małe losowe liczby)
        # W1 to nasza docelowa macierz wektorów - z niej po treningu wyciągniemy wektory słów!
        # Rozmiar: (Ilość słów x Wymiar wektora)
        self.W1 = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embedding_dim))

        # W2 to macierz pomocnicza (wektory kontekstowe) używana tylko podczas treningu.
        # Rozmiar: (Wymiar wektora x Ilość słów)
        self.W2 = np.random.uniform(-0.1, 0.1, (self.embedding_dim, self.vocab_size))

    def softmax(self, x):
        """Zamienia surowe wyniki (logits) na prawdopodobieństwa sumujące się do 1."""
        # Odejmowanie np.max(x) to sztuczka zapewniająca stabilność numeryczną (żeby nie było błędu overflow)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, center_word_id):
        """
        Przejście w przód (Forward Pass).
        Na podstawie słowa środkowego próbujemy zgadnąć słowo z kontekstu.
        """
        # 1. Pobieramy wektor ukryty dla słowa głównego (reprezentacja słowa)
        self.h = self.W1[center_word_id]

        # 2. Obliczamy "surowe" wyniki (logits) dla WSZYSTKICH słów w słowniku
        self.u = np.dot(self.h, self.W2)

        # 3. Przepuszczamy przez funkcję Softmax, aby uzyskać prawdopodobieństwa
        self.y_pred = self.softmax(self.u)

        return self.y_pred

    def backward(self, center_word_id, context_word_id):
        """
        Przejście wstecz (Backward Pass / Backpropagation).
        Obliczamy błąd i aktualizujemy wagi macierzy.
        """
        # 1. Tworzymy wektor docelowy dla prawdziwego słowa z kontekstu
        # Wektor składa się z samych zer, tylko na indeksie poprawnego słowa ma 1.
        y_true = np.zeros(self.vocab_size)
        y_true[context_word_id] = 1.0

        # 2. Obliczamy błąd przewidywania
        # To jest pochodna funkcji straty (Cross-Entropy) połączonej z Softmaxem.
        e = self.y_pred - y_true

        # 3. Obliczamy gradienty (jak bardzo musimy zmienić każdą wagę)
        # Gradient dla W2: mnożenie zewnętrzne wektora ukrytego 'h' i błędu 'e'
        dW2 = np.outer(self.h, e)

        # Gradient dla h (czyli wiersza z W1): iloczyn macierzy W2 i wektora błędu 'e'
        dh = np.dot(self.W2, e)

        # 4. AKTUALIZACJA WAG (Spadek Gradientu / Gradient Descent)
        self.W2 -= self.learning_rate * dW2
        self.W1[center_word_id] -= self.learning_rate * dh

        # 5. Obliczamy i zwracamy wartość błędu (Loss) dla celów monitorowania
        loss = -np.log(self.y_pred[context_word_id] + 1e-9)  # 1e-9 zabezpiecza przed log(0)
        return loss