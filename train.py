import numpy as np

from data_loader import DataLoader
from model import Word2VecModel

# To show progress:
from tqdm import tqdm


def get_similar_words(target_word, word_vectors, word_to_id, id_to_word, top_n=3):
    """
    Znajduje słowa najbardziej podobne do podanego słowa za pomocą podobieństwa kosinusowego.
    """
    # 1. Sprawdzamy, czy słowo w ogóle istnieje w naszym słowniku
    if target_word not in word_to_id:
        return f"Słowo '{target_word}' nie występuje w słowniku."

    target_id = word_to_id[target_word]
    target_vector = word_vectors[target_id]

    # 2. Obliczamy iloczyn skalarny (dot product) między naszym słowem a wszystkimi innymi wektorami
    dot_products = np.dot(word_vectors, target_vector)

    # 3. Obliczamy długości (normy) wektorów
    # Używamy np.linalg.norm do obliczenia długości euklidesowej
    norms_all = np.linalg.norm(word_vectors, axis=1)
    norm_target = np.linalg.norm(target_vector)

    # Zabezpieczenie przed dzieleniem przez zero
    norms_all[norms_all == 0] = 1e-9
    norm_target = norm_target if norm_target != 0 else 1e-9

    # 4. Obliczamy podobieństwo kosinusowe dla wszystkich słów
    similarities = dot_products / (norms_all * norm_target)

    # 5. Sortujemy wyniki (np.argsort sortuje rosnąco, więc odwracamy tablicę [::-1])
    sorted_indices = np.argsort(similarities)[::-1]

    # 6. Zbieramy najlepsze wyniki, pomijając samo słowo docelowe (które zawsze ma podobieństwo 1.0)
    similar_words = []
    for idx in sorted_indices:
        if idx != target_id:
            word = id_to_word[idx]
            sim_score = similarities[idx]
            similar_words.append((word, sim_score))
            if len(similar_words) == top_n:
                break

    return similar_words

def train():

    full_text = open('data/corpus.txt', 'r', encoding='utf-8').read()

    max_words = 20000
    text_subset = " ".join(full_text.split()[:max_words])

    print(f"Selected a subset of {max_words} words for testing.")

    loader = DataLoader(text_subset, window_size=3, min_count=3)
    training_data = loader.get_training_pairs()

    # 2. INICJALIZACJA MODELU
    # Ustawiamy mały wymiar wektora (np. 10), żeby na tym małym tekście policzyło się to w sekundę.
    print("\nInicjalizacja modelu Word2Vec...")
    model = Word2VecModel(vocab_size=loader.vocab_size, embedding_dim=50, learning_rate=0.05)

    # 3. PĘTLA UCZĄCA (TRAINING LOOP)
    epochs = 5  # Ile razy model ma przejść przez cały nasz tekst

    print("\nRozpoczynamy trening...")
    for epoch in range(epochs):
        total_loss = 0

        # Używamy enumerate, żeby mieć indeks 'i', oraz ustawiamy mininterval
        # mininterval=0.5 sprawia, że pasek odświeża się maksymalnie 2 razy na sekundę
        pbar = tqdm(enumerate(training_data), total=len(training_data),
                    desc=f"Epoka {epoch + 1}/{epochs}", mininterval=0.5)

        for i, (center_word_id, context_word_id) in pbar:

            # Przejście w przód i w tył
            model.forward(center_word_id)
            loss = model.backward(center_word_id, context_word_id)
            total_loss += loss

            # Aktualizujemy napis 'loss' na pasku TYLKO co 1000 kroków
            # Dzięki temu nie zapychamy bufora konsoli setkami tysięcy wiadomości
            if i % 1000 == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

        # Wyświetlamy średni błąd co 20 epok, żeby widzieć, czy model się uczy
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoka {epoch + 1}/{epochs} | Średnia strata (Loss): {avg_loss:.4f}")

        # 4. WYCIĄGNIĘCIE WYNIKÓW I TEST PODOBIEŃSTWA
        print("\nTrening zakończony!")
        word_vectors = model.W1

        # Testujemy nasze nowe narzędzie!
        test_words = ["witch", "wolf", "king"]

        print("\n--- TEST PODOBIEŃSTWA SŁÓW ---")
        for word in test_words:
            # Pamiętaj, że nasz data_loader usunął polskie znaki, jeśli użyłeś czyszczenia regex!
            # Dlatego dla pewności szukamy po słowach bez znaków diakrytycznych, np. "krol", "mezczyzna"
            clean_word = loader._clean_and_tokenize(word)
            if clean_word:
                search_word = clean_word[0]
                results = get_similar_words(search_word, word_vectors, loader.word_to_id, loader.id_to_word, top_n=2)

                print(f"\nSłowa najbliższe dla '{search_word}':")
                if isinstance(results, str):
                    print(results)
                else:
                    for sim_word, score in results:
                        print(f" -> {sim_word} (podobieństwo: {score:.4f})")


if __name__ == "__main__":
    train()