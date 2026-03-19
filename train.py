import numpy as np
from data_loader import DataLoader
from model import Word2VecModel
from tqdm import tqdm
import json


# Hyperparameters
WINDOW_SIZE = 5
MIN_COUNT = 5
EMBEDDING_DIM = 100
LEARNING_RATE = 0.03
EPOCHS = 5

NUM_NEGATIVE_SAMPLES = 5
MAX_WORDS = 100000
TOP_N_SIMILAR = 2
CORPUS_PATH = 'data/corpus.txt'
TEST_WORDS = ["death", "wolf", "king"]



EVALUATION_CASES = [
    ("dog", ["king", "queen", "prince", "dog"]),
    ("castle", ["wolf", "fox", "bear", "castle"]),
    ("happy", ["wicked", "evil", "dark", "happy"]),
    ("axe", ["mother", "father", "brother", "axe"]),
    ("forest", ["house", "door", "cottage", "forest"]),
    ("sun", ["dark", "black", "night", "sun"]),
    ("water", ["gold", "silver", "money", "water"]),
    ("king", ["wolf", "dog", "cat", "king"]),
    ("bird", ["door", "house", "bed", "bird"]),
    ("stone", ["beautiful", "pretty", "young", "stone"])

]

def get_similar_words(target_word, word_vectors, word_to_id, id_to_word, top_n=3):
    if target_word not in word_to_id:
        return f"Word '{target_word}' not found in vocabulary."

    target_id = word_to_id[target_word]
    target_vector = word_vectors[target_id]

    dot_products = np.dot(word_vectors, target_vector)

    norms_all = np.linalg.norm(word_vectors, axis=1)
    norm_target = np.linalg.norm(target_vector)

    norms_all[norms_all == 0] = 1e-9
    norm_target = norm_target if norm_target != 0 else 1e-9

    similarities = dot_products / (norms_all * norm_target)
    sorted_indices = np.argsort(similarities)[::-1]

    similar_words = []
    for idx in sorted_indices:
        if idx != target_id:
            word = id_to_word[idx]
            sim_score = similarities[idx]
            similar_words.append((word, sim_score))
            if len(similar_words) == top_n:
                break

    return similar_words

# For testing the model
def evaluate_odd_one_out(word_vectors, word_to_id, test_cases):
    correct = 0
    total = 0

    for expected_outcast, words in test_cases:
        valid_words = [w for w in words if w in word_to_id]

        if len(valid_words) < 3 or expected_outcast not in valid_words:
            continue

        vectors = np.array([word_vectors[word_to_id[w]] for w in valid_words])
        centroid = np.mean(vectors, axis=0)

        norms_vectors = np.linalg.norm(vectors, axis=1)
        norm_centroid = np.linalg.norm(centroid)

        norms_vectors[norms_vectors == 0] = 1e-9
        norm_centroid = norm_centroid if norm_centroid != 0 else 1e-9

        similarities = np.dot(vectors, centroid) / (norms_vectors * norm_centroid)

        min_idx = np.argmin(similarities)
        predicted_outcast = valid_words[min_idx]

        if predicted_outcast == expected_outcast:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy, total

# for checking how many test cases appear
def diagnose_evaluation_cases(word_to_id, test_cases):
    print("\n--- DIAGNOSTIC: MISSING WORDS IN TEST CASES ---")
    all_good = True
    for expected_outcast, words in test_cases:
        missing = [w for w in words if w not in word_to_id]
        if missing:
            print(f"Case '{expected_outcast}' missing words: {missing}")
            all_good = False

    if all_good:
        print("All words from test cases are present in the vocabulary!")

def train():
    full_text = open(CORPUS_PATH, 'r', encoding='utf-8').read()
    text_subset = " ".join(full_text.split()[:MAX_WORDS])

    print(f"Selected a subset of {MAX_WORDS} words for testing.")

    loader = DataLoader(text_subset, window_size=WINDOW_SIZE, min_count=MIN_COUNT)
    training_data = loader.get_training_pairs()

    diagnose_evaluation_cases(loader.word_to_id, EVALUATION_CASES)

    print("\nInitializing Word2Vec model...")
    model = Word2VecModel(
        vocab_size=loader.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
        num_negative_samples=NUM_NEGATIVE_SAMPLES
    )


    print("\nStarting training...")
    for epoch in range(EPOCHS):
        total_loss = 0

        pbar = tqdm(enumerate(training_data), total=len(training_data),
                    desc=f"Epoch {epoch + 1}/{EPOCHS}", mininterval=0.5)

        for i, (center_word_id, context_word_id) in pbar:
            loss = model.train_step(center_word_id, context_word_id)
            total_loss += loss

            if i % 1000 == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    print("\nTraining finished!")
    word_vectors = model.W1

    np.save('word_vectors.npy', word_vectors)
    with open('word_to_id.json', 'w', encoding='utf-8') as f:
        json.dump(loader.word_to_id, f)
    print("\nModel weights and vocabulary saved to disk.")

    print("\n--- ODD-ONE-OUT EVALUATION ---")
    accuracy, valid_tests = evaluate_odd_one_out(word_vectors, loader.word_to_id, EVALUATION_CASES)
    print(f"Evaluated on {valid_tests} valid cases.")
    print(f"Model Accuracy: {accuracy:.2f}%")

    print("\n--- WORD SIMILARITY TEST ---")
    for word in TEST_WORDS:
        clean_word = loader._clean_and_tokenize(word)
        if clean_word:
            search_word = clean_word[0]
            results = get_similar_words(search_word, word_vectors, loader.word_to_id, loader.id_to_word, top_n=TOP_N_SIMILAR)

            print(f"\nClosest words to '{search_word}':")
            if isinstance(results, str):
                print(results)
            else:
                for sim_word, score in results:
                    print(f" -> {sim_word} (similarity: {score:.4f})")

if __name__ == "__main__":
    train()