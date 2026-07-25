
from pathlib import Path
import re
from collections import Counter, defaultdict
from difflib import get_close_matches

DATA_FILE = Path(__file__).with_name("DATA.txt")


def load_text(path: Path = DATA_FILE) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_models(text: str):
    tokens = clean_text(text).split()
    if not tokens:
        return {
            "unigrams": Counter(),
            "bigrams": defaultdict(Counter),
            "trigrams": defaultdict(Counter),
            "total_unigrams": 0,
        }

    unigrams = Counter(tokens)
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    for i in range(len(tokens) - 1):
        bigrams[tokens[i]][tokens[i + 1]] += 1

    for i in range(len(tokens) - 2):
        trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

    return {
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "total_unigrams": sum(unigrams.values()),
    }


def autocorrect_word(word: str, vocabulary: Counter) -> str:
    word = word.lower().strip()
    if not word:
        return word
    if word in vocabulary:
        return word

    candidates = get_close_matches(word, list(vocabulary.keys()), n=5, cutoff=0.75)
    if candidates:
        return max(candidates, key=vocabulary.get)

    fallback = [term for term in vocabulary if term.startswith(word[:1])]
    if fallback:
        return max(fallback, key=vocabulary.get)

    return word


def predict_next_word(user_input: str, models, top_k: int = 5):
    tokens = clean_text(user_input).split()
    if not tokens:
        return []

    total_unigrams = models.get("total_unigrams", 0) or sum(models["unigrams"].values())
    last_word = tokens[-1]

    def scored_candidates(candidate_counts, context_word=None):
        if not candidate_counts:
            return []

        parent_count = sum(candidate_counts.values())
        scored = []
        for word, count in candidate_counts.items():
            p_ngram = count / parent_count
            p_unigram = models["unigrams"][word] / total_unigrams if total_unigrams else 0
            p_bigram = 0
            if context_word and context_word in models["bigrams"]:
                bigram_context = models["bigrams"][context_word]
                p_bigram = bigram_context[word] / sum(bigram_context.values()) if bigram_context else 0

            score = 0.7 * p_ngram + 0.2 * p_bigram + 0.1 * p_unigram
            scored.append((word, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    if len(tokens) >= 2:
        context = (tokens[-2], last_word)
        if context in models["trigrams"]:
            return scored_candidates(models["trigrams"][context], context_word=last_word)

    if last_word in models["bigrams"]:
        return scored_candidates(models["bigrams"][last_word], context_word=last_word)

    return [(word, count / total_unigrams) for word, count in models["unigrams"].most_common(top_k)]


def main():
    text = load_text()
    models = build_models(text)
    vocabulary = models["unigrams"]

    print("Loaded text from DATA.txt")
    user_input = input("Enter a word or phrase: ").strip()

    corrected_input = " ".join(
        autocorrect_word(part, vocabulary) for part in user_input.split()
    )
    predictions = predict_next_word(corrected_input, models)

    print(f"\nYou entered: {user_input}")
    print(f"Suggested correction: {corrected_input}")
    print("Likely next words:")
    if predictions:
        for word, _ in predictions:
            print(f"- {word}")
    else:
        print("No prediction available.")


if __name__ == "__main__":
    main()

