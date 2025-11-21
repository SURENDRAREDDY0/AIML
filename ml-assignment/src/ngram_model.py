# import random

# class TrigramModel:
#     def __init__(self):
#         """
#         Initializes the TrigramModel.
#         """
#         # TODO: Initialize any data structures you need to store the n-gram counts.
       
#         pass

#     def fit(self, text):
#         """
#         Trains the trigram model on the given text.

#         Args:
#             text (str): The text to train the model on.
#         """
#         # TODO: Implement the training logic.
#         # This will involve:
#         # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
#         # 2. Tokenizing the text into words.
#         # 3. Padding the text with start and end tokens.
#         # 4. Counting the trigrams.
#         pass

#     def generate(self, max_length=50):
#         """
#         Generates new text using the trained trigram model.

#         Args:
#             max_length (int): The maximum length of the generated text.

#         Returns:
#             str: The generated text.
#         """
#         # TODO: Implement the generation logic.
#         # This will involve:
#         # 1. Starting with the start tokens.
#         # 2. Probabilistically choosing the next word based on the current context.
#         # 3. Repeating until the end token is generated or the maximum length is reached.
#         pass
import re
import random
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """

        # Nested dictionary for trigram counts:
        # counts[w1][w2][w3] = frequency
        self.counts = defaultdict(lambda: defaultdict(Counter))

        # Count for contexts (w1, w2) → {w3: freq}
        self.context_counts = defaultdict(Counter)

        # Special tokens
        self.start = "<s>"
        self.end = "</s>"
        self.unk = "<UNK>"

        # Vocabulary
        self.vocab = set()

    # -----------------------------------------------------
    # CLEANING & TOKENIZATION
    # -----------------------------------------------------
    def _clean_text(self, text):
        """
        Lowercase, remove non-alphanumeric, split into sentences.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\!\?]", " ", text)
        sentences = re.split(r"[.!?]+", text)

        cleaned = []
        for s in sentences:
            tokens = s.strip().split()
            if tokens:
                cleaned.append(tokens)

        return cleaned

    def _build_vocab(self, sentences, min_freq=2):
        """
        Build vocabulary and replace rare words with <UNK>.
        """
        freq = Counter()

        for s in sentences:
            for w in s:
                freq[w] += 1

        self.vocab = {w for w, c in freq.items() if c >= min_freq}
        self.vocab.add(self.unk)

    def _replace_unk(self, sentences):
        """
        Replace words not in vocab with <UNK>.
        """
        processed = []
        for s in sentences:
            new_s = [w if w in self.vocab else self.unk for w in s]
            processed.append(new_s)
        return processed

    # -----------------------------------------------------
    # FIT MODEL
    # -----------------------------------------------------
    def fit(self, text):
        """
        Trains the trigram model on the given text.
        """
        # 1. Clean & tokenize
        sentences = self._clean_text(text)

        # 2. Build vocab & replace unknowns
        self._build_vocab(sentences)
        sentences = self._replace_unk(sentences)

        # 3. Pad sentences and count trigrams
        for s in sentences:
            s = [self.start, self.start] + s + [self.end]

            for i in range(len(s) - 2):
                w1, w2, w3 = s[i], s[i+1], s[i+2]

                self.counts[w1][w2][w3] += 1
                self.context_counts[(w1, w2)][w3] += 1

    # -----------------------------------------------------
    # SAMPLING NEXT WORD
    # -----------------------------------------------------
    def _sample_next(self, w1, w2):
        """
        Sample next word based on trigram probabilities.
        """
        counter = self.context_counts.get((w1, w2))

        # Fallback if no context found
        if not counter:
            return self.unk

        total = sum(counter.values())
        rnd = random.random()
        cumulative = 0.0

        for word, count in counter.items():
            cumulative += count / total
            if rnd < cumulative:
                return word

        # Should not reach here, but safe fallback
        return random.choice(list(counter.keys()))

    # -----------------------------------------------------
    # GENERATE TEXT
    # -----------------------------------------------------
    def generate(self, max_length=50):
        """
        Generates text using the trained trigram model.
        """
        w1, w2 = self.start, self.start
        output_words = []

        for _ in range(max_length):
            next_word = self._sample_next(w1, w2)

            if next_word == self.end:
                break

            output_words.append(next_word)

            w1, w2 = w2, next_word

        return " ".join(output_words)
