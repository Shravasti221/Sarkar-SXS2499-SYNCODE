SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

# Entropy weighting (tune as needed)
WEIGHT_DISSIM = 1.0    # how much pairwise dissimilarity contributes
WEIGHT_LENGTH = 0.6    # length reward
WEIGHT_EXPERT = 1.0    # expert diversity reward
LOOP_PENALTY_SCALE = 1.5  # penalty multiplier for loops
MAX_LEN_NORM = 12      # length normalization constant (log scale)