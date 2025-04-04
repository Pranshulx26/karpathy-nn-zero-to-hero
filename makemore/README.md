# 📜 Makemore: Bigram & Trigram Language Models

Exploring character-level language models—from count-based to neural networks.

## 📌 Overview

This project implements Bigram and Trigram character-level language models, inspired by Andrej Karpathy’s makemore series. Key experiments:

* Bigram Model: Predicts next character using frequency counts.
* Single-Layer NN: Replicates bigram behavior with a neural net.
* Trigram Model (MLP): Uses 2-character context for better predictions.

**Key Insight:**

* Achieved similar performance between count-based bigram and neural net.
* Trigram MLP reduced loss by 15% over bigram (see Results).

## 🚀 Models Implemented

1.  **Bigram Model (Count-Based)**

    * Approach: Uses character co-occurrence frequencies (P(next_char | current_char)).
    * Training: Maximum Likelihood Estimation (MLE) on `names.txt`.
    * Code: `bigram.ipynb`

2.  **Bigram Model (Single-Layer NN)**

    * Approach: Mimics count-based bigram using a 1-layer neural net (no hidden layers).
    * Key Trick: Logits = unnormalized counts, optimized via cross-entropy.
    * Result: Matches count-based bigram performance (loss ~2.45).

3.  **Trigram Model (MLP)**

    * Approach: Predicts next character using 2-character context (P(c3 | c1, c2)).
    * Architecture: 1-hidden-layer MLP (32 neurons, ReLU).
    * Code: `trigram-language-model.ipynb`

## 📊 Results

| Model            | Loss (NLL) | Sample Output    |
| :--------------- | :--------- | :--------------- |
| Bigram (Counts)  | 2.45       | Kavin, Miley     |
| Bigram (NN)      | 2.47       | Jaxon, Aria      |
| Trigram (MLP)    | 2.08       | Sophia, Elijah   |

Trigram MLP outperforms bigram by 15% (lower loss = better).

## 🔍 Key Learnings

✅ **Counts vs. NNs:** A 1-layer NN can replicate count-based bigrams exactly if weights mimic log-counts.

✅ **Context Matters:** Trigrams capture more meaningful patterns (e.g., "th" → "e").

✅ **Loss as a Debug Tool:** Used loss curves to verify implementations.

## 💻 How to Run

```bash
git clone [https://github.com/Pranshulx26/karpathy-nn-zero-to-hero](https://github.com/Pranshulx26/karpathy-nn-zero-to-hero)
cd karpathy-nn-zero-to-hero/makemore_part_2
jupyter notebook makemore-part1-bigram.ipynb
```

jupyter notebook trigram-language-model.ipynb

🛠️ Extensions & Next Steps
Add embeddings (Lecture 5) to further reduce loss.
Visualize character probabilities (heatmaps for P(c3 | c1, c2)).
Benchmark against Karpathy’s MLP implementation.
📜 License
MIT. See LICENSE.

🎯 Why This Matters
This project demystifies:

🔹 How NNs relate to traditional count-based models.
🔹 The power of context in language modeling.
🔹 Debugging models via loss analysis.
Ideal for NLP engineers and deep learning practitioners!
