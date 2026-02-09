# 📩 Spam SMS Classification using LSTM (RNN)

End-to-end NLP project that classifies SMS messages as **SPAM** or **HAM** using a single-layer LSTM network, with a production-ready Streamlit inference app.

---

## Project Overview

| Aspect | Detail |
|---|---|
| **Problem** | Binary text classification (Spam vs Ham) |
| **Architecture** | Many-to-One LSTM sequence classifier |
| **Dataset** | [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (5 574 messages) |
| **Framework** | TensorFlow / Keras |
| **Deployment** | Streamlit web application |

---

## Repository Structure

```
spam-sms-classification-lstm/
├── app/
│   ├── app.py               # Streamlit inference application
│   └── requirements.txt      # Python dependencies
├── data/
│   └── archive/
│       └── spam.csv          # Raw UCI SMS dataset
├── model/
│   ├── spam_lstm_model.h5    # Trained LSTM model
│   └── tokenizer.pkl         # Fitted Keras tokenizer
├── notebook/
│   ├── spam_sms_lstm.ipynb   # Training notebook (Colab)
│   └── spam_sms_lstm.py      # Exported Python script
└── README.md
```

---

## Model Details

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 8 000 |
| Max sequence length | 40 tokens |
| Embedding dimension | 64 |
| LSTM units | 64 |
| Dropout | 0.2 |
| Output activation | Sigmoid |
| Optimizer | Adam |
| Loss | Binary cross-entropy |

### Class Imbalance Handling

The dataset is heavily skewed (~87 % ham, ~13 % spam). Balanced class weights were computed via `sklearn.utils.class_weight.compute_class_weight` and passed to `model.fit()`, ensuring the model does not trivially predict the majority class.

### Evaluation

The model is evaluated on a stratified 20 % test split using **Precision, Recall, and F1-score** — not accuracy — because accuracy is misleading on imbalanced data.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/dinraj910/Spam-SMS-Classification-using-LSTM-RNN-.git
cd Spam-SMS-Classification-using-LSTM-RNN-
```

### 2. Install dependencies

```bash
pip install -r app/requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

---

## How the Deployment Works

```
User SMS input
      │
      ▼
┌─────────────────┐
│  Tokenizer.pkl  │  ← Same tokenizer used during training
│  (word-level)   │
└────────┬────────┘
         │  texts_to_sequences → pad_sequences (maxlen=40)
         ▼
┌─────────────────┐
│  LSTM Model     │  ← Loaded from spam_lstm_model.h5
│  (Many-to-One)  │
└────────┬────────┘
         │  sigmoid output ≥ 0.5 → SPAM, else HAM
         ▼
   SPAM 🚨 / HAM ✅
```

---

## Real-World NLP Skills Demonstrated

| Skill | Evidence |
|---|---|
| **Text preprocessing** | Word-level tokenization, OOV handling, sequence padding |
| **Sequence modeling** | LSTM with masking and dropout for short, noisy text |
| **Class imbalance** | Balanced class weights — not oversampling or undersampling |
| **Proper evaluation** | Precision / Recall / F1 instead of misleading accuracy |
| **Model persistence** | Saved model (`.h5`) and tokenizer (`.pkl`) for reuse |
| **Inference pipeline** | Identical preprocessing at training and serving time |
| **Deployment** | Clean Streamlit app with input validation and caching |

---

## License

This project is for educational and portfolio purposes.
