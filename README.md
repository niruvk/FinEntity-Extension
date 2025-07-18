# 🧠 FinEntity-Extension: Benchmarking Entity-Level Sentiment for Financial Texts

**An Extension of FinEntity: Enhanced Entity-Level Sentiment Classification and Commodity Case Study**  
By Niranjan Vijaya Krishnan, Lily Weaver, Chaeyoung Lee

---

## 🔍 Overview

This project builds upon [FinEntity (Tang et al., 2023)](https://arxiv.org/abs/2302.13971), a framework for entity-level sentiment classification in financial news. We extend this work in two major directions:

1. **Model Benchmarking**: Evaluate recent language models (DeBERTa, RoBERTa, GPT-4o, LLaMA, Qwen) for financial NER + sentiment classification on the FinEntity dataset—testing with and without a CRF (Conditional Random Field) layer.
2. **Commodity Case Study**: Replicate the original cryptocurrency use case and expand it to **commodities** (Oil, Gold, Copper, Silver), analyzing correlations between entity-level sentiment and price trends.

---

## 📊 Key Results

| Model               | Positive F1 | Negative F1 | Micro Avg F1 | Notes                           |
|--------------------|-------------|-------------|---------------|----------------------------------|
| **DeBERTa-CRF**     | 0.94        | 0.88        | **0.89**      | Best overall performance         |
| FinBERT-CRF         | 0.84        | 0.88        | 0.84          | Best baseline from original work|
| GPT-4o (fine-tuned) | 0.81        | 0.79        | 0.85          | Strong LLM result (fine-tuned)  |
| GPT-3.5 (zero-shot) | 0.39        | 0.58        | 0.59          | Poor zero-shot performance      |
| LLaMA / Qwen        | 0.76–0.79   | 0.66–0.70   | 0.70–0.73     | Struggles with fine-tuning      |

🟢 CRF layers consistently improve performance for open-source models.  
📈 In commodity analysis, entity-level sentiment generally shows stronger or equal correlation with prices (MIC) compared to sequence-level sentiment.

---

## 📁 Repository Structure

```
FinEntity-Extension/
│
├── models/
│   ├── finbert_crf.py
│   ├── deberta_crf.py
│   └── ...
│
├── training/
│   ├── train_crf.py
│   ├── train_standard.py
│   └── utils.py
│
├── commodity_case_study/
│   ├── sentiment_extraction.py
│   ├── correlation_analysis.py
│   ├── price_data.py
│   └── visualizations.ipynb
│
├── data/
│   ├── finentity_dataset/
│   └── commodity_news/
│
├── results/
│   ├── model_scores.csv
│   ├── mic_scores.csv
│   └── plots/
│
├── requirements.txt
├── README.md
└── FinEntity_Paper.pdf
```

---

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries include:
- `transformers`
- `datasets`
- `sklearn`
- `seqeval`
- `yfinance`
- `minepy`

---

### 2. Train Models

#### FinBERT-CRF

```bash
python training/train_crf.py --model finbert --use_crf
```

#### DeBERTa-CRF

```bash
python training/train_crf.py --model deberta --use_crf
```

#### GPT-4o or GPT-3.5 (zero/few-shot)

Check `closed_model_prompting/` for inference scripts.

---

### 3. Run Commodity Analysis

```bash
python commodity_case_study/sentiment_extraction.py
python commodity_case_study/correlation_analysis.py
```

Outputs normalized sentiment and MIC scores for oil, gold, copper, and silver.

---

## 📈 Sample Visualization

> ![Entity vs. Price](results/plots/oil_entity_vs_price.png)  
_Entity-level sentiment (orange) vs. Oil Price (blue)_

---

## 📚 Citation

If you use this repo or build on it, please cite:

```bibtex
@article{krishnan2024finentityextension,
  title={An Extension of FinEntity: Entity-level Sentiment Classification for Financial Texts},
  author={Krishnan, Niranjan Vijaya and Weaver, Lily and Lee, Chaeyoung},
  journal={GitHub},
  year={2024},
  url={https://github.com/niruvk/FinEntity-Extension}
}
```

---

## 🔮 Future Work

- Benchmark Claude, Gemini, and other LLMs on FinEntity.
- Pre-train **FinDeBERTa** and **FinRoBERTa** on large financial corpora.
- Expand sentiment analysis to stocks, bonds, real estate, and derivatives.
- Apply to real-world trading tasks (P/L analysis, risk modeling, portfolio simulation).

---

## 📌 Acknowledgments

This project builds on Tang et al.’s [FinEntity (2023)](https://arxiv.org/abs/2302.13971).  
Thanks to OpenAI, Meta, Microsoft, and Alibaba for providing LLMs and pre-trained models.
