# cuad-audit

Finding legal clauses in contracts (CUAD v1, 12 clause types) and auditing the expert labels.

```
pip install -r requirements.txt
python -m src.data          # download CUAD, make the contract split, count clauses per category
python -m src.baseline      # TF-IDF + logistic regression (CPU, ~1 min)
python -m pytest -q
```

RoBERTa-base (GPU):

```
python -m src.roberta train --model roberta-base --run roberta-base
python -m src.roberta predict --run roberta-base --split test
python -m src.roberta download-cuad
python -m src.roberta predict --run cuad-roberta-base --split test
python -m src.metrics test tfidf cuad-roberta-base roberta-base
```

- `src/config.py`: paths and the 12 categories
- `src/data.py`: download, load and split the data
- `src/baseline.py`: sentence classifier baseline
- `src/roberta.py`: windowing, fine-tuning and prediction for RoBERTa
- `src/metrics.py`: CUAD's AUPR and model comparison
