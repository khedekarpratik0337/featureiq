# Pretrained Meta-Learner Models

This directory holds pretrained meta-learner model files (`.joblib`).

To train a model, run:

```bash
python scripts/train_meta_learner.py --training-data path/to/data.json
```

The trained model will be saved as `meta_learner_v1.joblib` in this directory and automatically discovered by FeatureIQ at runtime.
