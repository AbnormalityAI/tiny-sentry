# Evaluation report

Measured on the supplied code and synthetic corpus, 12 October 2026. This is a reproducibility report, not a benchmark claim. The holdout has 12 effective template examples, expanded into 48 correlated rows.

| Model | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.750 | 0.800 | 0.667 | 0.727 | 16 | 20 | 4 | 8 |
| finetuned | 0.750 | 0.714 | 0.833 | 0.769 | 20 | 16 | 8 | 4 |
| scratch | 0.708 | 0.708 | 0.708 | 0.708 | 17 | 17 | 7 | 7 |

64 pretrain rows, 128 fine-tune rows, 48 held-out rows. Test template IDs and texts are disjoint from training splits. Repeated names within templates mean rows are correlated. Dataset hashes and every misclassification are in `artifacts/report.json`.

All runs use seed 41, 90 epochs, learning rate 0.4, hashed unigram/bigram features, L2 normalization, 0.0001 weight decay on active features and a fixed threshold of 0.5. The fine-tune checkpoint starts from base weights; scratch starts from zero. This report describes the supplied hyperparameters; the test set is not read by training. No validation set is supplied, so no threshold tuning is claimed.

Adaptation trades precision for recall. The unchanged accuracy and high false-positive rate are deliberate disclosures. Next research step: evaluate independent human-written examples rather than optimize this small test set.
