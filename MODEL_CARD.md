# Model card: Tiny Sentry v0.1

## Identity

Hashed lexical logistic classifier, 1,024 weights and one bias. Original synthetic data, no third-party pretrained checkpoint. `artifacts/model.json` is the adapted checkpoint. MIT licensed.

## Intended use

Learning about reproducible adaptation, CPU training and honest evaluation. Not a protective control, content moderation product or authority detector. Never execute detected text.

## Training

64 social-engineering pretrain rows followed by 128 prompt-injection adaptation rows. Eight pretrain templates, sixteen adaptation templates and twelve test templates, split before expanding entity names. Binary labels: 1 means an attempt to redirect an assistant outside its task; 0 means ordinary task text or benign discussion. Seed 41; 90 SGD epochs per stage; learning rate 0.4. No model downloads, network calls or personal data.

## Evaluation

12 effective template examples expanded into 48 held-out synthetic rows. Accuracy 0.750, precision 0.714, recall 0.833, F1 0.769. Eight false positives and four false negatives. Full confusion matrices, errors and hashes are in the report. Base accuracy is also 0.750, so adaptation did not improve accuracy.

## Risks and limits

English lexical templates cannot capture intent or source authority. Quoted attacks may be flagged. Obfuscation, translation, long documents and new vocabulary may evade detection. The output score is uncalibrated. Correlated template variants and a small holdout make broad generalization claims invalid. There is no demographic bias assessment. Real deployments need independent data, privacy review, monitoring and safeguards that do not rely on this classifier.

## Reproduce

`python3 sentry.py train` writes base, adapted and evaluation artifacts. `python3 -m unittest -v` checks deterministic training, serialization and split separation.
