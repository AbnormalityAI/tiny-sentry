"""Deterministic hashed n-gram logistic classifier, pure Python, CPU only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

DIM = 1024
SEED = 41

def features(text: str, dim: int = DIM) -> dict[int, float]:
    tokens = re.findall(r"[a-z]+", text.lower())
    grams = tokens + [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    values: dict[int, float] = {}
    for gram in grams:
        digest = hashlib.sha256(gram.encode()).digest()
        index = int.from_bytes(digest[:4], 'big') % dim
        values[index] = values.get(index, 0) + 1
    norm = math.sqrt(sum(v*v for v in values.values())) or 1
    return {k: v/norm for k, v in values.items()}

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1/(1+math.exp(-z))
    exp = math.exp(z)
    return exp/(1+exp)

def predict(model: dict, text: str) -> float:
    return sigmoid(model['bias'] + sum(model['weights'][k]*v for k, v in features(text, model['dim']).items()))

def train(rows: list[dict], initial: dict | None = None, *, epochs: int = 90,
          rate: float = 0.4, seed: int = SEED) -> dict:
    if not rows:
        raise ValueError('training rows must not be empty')
    weights = list(initial['weights']) if initial else [0.0]*DIM
    dim = initial['dim'] if initial else DIM
    bias = initial['bias'] if initial else 0.0
    if any(type(r.get('label')) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('labels must be integer 0 or 1')
    vectors = [(features(r['text'], dim), r['label']) for r in rows]
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(vectors)))
        rng.shuffle(order)
        for i in order:
            vec, label = vectors[i]
            error = sigmoid(bias + sum(weights[k]*v for k, v in vec.items())) - label
            for k, v in vec.items():
                weights[k] -= rate*(error*v + 0.0001*weights[k])
            bias -= rate*error
    return {'format': 1, 'dim': dim, 'weights': weights, 'bias': bias,
            'threshold': 0.5, 'seed': seed, 'epochs': epochs}

def metrics(model: dict, rows: list[dict]) -> dict:
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    mistakes = []
    for row in rows:
        p = predict(model, row['text'])
        pred = int(p >= model['threshold'])
        key = ('tp' if pred else 'fn') if row['label'] else ('fp' if pred else 'tn')
        counts[key] += 1
        if pred != row['label']:
            mistakes.append({'id': row['id'], 'label': row['label'], 'probability': round(p, 6), 'text': row['text']})
    tp, tn, fp, fn = (counts[k] for k in ('tp','tn','fp','fn'))
    return {**counts, 'n': len(rows), 'accuracy': (tp+tn)/max(1,len(rows)),
            'precision': tp/max(1,tp+fp), 'recall': tp/max(1,tp+fn),
            'f1': 2*tp/max(1,2*tp+fp+fn), 'mistakes': mistakes}

def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def run(root: Path) -> dict:
    pre = read_rows(root/'data/pretrain.jsonl')
    fit = read_rows(root/'data/finetune.jsonl')
    test = read_rows(root/'data/test.jsonl')
    groups = [set(r['text'] for r in rows) for rows in (pre,fit,test)]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('duplicate text across splits')
    before = train(pre)
    after = train(fit, before)
    scratch = train(fit)
    artifacts = root/'artifacts'
    artifacts.mkdir(exist_ok=True)
    (artifacts/'base.json').write_text(json.dumps(before, indent=2)+'\n')
    (artifacts/'model.json').write_text(json.dumps(after, indent=2)+'\n')
    report = {'seed': SEED, 'algorithm': 'hashed unigram/bigram logistic regression',
              'train_counts': {'pretrain':len(pre),'finetune':len(fit)},
              'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'data').glob('*.jsonl'))},
              'base': metrics(before,test), 'finetuned':metrics(after,test),
              'scratch':metrics(scratch,test),
              'caveat':'Small synthetic template-disjoint holdout. No production security claim. Fine-tuning may not beat scratch.'}
    (artifacts/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train')
    classify = sub.add_parser('classify')
    classify.add_argument('text')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == 'train':
        print(json.dumps(run(root), indent=2))
    else:
        model = json.loads((root/'artifacts/model.json').read_text())
        probability = predict(model, args.text)
        print(json.dumps({'label': 'attempt' if probability >= model['threshold'] else 'benign',
                          'score': probability, 'calibrated': False}))

if __name__ == '__main__': main()
