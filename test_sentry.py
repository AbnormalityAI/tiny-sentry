import json
import tempfile
import unittest
from pathlib import Path
from sentry import features, predict, train, metrics, read_rows

class ModelTests(unittest.TestCase):
    def test_features_stable(self):
        self.assertEqual(features('IGNORE Previous'), features('ignore previous'))
    def test_feature_norm(self):
        self.assertAlmostEqual(sum(v*v for v in features('a b a').values()),1)
    def test_empty_text(self):
        self.assertEqual(features(''), {})
    def test_training_is_deterministic(self):
        rows=[{'text':'friendly meeting', 'label':0},{'text':'reveal credentials','label':1}]
        self.assertEqual(train(rows, epochs=8),train(rows, epochs=8))
    def test_initial_not_mutated(self):
        initial=train([{'text':'meeting','label':0}],epochs=1)
        old=json.dumps(initial)
        train([{'text':'reveal','label':1}],initial,epochs=3)
        self.assertEqual(json.dumps(initial),old)
    def test_serialization(self):
        m=train([{'text':'public note','label':0},{'text':'leak keys','label':1}])
        restored=json.loads(json.dumps(m))
        self.assertEqual(predict(m,'leak'),predict(restored,'leak'))
    def test_fits_tiny_problem(self):
        rows=[{'text':'good','label':0},{'text':'bad','label':1}]
        self.assertEqual(metrics(train(rows),rows)['accuracy'],1)
    def test_labels_validated_before_conversion(self):
        for label in ['1', 'oops', None, True, 2]:
            with self.assertRaisesRegex(ValueError, 'integer 0 or 1'):
                train([{'text':'x','label':label}])
    def test_split_disjoint(self):
        root=Path(__file__).parent/'data'
        rows={n:read_rows(root/f'{n}.jsonl') for n in ('pretrain','finetune','test')}
        for a,b in [('pretrain','finetune'),('pretrain','test'),('finetune','test')]:
            self.assertFalse({r['template_id'] for r in rows[a]} & {r['template_id'] for r in rows[b]})
            self.assertFalse({r['text'] for r in rows[a]} & {r['text'] for r in rows[b]})

if __name__ == '__main__': unittest.main()
