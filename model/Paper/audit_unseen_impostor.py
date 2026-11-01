"""Read-only verification of the archived identity-exclusion control."""
import argparse
import json
from pathlib import Path
import numpy as np
from model import model as p

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    d = p.load_dataset(args.data)
    root = Path(__file__).resolve().parents[2] / 'results/unseen-impostor-control'
    rows = json.loads((root / 'results.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    assert len(rows) == 270
    assert len({(r['seed'], r['claimed_user'], r['held_identity']) for r in rows}) == 270
    for r in rows:
        held = r['held_identity']
        assert held != r['claimed_user']
        assert r['seen']['test_indexes'] == r['excluded']['test_indexes']
        assert r['seen']['test_attempt_labels'] == r['excluded']['test_attempt_labels']
        for part in ('train_indexes', 'validation_indexes'):
            expected = [i for i in r['seen'][part] if d.user_ids[i] != held]
            assert expected == r['excluded'][part]
        for condition in ('seen', 'excluded'):
            a = r[condition]
            sets = [set(zip(d.user_ids[a[k]], d.session_ids[a[k]]))
                    for k in ('train_indexes', 'validation_indexes', 'test_indexes')]
            assert not any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
            for op in ('balanced', 'strict'):
                actual = p.biometric_metrics(np.array(a['test_attempt_labels']),
                    np.array(a['test_attempt_scores']), a['calibration'][op]['threshold'])
                for metric in ('far', 'frr', 'eer', 'roc_auc', 'tn', 'fp', 'fn', 'tp'):
                    assert np.isclose(actual[metric], a['metrics'][op][metric])
            assert a['calibration']['strict']['far'] == 0
    for c in ('seen', 'excluded'):
        for op in ('balanced', 'strict'):
            for m in ('far', 'frr', 'eer', 'roc_auc'):
                values = [np.mean([r[c]['metrics'][op][m] for r in rows if r['seed'] == s])
                          for s in (0, 1, 2)]
                for key, v in [('mean', np.mean(values)), ('minimum', min(values)), ('maximum', max(values))]:
                    assert np.isclose(summary[c][op][m][key], v)
    print('PASS: 270 matched pairs; exclusion, isolation, 8,640 metric checks and summaries.')

if __name__ == '__main__':
    main()
