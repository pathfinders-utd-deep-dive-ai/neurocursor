"""Read-only checks of the revision's records, sources, and archived evidence.

Run from the repository root with --data pointing to the private processed
dataset. This checks saved evidence; it does not rerun neural training or
independently authenticate participants or consent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from model import model as p, reporting

ROOT = Path(__file__).resolve().parents[2]
DATA_HASH = 'ef5df46c0f3c02bf458084e435079c4911026d81cea36f391823ec63f0570a95'
DISCLOSURE_HASH = '4dbe7fd254a231104f1c53659fa05b6bd0be1ad4c16a9e16df0d16a2b4c7b935'


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, context):
    check(np.isclose(actual, expected), f'{context}: {actual} != {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    path = Path(args.data)
    check(hashlib.sha256(path.read_bytes()).hexdigest() == DATA_HASH, 'Dataset hash')
    dataset = p.load_dataset(str(path))
    check(len(dataset.sequences) == 1330, 'Movement count')
    check(len(np.unique(dataset.user_ids)) == 10, 'Identity count')
    summary = json.loads((ROOT/'results/reviewer-strengthened/summary.json').read_text())
    configurations = verifiers = metric_checks = 0
    for mode, count in [('enrollment', 45), ('temporal', 21), ('feature', 6)]:
        rows = json.loads((ROOT/f'results/reviewer-strengthened/{mode}/results.json').read_text())
        check(len(rows) == count, f'{mode} grid size')
        check(len({(r['seed'], r['condition']) for r in rows}) == count, 'Duplicate conditions')
        for row in rows:
            check(row['dataset_sha256'] == DATA_HASH, 'Record dataset hash')
            ix = np.flatnonzero(np.isin(dataset.user_ids, row['users']))
            ids, sessions = dataset.user_ids[ix], dataset.session_ids[ix]
            parts = [set(zip(ids[row[k]], sessions[row[k]]))
                     for k in ('train_indexes', 'validation_indexes', 'test_indexes')]
            check(not any(parts[i] & parts[j] for i in range(3) for j in range(i+1, 3)),
                  'Session overlap')
            train = np.asarray(row['train_indexes'])
            counts = {str(u): len(np.unique(sessions[train][ids[train] == u]))
                      for u in row['users']}
            check(counts == row['training_session_counts'], 'Actual training counts')
            for user, value in row['configuration']['users'].items():
                verifiers += 1
                for k, m in value['interaction_count_metrics'].items():
                    if m.get('status') == 'unavailable':
                        continue
                    for q in (m, m['security_metrics']):
                        tn, fp, fn, tp = (q[x] for x in ('tn', 'fp', 'fn', 'tp'))
                        expected = dict(far=fp/(fp+tn), frr=fn/(fn+tp),
                            accuracy=(tn+tp)/(tn+fp+fn+tp), recall=tp/(tp+fn),
                            precision=tp/(tp+fp) if tp+fp else 0.,
                            f1_score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
                        for name, v in expected.items():
                            close(q[name], v, f'{mode}/{user}/{k}/{name}')
                            metric_checks += 1
                    close(m['threshold'], m['validation_calibration']['threshold'], 'Balanced threshold')
                    close(m['security_metrics']['threshold'], m['security_calibration']['threshold'], 'Strict threshold')
                    close(m['security_calibration']['far'], 0., 'Strict validation FAR')
                    fpr = np.asarray(m['roc_curve']['false_positive_rate'])
                    tpr = np.asarray(m['roc_curve']['true_positive_rate'])
                    j = np.argmin(abs(fpr - (1-tpr)))
                    close(m['eer'], (fpr[j]+1-tpr[j])/2, 'Discrete ROC EER')
                    close(m['roc_auc'], np.trapz(tpr, fpr), 'ROC AUC')
                history = value.get('training_history')
                check(not history or len(history['loss']) <= 50, 'Epoch limit')
            for op in ('balanced', 'strict'):
                actual = reporting.aggregate_k(row['configuration'], 5,
                    'balanced' if op == 'balanced' else 'security')
                for name, value in actual.items():
                    close(value, row[op][name], f'{mode}/{op}/{name} aggregate')
            configurations += 1
        for seed in (0, 1, 2):
            group = [r for r in rows if r['seed'] == seed]
            for key in ('validation_indexes', 'test_indexes'):
                check(len({tuple(r[key]) for r in group}) == 1, 'Unchanged holdouts')
            if mode == 'enrollment':
                for model in ('logistic-regression', 'cnn-gru', 'gru-only'):
                    nested = [next(r for r in group if r['condition'] ==
                              f'{model}-full-ordered-{budget}')
                              for budget in (1, 5, 10, 20, 'all')]
                    for a, b in zip(nested, nested[1:]):
                        check(set(a['train_indexes']) <= set(b['train_indexes']), 'Nested training budgets')
                    for budget, r in zip((1, 5, 10, 20), nested):
                        check(set(r['training_session_counts'].values()) == {budget}, 'Exact budget')
        for condition in {r['condition'] for r in rows}:
            group = [r for r in rows if r['condition'] == condition]
            check(sorted(r['seed'] for r in group) == [0, 1, 2], 'Complete split grid')
            for op in ('balanced', 'strict'):
                for name in ('far', 'frr', 'eer', 'roc_auc'):
                    values = [r[op][name] for r in group]
                    q = summary[mode][condition][op][name]
                    for field, v in [('mean', np.mean(values)), ('minimum', min(values)), ('maximum', max(values))]:
                        close(q[field], v, f'{mode}/{condition}/{field}')
    bundle = ROOT/'results/paper-real-data-2026-07-29'
    archived = 0
    for line in (bundle/'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        check(hashlib.sha256((bundle/name.lstrip('*')).read_bytes()).hexdigest() == digest,
              f'Archived evidence hash: {name}')
        archived += 1
    text = (ROOT/'model/Paper/paper.tex').read_text()
    start = text.index('\\section*{AI-Assisted Development')
    disclosure = text[start:text.index('\\FloatBarrier', start)]
    check(hashlib.sha256(disclosure.encode()).hexdigest() == DISCLOSURE_HASH, 'Protected disclosure')
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)
    check(len(labels) == len(set(labels)), 'Duplicate labels')
    check(not set(refs)-set(labels), 'Unresolved cross-references')
    bibs = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', text))
    cites = {x.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', text)
             for x in group.split(',')}
    check(not cites-bibs, 'Unresolved citations')
    print(f'PASS: {configurations} configurations, {verifiers} verifiers, '
          f'{metric_checks} confusion-derived metric checks, {archived} archived hashes.')
    print(f'PASS: ROC/AUC/EER, calibration provenance, split isolation, nested budgets, '
          f'{len(labels)} labels, {len(cites)} cited keys, and protected disclosure.')


if __name__ == '__main__':
    main()
