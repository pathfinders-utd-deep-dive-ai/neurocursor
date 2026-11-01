"""Additional reviewer experiments using fixed session holdouts.

Run from repository root: python -m model.Paper.reviewer_experiments --data PATH
Results are exploratory; no test-driven enrollment stopping rule is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from model import model as pipeline
from model import reporting


def permute_time(data, seed):
    rng = np.random.default_rng(seed)
    result = np.stack([x[rng.permutation(len(x))] for x in data])
    assert np.array_equal(np.sort(result, axis=1), np.sort(data, axis=1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='results/reviewer-experiments')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--neural', action='store_true')
    cli = parser.parse_args()
    destination = Path(cli.output)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = pipeline.load_dataset(cli.data)
    digest = hashlib.sha256(Path(cli.data).read_bytes()).hexdigest()
    pipeline.FEATURE_SETS['low-order'] = tuple(
        f for f in pipeline.FULL_FEATURE_NAMES
        if not any(term in f for term in
                   ('acceleration', 'jerk', 'angular_velocity', 'curvature'))
    )
    records = []
    for seed in cli.seeds:
        train, validation, test, split_name = pipeline.create_split(
            dataset, seed=seed, mode='session'
        )
        partition_sessions = [set(zip(dataset.user_ids[ix], dataset.session_ids[ix]))
                              for ix in (train, validation, test)]
        assert not (partition_sessions[0] & partition_sessions[1])
        assert not (partition_sessions[0] & partition_sessions[2])
        assert not (partition_sessions[1] & partition_sessions[2])
        # One nested session ordering per identity; held-out samples never change.
        rng = np.random.default_rng(seed + 1000)
        ordered = {}
        for user in np.unique(dataset.user_ids):
            sessions = np.unique(dataset.session_ids[train][dataset.user_ids[train] == user])
            ordered[str(user)] = rng.permutation(sessions)
        conditions = [('logistic-regression', 'full', 'ordered', b)
                      for b in (1, 3, 5, 10, 20, None)]
        conditions += [('logistic-regression', 'low-order', 'ordered', None)]
        if cli.neural:
            conditions += [(m, f, order, None)
                for m in ('cnn-gru', 'gru-only')
                for f, order in [('full', 'ordered'), ('full', 'permuted'),
                                 ('low-order', 'ordered')]]
        for model_type, feature_set, order, budget in conditions:
            selected = np.array([i for i in train if dataset.session_ids[i] in
                ordered[str(dataset.user_ids[i])][:budget]], dtype=int)
            counts = {u: min(len(s), budget) if budget else len(s)
                      for u, s in ordered.items()}
            # Capped budgets retain all identities and explicitly record support.
            assert set(selected).isdisjoint(validation)
            assert set(selected).isdisjoint(test)
            condition = f'{model_type}-{feature_set}-{order}-budget-{budget or "all"}'
            folder = destination / f'seed-{seed}' / condition
            cached = folder / 'result.json'
            if cached.exists():
                record = json.loads(cached.read_text())
                assert record['dataset_sha256'] == digest
                records.append(record)
                continue
            original_prepare = pipeline.prepare_sequences
            if order == 'permuted':
                # Jointly permute channel vectors: preserve each channel's
                # marginals and contemporaneous cross-channel relationships.
                def permuted_prepare(*args, **kwargs):
                    data = original_prepare(*args, **kwargs)
                    return permute_time(data, seed + 2000)
                pipeline.prepare_sequences = permuted_prepare
            args = SimpleNamespace(sequence_length=128, learning_rate=1e-3,
                epochs=50, batch_size=32, patience=8, verbose=0,
                seed=seed, target_far=0., permutations=0, bootstrap_samples=0,
                validation_size=.15, test_size=.1)
            pipeline.tf.keras.utils.set_random_seed(seed)
            try:
                configuration = pipeline.run_configuration(dataset,
                    (selected, validation, test, split_name), folder, args,
                    model_type, feature_set, persist_models=False,
                    include_statistics=False)
            finally:
                pipeline.prepare_sequences = original_prepare
            record = dict(seed=seed, condition=condition,
                dataset_sha256=digest, training_session_counts=counts,
                exact_budget_users=[u for u, n in counts.items() if n == budget],
                train_indexes=selected.tolist(), validation_indexes=validation.tolist(),
                test_indexes=test.tolist(), configuration=configuration,
                balanced=reporting.aggregate_k(configuration, 5),
                strict=reporting.aggregate_k(configuration, 5, 'security'))
            pipeline.save_json(cached, record)
            records.append(record)
            pipeline.save_json(destination / 'results.json', records)
    pipeline.save_json(destination / 'results.json', records)


if __name__ == '__main__':
    main()
