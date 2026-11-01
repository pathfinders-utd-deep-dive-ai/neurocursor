"""Replicated reviewer controls; protocol fixed before these new fits.

These reuse development data and are not an independent confirmatory cohort.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from model import model as p
from model import reporting
from model.Paper.reviewer_experiments import permute_time

SEEDS = (0, 1, 2)


def conditions(mode):
    if mode == 'enrollment':
        return [(m, 'full', 'ordered', b, False)
                for m in ('logistic-regression', 'cnn-gru', 'gru-only')
                for b in (1, 5, 10, 20, None)]
    if mode == 'temporal':
        return [(m, 'position-only', o, None, True)
                for m in ('cnn-gru', 'gru-only')
                for o in ('ordered', 'permuted', 'reversed')] + [
                    ('logistic-regression', 'position-only', 'ordered', None, True)]
    return [('logistic-regression', f, 'ordered', None, True)
            for f in ('full', 'low-order')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['enrollment','temporal','feature'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output', default='results/reviewer-strengthened')
    args = parser.parse_args()
    p.tf.config.threading.set_inter_op_parallelism_threads(1)
    p.tf.config.threading.set_intra_op_parallelism_threads(1)
    p.tf.config.experimental.enable_op_determinism()
    destination = Path(args.output) / args.mode
    destination.mkdir(parents=True, exist_ok=True)
    dataset = p.load_dataset(args.data)
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    p.FEATURE_SETS['position-only'] = ('x_normalized', 'y_normalized')
    p.FEATURE_SETS['low-order'] = tuple(f for f in p.FULL_FEATURE_NAMES
        if not any(t in f for t in ('acceleration','jerk','angular_velocity','curvature')))
    # Eligibility depends only on session counts, before model fitting.
    # Twenty training sessions plus rounding at 15%/10% requires >=27 total.
    eligible = [u for u in np.unique(dataset.user_ids)
        if len(np.unique(dataset.session_ids[dataset.user_ids == u])) >= 27]
    if args.mode == 'enrollment':
        ix = np.flatnonzero(np.isin(dataset.user_ids, eligible))
        dataset = dataclasses.replace(dataset,
            sequences=[dataset.sequences[i] for i in ix],
            **{name:getattr(dataset,name)[ix] for name in
               ('user_ids','session_ids','attempt_ids','movement_indexes','durations')})
    records = []
    for seed in args.seeds:
        train, val, test, split_name = p.create_split(dataset, seed=seed, mode='session')
        rng = np.random.default_rng(seed + 1000)
        ordered = {str(u):rng.permutation(np.unique(dataset.session_ids[train][
            dataset.user_ids[train] == u])) for u in np.unique(dataset.user_ids)}
        if args.mode == 'enrollment':
            assert all(len(s) >= 20 for s in ordered.values())
        partitions=[set(zip(dataset.user_ids[a],dataset.session_ids[a]))
                    for a in (train,val,test)]
        assert not any(partitions[i]&partitions[j] for i in range(3) for j in range(i+1,3))
        for model, features, order, budget, distribution_only in conditions(args.mode):
            selected=np.array([i for i in train if dataset.session_ids[i] in
                ordered[str(dataset.user_ids[i])][:budget]],dtype=int)
            counts={u:len(s[:budget]) for u,s in ordered.items()}
            if budget is not None: assert all(n==budget for n in counts.values())
            condition=f'{model}-{features}-{order}-{budget or "all"}'
            folder=destination/f'seed-{seed}'/condition
            cached=folder/'result.json'
            if cached.exists():
                record=json.loads(cached.read_text())
                assert record['dataset_sha256']==digest
                records.append(record)
                continue
            transform=None
            if order=='permuted': transform=lambda x:permute_time(x,seed+2000)
            if order=='reversed': transform=lambda x:x[:,::-1,:].copy()
            settings=SimpleNamespace(sequence_length=128,learning_rate=.001,
                epochs=50,batch_size=32,patience=8,verbose=0,seed=seed,target_far=0.,
                permutations=0,bootstrap_samples=0,validation_size=.15,test_size=.1)
            p.tf.keras.utils.set_random_seed(seed)
            print(f'START {args.mode} seed={seed} {condition}',flush=True)
            configuration=p.run_configuration(dataset,(selected,val,test,split_name),
                folder,settings,model,features,False,False,
                sequence_transform=transform,
                classical_distribution_only=distribution_only)
            configuration['runtime']['seed'] = seed
            record=dict(seed=seed,condition=condition,mode=args.mode,
                dataset_sha256=digest,users=sorted(ordered),
                training_session_counts=counts,features=list(p.FEATURE_SETS[features]),
                distribution_only=distribution_only,
                train_indexes=selected.tolist(),validation_indexes=val.tolist(),
                test_indexes=test.tolist(),configuration=configuration,
                balanced=reporting.aggregate_k(configuration,5),
                strict=reporting.aggregate_k(configuration,5,'security'))
            p.save_json(cached,record)
            records.append(record)
            p.save_json(destination/'results.json',records)
            print(f'DONE {condition}',flush=True)
    p.save_json(destination/'results.json',records)


if __name__=='__main__': main()
