"""Validate and summarize the fixed strengthened-reviewer experiment grid."""
from __future__ import annotations
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from model import model as p
from model import reporting

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    cli=parser.parse_args()
    source=p.load_dataset(cli.data)
    assert hashlib.sha256(Path(cli.data).read_bytes()).hexdigest()==(
        'ef5df46c0f3c02bf458084e435079c4911026d81cea36f391823ec63f0570a95')
    root=ROOT/'results/reviewer-strengthened'
    summary={}
    for mode,expected in [('enrollment',45),('temporal',21),('feature',6)]:
        paths=sorted((root/mode).glob('seed-*/*/result.json'))
        for parallel in sorted(root.parent.glob('reviewer-strengthened-parallel-*')):
            paths.extend(sorted((parallel/mode).glob('seed-*/*/result.json')))
        unique={}
        for path in paths:
            row=json.loads(path.read_text())
            key=(row['seed'],row['condition'])
            if key in unique:
                for op in ('balanced','strict'):
                    assert all(np.isclose(row[op][m],unique[key][op][m])
                               for m in ('far','frr','eer','roc_auc'))
            unique[key]=row
        rows=[unique[k] for k in sorted(unique)]
        assert len(rows)==expected,(mode,len(rows),expected)
        grouped=defaultdict(list)
        for row in rows:
            assert row['dataset_sha256']=='ef5df46c0f3c02bf458084e435079c4911026d81cea36f391823ec63f0570a95'
            row['configuration']['runtime']['seed']=row['seed']
            source_ix=np.flatnonzero(np.isin(source.user_ids,row['users']))
            users=source.user_ids[source_ix]
            sessions=source.session_ids[source_ix]
            partitions=[set(zip(users[row[k]],sessions[row[k]]))
                        for k in ('train_indexes','validation_indexes','test_indexes')]
            assert not any(partitions[i]&partitions[j]
                           for i in range(3) for j in range(i+1,3))
            selected=np.asarray(row['train_indexes'])
            actual_counts={str(u):len(np.unique(sessions[selected][users[selected]==u]))
                           for u in row['users']}
            assert actual_counts==row['training_session_counts']
            if row['distribution_only']:
                row['configuration']['implementation_notes']['classical_features']=(
                    'Mean, standard deviation, minimum and maximum per channel only.')
            for name,op in [('balanced','balanced'),('strict','security')]:
                actual=reporting.aggregate_k(row['configuration'],5,op)
                assert all(np.isclose(actual[k],row[name][k]) for k in actual)
            grouped[row['condition']].append(row)
        for seed in (0,1,2):
            split=[r for r in rows if r['seed']==seed]
            assert len({tuple(r['validation_indexes']) for r in split})==1
            assert len({tuple(r['test_indexes']) for r in split})==1
            for r in split:
                a,b,c=map(set,[r['train_indexes'],r['validation_indexes'],r['test_indexes']])
                assert not(a&b or a&c or b&c)
        result={}
        for condition,group in grouped.items():
            assert sorted(r['seed'] for r in group)==[0,1,2]
            result[condition]={}
            for op in ('balanced','strict'):
                result[condition][op]={m:{
                    'mean':float(np.mean([r[op][m] for r in group])),
                    'minimum':float(min(r[op][m] for r in group)),
                    'maximum':float(max(r[op][m] for r in group)),
                    'by_seed':{str(r['seed']):r[op][m] for r in group}}
                    for m in ('far','frr','eer','roc_auc')}
        if mode=='enrollment':
            stable={}
            for model in ('logistic-regression','cnn-gru','gru-only'):
                budgets=(1,5,10,20,None)
                groups=[[r for r in rows if r['condition']==
                    f'{model}-full-ordered-{b or "all"}'] for b in budgets]
                by_seed=[{r['seed']:r for r in group} for group in groups]
                for i,group in enumerate(groups):
                    for r in group:
                        n=list(r['training_session_counts'].values())
                        if budgets[i]:assert all(v==budgets[i] for v in n)
                    if i:
                        for seed in (0,1,2):assert set(by_seed[i-1][seed]['train_indexes'])<=set(by_seed[i][seed]['train_indexes'])
                passing=[]
                for i in range(len(budgets)-1):
                    if not all(r['strict']['far']<=.01+1e-12 and r['strict']['frr']<=.10+1e-12
                               for r in groups[i]):continue
                    if all(abs(by_seed[j][s]['strict']['far']-by_seed[i][s]['strict']['far'])<=.01+1e-12
                           and abs(by_seed[j][s]['strict']['frr']-by_seed[i][s]['strict']['frr'])<=.05+1e-12
                           for j in range(i+1,len(budgets)) for s in (0,1,2)):
                        passing.append(budgets[i])
                stable[model]=passing[0] if passing else None
            result['minimum_meeting_protocol']=stable
        p.save_json(root/mode/'results.json',rows)
        summary[mode]=result
    p.save_json(root/'summary.json',summary)
    for mode,groups in summary.items():
        print('\n',mode)
        for name,item in groups.items():
            if 'strict' not in item:print(name,item);continue
            print(name, 'strict FAR/FRR, EER:',
                  *[f'{100*item[op][m]["mean"]:.2f}' for op,m in
                    [('strict','far'),('strict','frr'),('balanced','eer')]])


if __name__=='__main__':main()
