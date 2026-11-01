"""Fixed logistic seen/excluded-impostor control; see the archived protocol."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
from model import model as p


def scale_fit(x, indexes):
    mean=x[indexes].mean(axis=0,keepdims=True)
    std=x[indexes].std(axis=0,keepdims=True)
    std[std<1e-7]=1.
    return (x-mean)/std


def attempts(d, indexes, scores, claimed):
    return p.aggregate_attempt_scores(scores,(d.user_ids[indexes]==claimed).astype(int),
        d.user_ids[indexes],d.session_ids[indexes],d.attempt_ids[indexes],
        d.movement_indexes[indexes],d.durations[indexes],5)


def evaluate(d, claimed, held, train, val, test, x, seed):
    check_sets=[set(zip(d.user_ids[ix],d.session_ids[ix])) for ix in (train,val,test)]
    assert not any(check_sets[i]&check_sets[j] for i in range(3) for j in range(i+1,3))
    z=scale_fit(x,train)
    model=p.build_classical_model('logistic-regression',seed)
    model.fit(z[train],(d.user_ids[train]==claimed).astype(int))
    v=attempts(d,val,model.predict_proba(z[val])[:,1],claimed)
    pair=test[np.isin(d.user_ids[test],[claimed,held])]
    q=attempts(d,pair,model.predict_proba(z[pair])[:,1],claimed)
    assert v is not None and q is not None
    calibrations={'balanced':p.calibrate_eer_threshold(v[1],v[0]),
                  'strict':p.calibrate_far_threshold(v[1],v[0],0.)}
    return dict(calibration=calibrations,
        metrics={k:p.biometric_metrics(q[1],q[0],c['threshold']) for k,c in calibrations.items()},
        test_attempt_scores=q[0],test_attempt_labels=q[1],
        train_indexes=train,validation_indexes=val,test_indexes=pair)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',default='results/unseen-impostor-control')
    args=ap.parse_args();path=Path(args.data);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='ef5df46c0f3c02bf458084e435079c4911026d81cea36f391823ec63f0570a95'
    d=p.load_dataset(str(path));x=p.prepare_sequences(d.sequences,128)
    x=p.summarize_sequences(x,p.FULL_FEATURE_NAMES);assert x.shape==(1330,121)
    users=np.unique(d.user_ids);rows=[];dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    for seed in (0,1,2):
        tr,va,te,_=p.create_split(d,seed=seed,mode='session')
        for claimed in users:
            # Seen scores are reused across h; this is one model fit, not nine.
            z=scale_fit(x,tr);model=p.build_classical_model('logistic-regression',seed)
            model.fit(z[tr],(d.user_ids[tr]==claimed).astype(int))
            vs=attempts(d,va,model.predict_proba(z[va])[:,1],claimed)
            ts=model.predict_proba(z[te])[:,1]
            seen_cal={'balanced':p.calibrate_eer_threshold(vs[1],vs[0]),
                      'strict':p.calibrate_far_threshold(vs[1],vs[0],0.)}
            for held in users:
                if held==claimed:continue
                mask=np.isin(d.user_ids[te],[claimed,held]);pair=te[mask]
                q=attempts(d,pair,ts[mask],claimed);assert q is not None
                seen=dict(calibration=seen_cal,
                    metrics={k:p.biometric_metrics(q[1],q[0],c['threshold']) for k,c in seen_cal.items()},
                    test_attempt_scores=q[0],test_attempt_labels=q[1],
                    train_indexes=tr,validation_indexes=va,test_indexes=pair)
                a=tr[d.user_ids[tr]!=held];b=va[d.user_ids[va]!=held]
                assert held not in d.user_ids[a] and held not in d.user_ids[b]
                unseen=evaluate(d,claimed,held,a,b,te,x,seed)
                assert np.array_equal(seen['test_indexes'],unseen['test_indexes'])
                rows.append(dict(seed=seed,claimed_user=str(claimed),held_identity=str(held),
                    dataset_sha256=digest,seen=seen,excluded=unseen))
            print(f'Completed seed={seed} claimed={claimed}',flush=True)
    assert len(rows)==270
    summary={}
    for condition in ('seen','excluded'):
        summary[condition]={}
        for op in ('balanced','strict'):
            summary[condition][op]={}
            for metric in ('far','frr','eer','roc_auc'):
                by_seed={str(seed):float(np.mean([r[condition]['metrics'][op][metric]
                    for r in rows if r['seed']==seed])) for seed in (0,1,2)}
                summary[condition][op][metric]=dict(mean=float(np.mean(list(by_seed.values()))),
                    minimum=min(by_seed.values()),maximum=max(by_seed.values()),by_seed=by_seed)
    p.save_json(dest/'results.json',rows);p.save_json(dest/'summary.json',summary)
    for condition in summary:
        print(condition,{m:round(100*summary[condition]['strict'][m]['mean'],4)
                         for m in ('far','frr','eer','roc_auc')},flush=True)


if __name__=='__main__':main()
