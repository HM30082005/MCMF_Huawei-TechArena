"""Tree comparison in a process that does not import PyTorch."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/history_upgrade'
CACHE=ROOT/'analysis/.cache'


def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load_data(prefix):
    keys=['user','items','time','hist_len']+(['label'] if prefix=='train' else [])
    return {k:np.load(CACHE/f'{prefix}_{k}.npy',mmap_mode='r') for k in keys}


def first_events(data,cutoff,end=None):
    ix=np.flatnonzero((data['time']>=cutoff)&((data['time']<end) if end else True))
    _,first=np.unique(data['user'][ix],return_index=True)
    return np.sort(ix[first])


def auc(y,score):
    y=np.asarray(y).ravel().astype(bool)
    n=int(y.sum())
    return float((rankdata(np.asarray(score).ravel())[y].sum(dtype=np.float64)-n*(n+1)/2)/(n*(len(y)-n)))


def metrics(y,score):
    score=np.asarray(score,dtype=np.float64)
    np.testing.assert_allclose(auc(y,score),roc_auc_score(np.asarray(y).ravel(),score.ravel()),rtol=0,atol=1e-12)
    return dict(roc_auc=auc(y,score),log_loss=float(np.mean(np.logaddexp(0,score)-y*score)),
                rows=int(np.asarray(y).size),clicks=int(np.asarray(y).sum()))


def params(kind,size,threads):
    if kind=='CatBoost':
        return dict(depth=size,iterations=1200,learning_rate=.05,loss_function='Logloss',eval_metric='AUC',
                    random_seed=42,thread_count=threads,task_type='CPU',has_time=True,boosting_type='Plain',
                    l2_leaf_reg=20,random_strength=1,bootstrap_type='Bernoulli',subsample=.8,rsm=.85,
                    border_count=128,verbose=100,allow_writing_files=True,train_dir=str(OUT/'catboost_info'))
    return dict(objective='binary',metric='auc',learning_rate=.05,num_leaves=size,min_data_in_leaf=200,
                feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbosity=-1,
                seed=42,deterministic=True,force_col_wise=True,num_threads=threads)


def load_model(record):
    if record['kind']=='CatBoost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier()
        model.load_model(record['path'])
        return model
    import lightgbm as lgb
    return lgb.Booster(model_file=record['path'])


def predict(model,kind,X,threads):
    if kind=='CatBoost':
        return model.predict(X,prediction_type='RawFormulaVal',thread_count=threads)
    return model.predict(X,raw_score=True,num_threads=threads)


def fit(kind,size,X,y,names,stage,args,valid=None,trees=None):
    name=f'{stage}_{kind}_{size}'
    path=OUT/'trees'/(name+('.cbm' if kind=='CatBoost' else '.txt'))
    path.parent.mkdir(parents=True,exist_ok=True)
    meta_path=path.with_suffix('.json')
    config=params(kind,size,args.threads)
    if trees is not None and kind=='CatBoost':
        config['iterations']=trees
    identity={'recipe':json.loads((OUT/'recipe.json').read_text()),'kind':kind,'size':size,'stage':stage,
              'trees_requested':trees,'config':config,'features':names}
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['identity']==identity:
            log('Reusing tree model: '+name)
            return load_model(meta),meta
    begin=time.time()
    log(f'Fitting {name}: {len(y):,} rows, {len(names)} features')
    if kind=='CatBoost':
        from catboost import CatBoostClassifier,Pool
        pool=Pool(X,label=y,feature_names=names,thread_count=args.threads)
        pool.quantize(border_count=128)
        vp=Pool(valid[0],label=valid[1],feature_names=names,thread_count=args.threads) if valid else None
        model=CatBoostClassifier(**config)
        model.fit(pool,eval_set=vp,use_best_model=valid is not None,
                  early_stopping_rounds=80 if valid else None,save_snapshot=True,
                  snapshot_file=str(OUT/'trees'/(name+'.snapshot')),snapshot_interval=30)
        count=model.tree_count_
        temporary=path.with_name(path.name+'.tmp')
        model.save_model(temporary,format='cbm')
        temporary.replace(path)
    else:
        import lightgbm as lgb
        pool=lgb.Dataset(X,label=y,feature_name=names,free_raw_data=True)
        callbacks=[lgb.log_evaluation(100)]
        kwargs={}
        if valid:
            vp=lgb.Dataset(valid[0],label=valid[1],reference=pool,feature_name=names)
            kwargs['valid_sets']=[vp]
            callbacks.append(lgb.early_stopping(80,first_metric_only=True,verbose=True))
        model=lgb.train(config,pool,num_boost_round=trees or 1200,callbacks=callbacks,**kwargs)
        count=model.best_iteration if valid else model.current_iteration()
        temporary=path.with_name(path.name+'.tmp')
        model.save_model(str(temporary),num_iteration=count)
        temporary.replace(path)
    meta=dict(identity=identity,kind=kind,size=size,trees=int(count),path=str(path),seconds=time.time()-begin,
              library_version=__import__('catboost' if kind=='CatBoost' else 'lightgbm').__version__)
    if valid:
        score=predict(model,kind,valid[0],args.threads)
        meta['development']=metrics(valid[1],score)
    atomic_json(meta_path,meta)
    return model,meta


def feature_set(through_day,query_name):
    path=OUT/'features'/f'train_through{through_day}.npy'
    meta=json.loads(path.with_suffix('.json').read_text())
    query=OUT/'features'/(query_name+'.npy')
    qm=json.loads(query.with_suffix('.json').read_text())
    if meta['feature_names']!=qm['feature_names']:
        raise ValueError('Training/query feature names disagree')
    if meta['identity']['data_fingerprint']!=qm['identity']['data_fingerprint']:
        raise ValueError('Training/query source fingerprints disagree')
    train=np.load(path,mmap_mode='r')
    valid=np.load(query,mmap_mode='r')
    names=meta['feature_names']
    return train.reshape(-1,len(names)),valid.reshape(-1,len(names)),meta,names


def confidence_and_comparison(y,score,baseline):
    rng=np.random.default_rng(927)
    values,deltas=[],[]
    for _ in range(100):
        ix=rng.integers(0,len(y),len(y))
        current=auc(y[ix],score[ix])
        values.append(current)
        deltas.append(current-auc(y[ix],baseline[ix]))
    return {'method':'100 paired percentile-bootstrap resamples of whole first-event-per-user groups','seed':927,
            'auc_lower_95':float(np.quantile(values,.025)),'auc_upper_95':float(np.quantile(values,.975)),
            'change_from_strategy2':auc(y,score)-auc(y,baseline),
            'change_lower_95':float(np.quantile(deltas,.025)),'change_upper_95':float(np.quantile(deltas,.975))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['development','confirmation','final'])
    parser.add_argument('--threads',type=int,default=12)
    args=parser.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('Tree worker must not import PyTorch')
    data=load_data('train')
    phase=args.phase
    start=1737331200
    through={'development':3,'confirmation':5,'final':6}[phase]
    query={'development':'development','confirmation':'confirmation','final':'evaluation'}[phase]
    X,Q,meta,names=feature_set(through,query)
    y=np.asarray(data['label'][meta['first_event']:meta['last_event']]).reshape(-1)
    if len(y)!=len(X):
        raise ValueError('Fitting outcomes and feature rows disagree')
    result_path=OUT/'results.json'
    if phase=='development':
        ix=first_events(data,start+4*86400,start+6*86400)
        vy=np.asarray(data['label'][ix]).ravel()
        if len(vy)!=len(Q):
            raise ValueError('Development outcomes and feature rows disagree')
        report={'feature_names':names,'development':[],'protocol':{'development_fit':'Jan 22-23, with earlier days used for MF fitting',
                'development_validation':'First event per user Jan 24-25','confirmation_validation':'First event per user Jan 26',
                'mf_history_training':'Fitting data only; own supplied history remains an inference input',
                'identical_numeric_tree_inputs':True,'model_selection':'Development only'}}
        for kind,sizes in [('CatBoost',[6,8]),('LightGBM',[31,63])]:
            for size in sizes:
                model,record=fit(kind,size,X,y,names,'development',args,valid=(Q,vy))
                report['development'].append(record)
                atomic_json(result_path,report)
                del model
                gc.collect()
        selected=max(report['development'],key=lambda r:r['development']['roc_auc'])
        best_by_kind={kind:max([r for r in report['development'] if r['kind']==kind],key=lambda r:r['development']['roc_auc'])
                      for kind in ['CatBoost','LightGBM']}
        selection={'selected':selected,'best_by_kind':best_by_kind,
                   'seed_selection':json.loads((OUT/'seed_selection.json').read_text()),
                   'recipe':json.loads((OUT/'recipe.json').read_text())}
        atomic_json(OUT/'selection_locked.json',selection)
        report['selected']={'kind':selected['kind'],'size':selected['size'],'trees':selected['trees'],
                            'development_auc':selected['development']['roc_auc']}
        atomic_json(result_path,report)
        log('LOCKED BEFORE CONFIRMATION: '+json.dumps(report['selected']))
        return
    report=json.loads(result_path.read_text())
    selection=json.loads((OUT/'selection_locked.json').read_text())
    selected=selection['selected']
    if phase=='confirmation':
        ix=first_events(data,start+6*86400)
        labels=np.asarray(data['label'][ix])
        with np.load(ROOT/'artifacts/strategy2/confirmation_predictions.npz') as old:
            np.testing.assert_array_equal(ix,old['event_indices'])
            np.testing.assert_array_equal(labels,old['labels'])
            baseline=old['logits']
        records={}
        for kind,best in selection['best_by_kind'].items():
            model,record=fit(kind,best['size'],X,y,names,'confirmation',args,trees=best['trees'])
            scores=predict(model,kind,Q,args.threads).reshape(-1,10)
            reloaded=load_model(record)
            actual=predict(reloaded,kind,Q,args.threads).reshape(-1,10)
            np.testing.assert_allclose(scores,actual,rtol=0,atol=1e-12)
            np.savez(OUT/f'confirmation_{kind}.npz',event_indices=ix,labels=labels,logits=scores)
            records[kind]=dict(metrics=metrics(labels,scores),checkpoint_reload_verified=True,
                               model=record['path'],trees=record['trees'])
            log(kind+' CONFIRMATION: '+json.dumps(records[kind]['metrics']))
            if kind==selected['kind']:
                np.savez(OUT/'confirmation_predictions.npz',event_indices=ix,labels=labels,logits=scores)
                chosen=scores
            del model,reloaded
            gc.collect()
        fit_counts=np.bincount(data['user'][:meta['last_event']],minlength=int(data['user'].max())+1)
        new=fit_counts[data['user'][ix]]==0
        h=data['hist_len'][ix]
        slices={}
        for name,mask in [('known_user',~new),('new_user',new),('empty_history',h==0),
                          ('short_history_1_9',(h>0)&(h<10)),('history_10_plus',h>=10)]:
            if mask.any():
                slices[name]=dict(metrics(labels[mask],chosen[mask]),events=int(mask.sum()))
        report['confirmation']={'by_learner':records,'selected_kind':selected['kind'],
                                'selected':metrics(labels,chosen),'slices':slices,
                                'statistics':confidence_and_comparison(labels,chosen,baseline)}
        atomic_json(result_path,report)
        return
    report['final_models']={}
    for kind,best in selection['best_by_kind'].items():
        model,record=fit(kind,best['size'],X,y,names,'final',args,trees=best['trees'])
        scores=predict(model,kind,Q,args.threads).reshape(-1,10)
        reloaded=load_model(record)
        np.testing.assert_allclose(scores,predict(reloaded,kind,Q,args.threads).reshape(-1,10),rtol=0,atol=1e-12)
        np.save(OUT/f'evaluation_{kind}_logits.npy',scores)
        record['checkpoint_reload_verified']=True
        report['final_models'][kind]=record
        if kind==selected['kind']:
            np.save(OUT/'evaluation_logits.npy',scores)
            importance=(model.get_feature_importance() if kind=='CatBoost'
                        else model.feature_importance(importance_type='gain'))
            report['feature_importance']=sorted([{'feature':n,'importance':float(v)} for n,v in zip(names,importance)],key=lambda r:r['importance'],reverse=True)
            report['final_model']=record
            report['final_checkpoint_reload_verified']=True
        atomic_json(result_path,report)
        log('FINAL MODEL SAVED: '+record['path'])
        del model,reloaded
        gc.collect()


if __name__=='__main__':
    main()
