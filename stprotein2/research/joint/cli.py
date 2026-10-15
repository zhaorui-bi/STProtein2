"""Development split runner; prediction is saved before optional scoring."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from stprotein2.config import ModelConfig, TrainConfig, from_dict
from stprotein2.data import Cohort, PANEL, ReferenceTransform
from stprotein2.evaluation.metrics import evaluate
from stprotein2.graph import FrozenRouter, physical_graph
from stprotein2.io import freeze_run, save_torch, sha256, write_json
from stprotein2.models.residual import STProtein2
from stprotein2.training.basic import fit, predict


def run_split(args):
    specification = json.loads(Path(args.config).read_text())
    groups = json.loads(Path(args.groups).read_text())['groups']
    cohort = Cohort(args.registry, groups)
    for entry in cohort.specs.values():
        if any(accession in entry.get('rna','') or accession in entry.get('adt','')
               for accession in ('GSE278662','GSE322883')):
            raise ValueError('Reserved confirmatory cohorts cannot enter the development runner')
    model_cfg = from_dict(ModelConfig, specification.get('model', {}))
    train_cfg = from_dict(TrainConfig, specification.get('train', {}))
    output = Path(args.output).resolve()
    if args.query not in cohort.specs or args.validation not in cohort.specs:
        raise ValueError('Unknown query or validation')
    query_group, val_group = groups[args.query], groups[args.validation]
    if query_group == val_group:
        raise ValueError('Query and inner-validation groups overlap')
    query_names = [n for n,g in groups.items() if g==query_group]
    val_names = [n for n,g in groups.items() if g==val_group]
    if len(val_names)!=1:
        raise ValueError('This initial single-inner-group pilot requires a one-slice validation group; use nested runner for multi-slice validation')
    references = [n for n,g in groups.items() if g not in {query_group,val_group}]
    all_references = [n for n,g in groups.items() if g != query_group]
    inputs = {entry[modality]:sha256(entry[modality]) for name,entry in cohort.specs.items() for modality in ('rna','adt')}
    freeze_run(output, {'stage':'development_single_inner_group_pilot','model':model_cfg.to_dict(),
               'train':train_cfg.to_dict(),'groups':groups,'query_names':query_names,'validation_names':val_names,
               'inner_training_names':references,'refit_names':all_references,'input_sha256':inputs,
               'query_labels_for_selection':False,'oof_gate_selection':'not yet performed; open gates exploratory'})
    with threadpool_limits(train_cfg.threads):
        inner_dir=output/'inner'
        if not (inner_dir/'training.json').exists():
            print('Preparing inner references: '+','.join(references),flush=True)
            transform=ReferenceTransform(model_cfg.n_genes).fit([cohort.rna(n) for n in references],references)
            paired=cohort.references(references,transform,train_cfg.count_keep_rates,train_cfg.count_views_per_rate,train_cfg.seed)
            router=FrozenRouter(seed=train_cfg.seed).fit([s.rna.x for s in paired],[s.rna.group for s in paired])
            validation=cohort.references(val_names,transform)[0]
            _,report=fit(paired,model_cfg,train_cfg,router,inner_dir,args.device,validation)
        else:
            report=json.loads((inner_dir/'training.json').read_text())
        epochs=int(report['selected_epoch'])
        final_train=replace(train_cfg,epochs=epochs,warmup_epochs=min(train_cfg.warmup_epochs,epochs))
        final_dir=output/'refit'
        if not (final_dir/'training.json').exists():
            print(f'Refit all references for {epochs} epochs',flush=True)
            transform=ReferenceTransform(model_cfg.n_genes).fit([cohort.rna(n) for n in all_references],all_references)
            paired=cohort.references(all_references,transform,final_train.count_keep_rates,final_train.count_views_per_rate,final_train.seed)
            router=FrozenRouter(seed=train_cfg.seed).fit([s.rna.x for s in paired],[s.rna.group for s in paired])
            joblib.dump({'transform':transform,'router':router,'train_config':final_train.to_dict()},output/'preprocessing.joblib')
            model,report=fit(paired,model_cfg,final_train,router,final_dir,args.device)
        else:
            saved=joblib.load(output/'preprocessing.joblib')
            transform,router=saved['transform'],saved['router']
            checkpoint=torch.load(final_dir/'model.pt',map_location=args.device,weights_only=True)
            model=STProtein2(from_dict(ModelConfig,checkpoint['model_config'])).to(args.device)
            model.load_state_dict(checkpoint['state_dict'])
        print('RNA-only query prediction',flush=True)
        before=list(cohort.protein_reads)
        predictions={}
        for query in query_names:
            view=cohort.input(query,transform)
            arrays,diagnostics=predict(model,view,router,final_train,args.device)
            dest=output/'predictions'/query;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'prediction.npz',**arrays,proteins=np.asarray(PANEL))
            write_json(dest/'diagnostics.json',diagnostics)
            predictions[query]=sha256(dest/'prediction.npz')
        if before!=cohort.protein_reads:
            raise RuntimeError('Prediction accessed measured protein')
        write_json(output/'prediction_complete.json',{'queries':query_names,'sha256':predictions,
                     'protein_reads_before_prediction':before,'query_protein_read_for_prediction':False})
    print('PREDICTIONS_COMPLETE '+str(output),flush=True)


def score_run(args):
    output=Path(args.output).resolve()
    manifest=json.loads((output/'manifest.json').read_text())
    completed=json.loads((output/'prediction_complete.json').read_text())
    cohort=Cohort(args.registry,manifest['protocol']['groups'])
    summaries={}
    for query in completed['queries']:
        for modality in ('rna','adt'):
            source=cohort.specs[query][modality]
            if source not in manifest['protocol']['input_sha256'] or sha256(source)!=manifest['protocol']['input_sha256'][source]:
                raise ValueError('Scoring input differs from frozen run identity')
        path=output/'predictions'/query/'prediction.npz'
        if sha256(path)!=completed['sha256'][query]:
            raise ValueError('Prediction checksum changed')
        values=np.load(path,allow_pickle=False)
        if not np.array_equal(values['barcodes'],np.asarray(cohort.rna(query).obs_names,dtype=str)) or tuple(values['proteins'])!=PANEL:
            raise ValueError('Prediction identity mismatch')
        truth=cohort.measured_protein(query)
        graph=physical_graph(cohort.rna(query).obsm['spatial'])
        report=evaluate(truth,values['prediction'],PANEL,graph.edge_index)
        report['base_summary']=evaluate(truth,values['base'],PANEL)['summary']
        if 'ungated_prediction' in values:
            report['exploratory_ungated_summary']=evaluate(truth,values['ungated_prediction'],PANEL)['summary']
        report['scope']='seen development query; not independent confirmation or SOTA evidence alone'
        write_json(path.parent/'scores.json',report)
        summaries[query]=report['summary']
    write_json(output/'development_scores.json',summaries)
    print(json.dumps(summaries,indent=2),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='stprotein2')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run-split',help='single-inner-group development pilot, save query predictions without scoring')
    for name in ('registry','groups','config','query','validation','output'):
        run.add_argument('--'+name,required=True)
    run.add_argument('--device',default='cpu')
    run.set_defaults(function=run_split)
    nested=sub.add_parser('run-nested',help='development leave-source-group-out with nested epoch and OOF gate selection')
    for name in ('registry','groups','config','query','output'):
        nested.add_argument('--'+name,required=True)
    nested.add_argument('--device',default='cpu')
    from stprotein2.research.joint.nested import run_nested
    nested.set_defaults(function=run_nested)
    score=sub.add_parser('score',help='separately score saved predictions on seen development data')
    score.add_argument('--registry',required=True);score.add_argument('--output',required=True)
    score.set_defaults(function=score_run)
    args=parser.parse_args(argv)
    args.function(args)

