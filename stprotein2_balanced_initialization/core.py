"""Unique canonical RNA sampling; the frozen original initializer stays unchanged."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch

from stprotein2.data import RNAView
from stprotein2.graph import FrozenRouter
from stprotein2.trainer import tensor_input


@dataclass(frozen=True)
class SamplingPlan:
    indices: dict
    record: dict


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name+" must be an integer >= "+str(minimum))


def sample_references(views, *, reference_names, per_source, seed=2025):
    """Give each source B spots, water-fill its slice quotas, never duplicate IDs.

    The allowed names are explicit so callers cannot accidentally append a query
    view. Fold/source exclusion remains the nested experiment runner's duty.
    Sampling and its record are invariant to input row and view order.
    """
    _integer(per_source,"per_source",1)
    _integer(seed,"seed",0)
    if not views or any(not isinstance(v,RNAView) for v in views):
        raise ValueError("Nonempty RNA-only reference views required")
    names=[v.name for v in views]
    if (len(set(names))!=len(names) or len(set(reference_names))!=len(reference_names)
            or set(names)!=set(reference_names)):
        raise ValueError("Exact unique allowed reference names required")
    if any(len(v.x)==0 or not v.group or not v.name for v in views):
        raise ValueError("Each reference needs a source, name, and at least one spot")
    if any(v.genes!=views[0].genes for v in views):
        raise ValueError("Reference RNA gene axes differ")
    by_name={v.name:v for v in views}
    sources=sorted({v.group for v in views})
    members={g:sorted(v.name for v in views if v.group==g) for g in sources}
    count=min(per_source,min(sum(len(by_name[n].x) for n in members[g]) for g in sources))
    indices={};records=[]
    for group in sources:
        ordered=members[group]
        # Rotating the stable slice order allocates remainder quotas without row-order dependence.
        offset=seed%len(ordered)
        order=ordered[offset:]+ordered[:offset]
        quotas={name:0 for name in ordered}
        remaining=count
        while remaining:
            for name in order:
                if remaining and quotas[name]<len(by_name[name].x):
                    quotas[name]+=1
                    remaining-=1
        for name in ordered:
            view=by_name[name]
            canonical=np.argsort(view.barcodes,kind="stable")
            salt=int.from_bytes(hashlib.sha256((group+"\0"+name).encode()).digest()[:8],"little")
            rng=np.random.default_rng([seed,salt])
            chosen=np.sort(rng.choice(len(view.x),quotas[name],replace=False))
            selected=canonical[chosen]
            indices[name]=selected
            records.append({"source":group,"slice":name,"available_spots":len(view.x),
                            "quota":quotas[name],"barcodes":view.barcodes[selected].tolist()})
    identity={"rule":"equal sources; capped equal slice quotas; unique canonical barcodes",
              "seed":seed,"per_source_limit":per_source,"actual_per_source":count,
              "reference_names":sorted(names),"reference_groups":sources,"slices":records}
    identity["sampling_sha256"]=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return SamplingPlan(indices,identity)


def _normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def fit_router(views, *, reference_names, components=50, clusters=32, per_source=4000, seed=2025):
    """Same PCA/spherical-center math as the original router; different recorded sampling."""
    _integer(components,"components",1)
    _integer(clusters,"clusters",2)
    plan=sample_references(views,reference_names=reference_names,per_source=per_source,seed=seed)
    by_name={v.name:v for v in views}
    matrix=np.concatenate([by_name[r["slice"]].x[plan.indices[r["slice"]]] for r in plan.record["slices"]])
    dimension=min(components,matrix.shape[1],len(matrix)-1)
    if dimension<1 or len(matrix)<clusters:
        raise ValueError("Insufficient unique reference samples for router")
    router=FrozenRouter(components=components,clusters=clusters,seed=seed,per_group=per_source)
    router.pca_=PCA(n_components=dimension,svd_solver="randomized",random_state=seed).fit(matrix)
    # Truncated PCA components can be noncontiguous; canonicalize before the first
    # transform so joblib round-tripping does not switch BLAS layout/roundoff.
    for name,value in vars(router.pca_).copy().items():
        if isinstance(value,np.ndarray):
            setattr(router.pca_,name,np.ascontiguousarray(value))
    z=_normalize(router.pca_.transform(matrix))
    centers=_normalize(KMeans(clusters,n_init=5,random_state=seed).fit(z).cluster_centers_)
    for _ in range(10):
        labels=(z@centers.T).argmax(1)
        centers=_normalize(np.stack([z[labels==j].mean(0) if np.any(labels==j) else centers[j] for j in range(clusters)]))
    router.centers_=centers.astype(np.float32)
    router.fitted_groups_=plan.record["reference_groups"]
    router.samples_per_group_=plan.record["actual_per_source"]
    router.sampling_record_=plan.record
    router.initialization_variant_="source_slice_spot_balanced_v1"
    return router,plan.record


@torch.no_grad()
def initialize_prototypes(model, views, graphs, *, reference_names, device="cpu", per_source=2000, seed=2025):
    """Initialize state then niche from RNA-only inputs, preserving Student train mode.

    The caller constructs/copies Teacher after this operation. No independent
    Teacher clustering, protein reading, or query fitting is performed here.
    """
    plan=sample_references(views,reference_names=reference_names,per_source=per_source,seed=seed)
    total=plan.record["actual_per_source"]*len(plan.record["reference_groups"])
    active=[name for name in ("state","niche") if getattr(model.representations,name) is not None]
    if any(total<len(getattr(model.representations,name).codes) for name in active):
        raise ValueError("Insufficient unique reference points for every active prototype bank")
    if set(graphs)!=set(reference_names) or any(graphs[v.name].n!=len(v.x) for v in views):
        raise ValueError("Exactly matching full reference graphs required")
    by_name={v.name:v for v in views}
    was_training=model.training
    model.eval()
    stages=[]
    try:
        for name in active:
            selected=[]
            for record in plan.record["slices"]:
                view=by_name[record["slice"]]
                output=model.representations(*tensor_input(view,graphs[view.name],device))
                ids=torch.as_tensor(plan.indices[view.name],device=device,dtype=torch.long)
                selected.append(output[name]["z"].index_select(0,ids).cpu())
            matrix=torch.cat(selected)
            getattr(model.representations,name).initialize(matrix,seed)
            stages.append({"bank":name,"unique_samples":len(matrix),
                           "sample_feature_sha256":hashlib.sha256(matrix.contiguous().numpy().tobytes()).hexdigest(),
                           "niche_uses_initialized_state":name=="niche" and "state" in active})
    finally:
        model.train(was_training)
    return {"sampling":plan.record,"stages":stages,"Teacher_action":"copy Student after initialization",
            "variant":"source_slice_spot_balanced_v1","measured_protein_access":False}
