"""Equivalent algebra, stochastic checkpoints and real optimizer/Teacher state."""
import copy
from dataclasses import replace
import io

import pytest
import torch
from torch.nn import functional as F

from stprotein2.model import STProtein2
from stprotein2.prototypes import EnsembleLinear, EMATeacher
from stprotein2_execution import ExecutionPolicy, configure_execution
from stprotein2_execution.memory import WeightModulatedEnsembleLinear
from test_architecture import setup_model


@pytest.mark.parametrize("member_input", [False,True])
def test_ensemble_arithmetic_and_all_gradients_match_double_precision(member_input):
    torch.manual_seed(35)
    original = EnsembleLinear(11,7,4).double()
    fast = copy.deepcopy(original)
    fast.__class__ = WeightModulatedEnsembleLinear
    shape = (19,4,11) if member_input else (19,11)
    x = torch.randn(shape,dtype=torch.float64,requires_grad=True)
    z = x.detach().clone().requires_grad_()
    a,b = original(x),fast(z)
    torch.testing.assert_close(a,b,atol=2e-15,rtol=3e-14)
    weight = torch.randn_like(a)
    (a*weight).sum().backward();(b*weight).sum().backward()
    torch.testing.assert_close(x.grad,z.grad,atol=2e-14,rtol=2e-13)
    for (name,p),(other,q) in zip(original.named_parameters(),fast.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,atol=1e-13,rtol=2e-13)


def _model(device="cpu"):
    source,args,parts = setup_model(device)
    cfg = replace(source.cfg,dropout=.15)
    model = STProtein2(cfg).to(device)
    model.load_state_dict(source.state_dict())
    return model,args,parts


def _seed(device,seed):
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed_all(seed)


def _step(model, teacher, optimizer, args, parts, seed=223):
    device=args[0].device.type
    if seed is not None:
        _seed(device,seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    target=torch.sin(torch.arange(len(args[0])*model.cfg.n_proteins,device=device,dtype=args[0].dtype)).reshape(len(args[0]),-1)
    result=model(*args,partitions=parts)
    taught=teacher(*args)
    proto=teacher.loss(result["representations"],taught,.05)
    auxiliary=result.get("auxiliary_members",result["base_members"])
    loss=F.mse_loss(result["prediction"],target)+.2*F.mse_loss(auxiliary,target[:,None].expand_as(auxiliary))
    loss=loss+.1*proto["state"]+.05*proto["niche"]
    loss.backward()
    gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    teacher.update(model.representations,.995)
    teacher.update_centers(taught)
    return result,gradients,loss.detach()


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_nonzero_full_Halo_training_all_gradients_and_teacher_updates(device):
    source,args,parts=_model(device)
    _seed(device,13)
    expected=copy.deepcopy(source)
    teacher=EMATeacher(expected.representations).to(device)
    optimizer=torch.optim.AdamW(expected.parameters(),lr=.0001,weight_decay=.0001)
    reference,gradients,loss=_step(expected,teacher,optimizer,args,None)
    assert reference["local_delta"].abs().max()>1e-4
    for checkpoint in (False,True):
        model=copy.deepcopy(source)
        parameters={n:id(p) for n,p in model.named_parameters()}
        identities=list(model.state_dict())
        configure_execution(model,ExecutionPolicy(True,checkpoint))
        assert list(model.state_dict())==identities
        assert parameters=={n:id(p) for n,p in model.named_parameters()}
        target_teacher=EMATeacher(model.representations).to(device)
        target_optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
        output,current,current_loss=_step(model,target_teacher,target_optimizer,args,[p.to(device) for p in reversed(parts)])
        for name in ("prediction","local_delta","state_delta"):
            torch.testing.assert_close(output[name],reference[name],atol=1e-5,rtol=2e-5)
        torch.testing.assert_close(current_loss,loss,atol=1e-6,rtol=1e-5)
        assert set(current)==set(gradients)
        for name in gradients:
            if gradients[name] is None:
                assert current[name] is None
            else:
                torch.testing.assert_close(current[name],gradients[name],atol=5e-6,rtol=2e-4,msg=lambda m:name+": "+m)
        assert all(p.grad is None for p in target_teacher.parameters())
        assert teacher.update_count.item()==target_teacher.update_count.item()==1
        # Adam can amplify a tiny near-zero-gradient sign difference; report bounded parameter tolerance.
        for name,value in expected.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],value,atol=3e-5,rtol=2e-4)
        for name,value in teacher.state_dict().items():
            torch.testing.assert_close(target_teacher.state_dict()[name],value,atol=2e-6,rtol=2e-4)
        # Availability masking is preserved by the algebraic input kernel.
        model.eval()
        x,mask,qc,graph=args
        mask=mask.clone();mask[:,2]=0
        changed=x.clone();changed[:,2]=10000
        with torch.no_grad():
            # Test the actual masking path exactly, before nondeterministic CUDA graph reductions.
            encoded=model.representations.rna(x,mask,qc)
            changed_encoded=model.representations.rna(changed,mask,qc)
            assert torch.equal(encoded,changed_encoded)
            a=model(x,mask,qc,graph)["prediction"]
            b=model(changed,mask,qc,graph)["prediction"]
        if device=="cpu":
            assert torch.equal(a,b)
        else:
            # Repeating even identical CUDA graph inputs can change atomic-sum roundoff.
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)


def test_policy_reversible_and_input_flags_validated():
    model,args,_=_model()
    before=copy.deepcopy(model.state_dict())
    ids={n:id(p) for n,p in model.named_parameters()}
    configure_execution(model,ExecutionPolicy(True,True))
    configure_execution(model,ExecutionPolicy(False,False))
    assert ids=={n:id(p) for n,p in model.named_parameters()}
    assert all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())
    with pytest.raises(ValueError):
        ExecutionPolicy(weight_modulated_input="yes")
    with pytest.raises(ValueError):
        ExecutionPolicy(weight_modulated_input=float("nan"))
    with pytest.raises(TypeError):
        configure_execution(model,{"weight_modulated_input":True})
    with pytest.raises(ValueError):
        configure_execution(torch.nn.Linear(4,3))


def test_checkpoint_preserves_rng_and_exact_interrupted_optimizer_teacher_restore():
    model,args,parts=_model()
    configure_execution(model,ExecutionPolicy(True,True))
    teacher=EMATeacher(model.representations)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    partitions=[p.to("cpu") for p in parts]
    _step(model,teacher,optimizer,args,partitions,seed=440)
    saved={"model":model.state_dict(),"teacher":teacher.state_dict(),"optimizer":optimizer.state_dict(),
           "rng":torch.get_rng_state(),"policy":ExecutionPolicy(True,True).to_dict()}
    buffer=io.BytesIO();torch.save(saved,buffer)
    _step(model,teacher,optimizer,args,partitions,seed=None)
    buffer.seek(0);restored=torch.load(buffer,weights_only=True)
    other,_,_=_model()
    configure_execution(other,ExecutionPolicy(**restored["policy"]))
    other.load_state_dict(restored["model"])
    other_teacher=EMATeacher(other.representations);other_teacher.load_state_dict(restored["teacher"])
    other_optimizer=torch.optim.AdamW(other.parameters(),lr=.001);other_optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"])
    _step(other,other_teacher,other_optimizer,args,partitions,seed=None)
    assert all(torch.equal(v,other.state_dict()[n]) for n,v in model.state_dict().items())
    assert all(torch.equal(v,other_teacher.state_dict()[n]) for n,v in teacher.state_dict().items())
    # Checkpoint recomputation must not consume extra global RNG draws.
    eager,args,parts=_model()
    configure_execution(eager,ExecutionPolicy(True,False))
    checked=copy.deepcopy(eager);configure_execution(checked,ExecutionPolicy(True,True))
    final=[]
    for candidate in (eager,checked):
        _seed("cpu",998)
        candidate.train()
        value=candidate(*args,partitions=[p.to("cpu") for p in parts])
        value["prediction"].square().mean().backward()
        final.append(torch.get_rng_state())
    assert torch.equal(final[0],final[1])
