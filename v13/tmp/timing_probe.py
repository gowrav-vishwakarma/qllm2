import sys, time, inspect, torch, math
sys.path.insert(0,'.')
from v13.model import V13LM, get_config
from v7.data import load_pretrain_mix, resolve_amp_dtype, build_lr_scheduler, build_param_groups
from v7.train import seed_everything
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high'); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
seed_everything(42)
cfg=get_config('v13_e3_k3_selective'); cfg.max_seq_len=2048
ds,val,tk=load_pretrain_mix(seq_len=2048,edu_score_min=3,token_budget=500_000_000,
    sources=('dclm','fineweb','smoltalk2_mid'),weights=(48,48,4),chat_vocab=True,
    fineweb_name='sample-10BT',holdout_pct=5,mix_seed=42,skip_docs={'dclm':0,'fineweb':0,'smoltalk2_mid':0},
    blend_warmup_tokens=1_000_000_000,token_counters={},doc_counters={})
dl=DataLoader(ds,batch_size=18,shuffle=False,num_workers=0,pin_memory=True)
m=V13LM(cfg).cuda().train()
pg=build_param_groups(m,weight_decay=0.01)
opt=torch.optim.AdamW(pg,lr=3e-4,betas=(0.9,0.95),fused=True)
sched=build_lr_scheduler(opt,'warmup_cosine',warmup_steps=500,total_steps=14063)
amp=resolve_amp_dtype('auto'); use=amp is not None
raw=m; emits='return_nll' in inspect.signature(m.ce_from_lm).parameters
it=iter(dl); t0=time.time(); gtok=0; step=0
while step<12:
    try: b=next(it)
    except StopIteration: it=iter(dl); b=next(it)
    x=b['input_ids'].cuda(non_blocking=True); lab=b['labels'].cuda(non_blocking=True)
    lm= m._hidden_to_lm(x)[0]
    main,nll=raw.ce_from_lm(lm,lab,chunk=4096,return_nll=emits)
    loss=main
    loss.backward()
    gnorm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0))
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step+=1; gtok+=x.shape[0]*x.shape[1]
    if step%10==0:
        print(f"LEAN s{step} {gtok/(time.time()-t0):.0f}t/s  (no aux, no hook, no state)",flush=True)
print(f"LEAN DONE {gtok/(time.time()-t0):.0f}t/s",flush=True)
