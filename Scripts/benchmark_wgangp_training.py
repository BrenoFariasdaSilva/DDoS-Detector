import argparse, importlib.util, sys, time, json, inspect
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas.io.formats.style
import torch

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

class SilentProgress:
    def __init__(self, batches): self.batches=batches
    def __iter__(self): return iter(self.batches)
    def set_description(self,*args,**kwargs): pass

def synchronize(device):
    if device.type=='cuda':
        for index in range(torch.cuda.device_count()): torch.cuda.synchronize(index)

def trial(module, device, parallel, seed, count, warmup=0, paired=False, large=False, latam=False, primary_critic=False, trace=None):
    # Inputs generated from a dedicated CPU RNG so model and training RNGs are untouched.
    if latam: large=True
    batch=256 if large else 64; width=16 if latam else (95 if large else 10); numeric=width if latam else width-4
    groups=[] if latam else [(numeric,numeric+2),(numeric+2,width)]
    masks=[] if latam else [np.array([[1,1],[1,0]],dtype=bool),np.array([[1,0],[1,1]],dtype=bool)]
    rng=torch.Generator().manual_seed(seed+901)
    ref=torch.randn(512,width,generator=rng); labels=torch.arange(512)%2
    for (start,stop), mask in zip(groups,masks):
        cat=torch.randint(0,2,(512,),generator=rng); cat[~torch.tensor(mask)[labels,cat]]=0
        ref[:,start:stop]=torch.nn.functional.one_hot(cat,2).float()
    moments=module.build_class_reference_moments(ref,labels.numpy(),sample_limit=256,seed=seed,categorical_groups=groups)
    correlations=module.build_class_reference_correlations(ref,labels.numpy(),sample_limit=256,seed=seed)
    quantiles=module.build_class_reference_numeric_quantiles(ref,labels.numpy(),numeric,sample_limit=256,seed=seed)
    module.V61_REFERENCE_MOMENTS={k:tuple(torch.as_tensor(v,device=device) for v in t) for k,t in moments.items()}
    module.V6_REFERENCE_CORRELATIONS={k:tuple(torch.as_tensor(v,device=device) for v in t) for k,t in correlations.items()}
    module.V64_REFERENCE_NUMERIC_QUANTILES={k:tuple(torch.as_tensor(v,device=device) for v in t) for k,t in quantiles.items()}
    batches=[]
    for _ in range(count+warmup):
        ix=torch.cat([torch.randint(0,256,(batch//2,),generator=rng)*2,torch.randint(0,256,(batch//2,),generator=rng)*2+1]); ix=ix[torch.randperm(batch,generator=rng)]; batches.append((ref[ix],labels[ix]))
    module.set_seed(seed)
    G=module.Generator(100 if large else 12,width,2,hidden_dims=[256,512] if large else [24,32],embed_dim=32 if large else 8,n_resblocks=3 if large else 2,categorical_groups=groups,categorical_class_masks=masks).to(device)
    D=module.Discriminator(width,2,hidden_dims=[512,256,128] if large else [32,16],embed_dim=32 if large else 8).to(device)
    if parallel:
        if hasattr(module,'wrap_training_parallel_models'):
            with redirect_stdout(sys.stderr):
                G,D=module.wrap_training_parallel_models(G,D,SimpleNamespace(gpu=None),{'training':{'critic_execution':'primary' if primary_critic else 'legacy'}},device)
        else:
            if primary_critic: raise ValueError('Candidate does not support primary critic')
            G=torch.nn.DataParallel(G); D=torch.nn.DataParallel(D)
    optG=torch.optim.Adam(G.parameters(),lr=1e-4,betas=(.5,.9)); optD=torch.optim.Adam(D.parameters(),lr=1e-4,betas=(.5,.9))
    config=module.get_default_config(); config['training']['coalesce_critic_forwards']=paired
    args=SimpleNamespace(batch_size=batch,latent_dim=100 if large else 12,critic_steps=5,lambda_gp=10.,use_amp=False,_mixed_categorical_groups=groups,_mixed_numeric_feature_count=numeric,csv_path='benchmark.csv',epochs=1,log_interval=1)
    hist={key:[] for key in ('steps','loss_D','loss_G','gp','D_real','D_fake','wasserstein')}
    def run(selected,step):
        return module.run_batch_training_loop(G,D,optG,optD,None,device,args,config,2,SilentProgress(selected),step,hist,len(selected),0,time.time(),'')
    if latam:
        batches=[(x.to(device),y.to(device)) for x,y in batches]  # Match the production GPU-resident source path.
        args.log_interval=50  # Match production logging cadence.
    if warmup: run(batches[:warmup],0)
    synchronize(device); start=time.perf_counter(); run(batches[warmup:],warmup); synchronize(device); elapsed=time.perf_counter()-start
    state={name:{k:v.detach().cpu().clone() for k,v in module.unwrap_flow_model(model).state_dict().items()} for name,model in [('G',G),('D',D)]}
    optimizer={}
    for name,opt in [('G',optG),('D',optD)]:
        for i,entry in enumerate(opt.state.values()):
            for key,v in entry.items():
                if torch.is_tensor(v): optimizer[f'{name}.{i}.{key}']=v.detach().cpu().clone()
    rng_state=[torch.get_rng_state().clone()]+([v.clone() for v in torch.cuda.get_rng_state_all()] if device.type=='cuda' else [])
    # Inference equivalence on identical latent draws, including masked category sampling.
    G.eval(); module.set_seed(seed+12345)
    with torch.no_grad():
        result=G(torch.randn(batch,args.latent_dim,device=device),torch.arange(batch,device=device)%2).cpu()
    if trace is not None:
        G.train(); D.train()
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device.type=='cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            run(batches[warmup:warmup+min(3,count)],warmup+count)
            synchronize(device)
        profiler.export_chrome_trace(str(trace))
        print(profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=20),file=sys.stderr)
    return dict(elapsed=elapsed,hist=hist,state=state,optimizer=optimizer,rng=rng_state,sample=result)

def compare(a,b):
    maximum=0.; exact=True; close=True
    values=[]
    for key in a['state']:
        values.extend((x,b['state'][key][n]) for n,x in a['state'][key].items())
    values.extend((x,b['optimizer'][n]) for n,x in a['optimizer'].items())
    values.extend((torch.tensor(x,dtype=torch.float64),torch.tensor(b['hist'][k],dtype=torch.float64)) for k,x in a['hist'].items())
    values.append((a['sample'],b['sample']))
    for x,y in values:
        if not torch.equal(x,y): exact=False
        if x.is_floating_point():
            maximum=max(maximum,(x-y).abs().max().item() if x.numel() else 0.)
            if not torch.allclose(x,y,atol=1e-6,rtol=1e-5): close=False
        elif not torch.equal(x,y): close=False
    rng=all(torch.equal(x,y) for x,y in zip(a['rng'],b['rng'])) and len(a['rng'])==len(b['rng'])
    return {'bitwise_equal':exact,'allclose':close,'max_abs_difference':maximum,'rng_equal':rng}

def main():
    p=argparse.ArgumentParser(description='Compare baseline and updated WGAN-GP on identical bounded mixed-type training batches; no dataset or checkpoint writes.')
    p.add_argument('--baseline',type=Path,required=True); p.add_argument('--candidate',type=Path,default=Path(__file__).resolve().parents[1]/'wgangp.py'); p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--data-parallel',action='store_true'); p.add_argument('--paired-critic',action='store_true',help='Experimental paired forward for candidate only'); p.add_argument('--steps',type=int,default=30); p.add_argument('--warmup',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--repetitions',type=int,default=3); p.add_argument('--small',action='store_true',help='Use small networks for quick smoke tests'); p.add_argument('--require-bitwise',action='store_true',help='Fail if any checked training tensor, metric or sample is not bitwise equal'); p.add_argument('--latam',action='store_true',help='Numeric-only 16-feature, 256-row GPU-resident LATAM workload'); p.add_argument('--primary-critic',action='store_true',help='Experimental candidate-only primary GPU critic with unchanged generator topology'); p.add_argument('--min-speedup',type=float,default=1.0); p.add_argument('--profile-dir',type=Path,help='Optional separate 3-batch profiler traces for baseline and candidate'); ns=p.parse_args()
    if min(ns.steps,ns.repetitions)<1 or ns.warmup<0: p.error('invalid step count')
    if ns.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA is unavailable')
    if ns.data_parallel and (ns.device!='cuda' or torch.cuda.device_count()<2): p.error('DataParallel requires at least two visible CUDA devices')
    if ns.primary_critic and not ns.data_parallel: p.error('--primary-critic requires --data-parallel')
    if ns.small and ns.latam: p.error('--small and --latam are mutually exclusive')
    if not np.isfinite(ns.min_speedup) or ns.min_speedup<=0: p.error('--min-speedup must be positive')
    if ns.device=='cpu': torch.set_num_threads(1)
    if ns.device=='cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark=True  # Match production runtime hardware setup.
    # Repository root is needed for Logger, telegram and generation helpers.
    sys.path.insert(0,str(ns.candidate.resolve().parent)); baseline=load_module(ns.baseline,'wgangp_benchmark_baseline'); candidate=load_module(ns.candidate,'wgangp_benchmark_candidate'); device=torch.device(ns.device)
    results=[]
    for i in range(ns.repetitions):
        # Alternate order to reduce systematic warm-cache bias. Warm-up is identical.
        if i%2==0:
            a=trial(baseline,device,ns.data_parallel,ns.seed+i,ns.steps,ns.warmup,large=not ns.small,latam=ns.latam); b=trial(candidate,device,ns.data_parallel,ns.seed+i,ns.steps,ns.warmup,ns.paired_critic,large=not ns.small,latam=ns.latam,primary_critic=ns.primary_critic)
        else:
            b=trial(candidate,device,ns.data_parallel,ns.seed+i,ns.steps,ns.warmup,ns.paired_critic,large=not ns.small,latam=ns.latam,primary_critic=ns.primary_critic); a=trial(baseline,device,ns.data_parallel,ns.seed+i,ns.steps,ns.warmup,large=not ns.small,latam=ns.latam)
        results.append(dict(seed=ns.seed+i,baseline_batches_s=ns.steps/a['elapsed'],candidate_batches_s=ns.steps/b['elapsed'],speedup=a['elapsed']/b['elapsed'],**compare(a,b)))
    import statistics
    print(json.dumps({'baseline':baseline.WGANGP_RELEASE_VERSION,'candidate':candidate.WGANGP_RELEASE_VERSION,'device':ns.device,'data_parallel':ns.data_parallel,'experimental_paired_critic':ns.paired_critic,'experimental_primary_critic':ns.primary_critic,'latam_numeric_workload':ns.latam,'results':results,'median_speedup':statistics.median(r['speedup'] for r in results),'warning':'A bounded training benchmark is not a full generated-data quality audit.'},indent=2))
    if ns.profile_dir is not None:
        ns.profile_dir.mkdir(parents=True,exist_ok=True)
        trial(baseline,device,ns.data_parallel,ns.seed,3,ns.warmup,large=not ns.small,latam=ns.latam,trace=ns.profile_dir/'baseline.json')
        trial(candidate,device,ns.data_parallel,ns.seed,3,ns.warmup,ns.paired_critic,large=not ns.small,latam=ns.latam,primary_critic=ns.primary_critic,trace=ns.profile_dir/'candidate.json')
    if not all(r['allclose'] and r['rng_equal'] and (not ns.require_bitwise or r['bitwise_equal']) for r in results): sys.exit(1)
    if statistics.median(r['speedup'] for r in results)<ns.min_speedup: sys.exit(2)

if __name__=='__main__': main()
