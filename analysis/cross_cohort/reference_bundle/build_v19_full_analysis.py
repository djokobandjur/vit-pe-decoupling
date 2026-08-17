#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, itertools, collections, math, csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

SEEDS=[42,123,456,789,1011,1213]
FAMILIES=['learned','sinusoidal','rope','alibi']
FAMILY_LABELS={'learned':'Learned','sinusoidal':'Sinusoidal','rope':'RoPE','alibi':'ALiBi'}
ARCH_LABELS={'vitb':'ViT-B/16','vits':'ViT-S/16'}
TAU_ABS=5e-7
TAU_REL=5e-6
BOOTSTRAP_REPS=200_000
BOOTSTRAP_SEED=20260802


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def near(a,b):
    return abs(a-b)<=max(TAU_ABS,TAU_REL*max(abs(a),abs(b),1e-12))

def dedupe_rows(points:pd.DataFrame, regime:str)->pd.DataFrame:
    pts=points.copy().sort_values(['rho','budget','source_index']).reset_index(drop=True)
    groups=[]
    for _,r in pts.iterrows():
        if not groups or not near(float(groups[-1][0]['rho']),float(r['rho'])):
            groups.append([r])
        else:
            groups[-1].append(r)
    selected=[]
    for g in groups:
        if regime=='adversarial':
            def key(r):
                loss=float(r.get('attack_loss',np.nan))
                loss_key=-loss if np.isfinite(loss) else float('inf')
                return (float(r['accuracy']),loss_key,float(r['budget']),int(r['source_index']))
            q=min(g,key=key)
        else:
            # Noise has already been averaged within budget; near-duplicate rho groups
            # are averaged because there is no loss-based selection rule.
            q=g[0].copy()
            q['rho']=float(np.mean([float(x['rho']) for x in g]))
            q['accuracy']=float(np.mean([float(x['accuracy']) for x in g]))
            q['budget']=float(min(float(x['budget']) for x in g))
            q['source_index']=int(min(int(x['source_index']) for x in g))
        selected.append(dict(q))
    return pd.DataFrame(selected).sort_values('rho').reset_index(drop=True)

def construct_curve(points:pd.DataFrame, regime:str, envelope:bool)->tuple[np.ndarray,np.ndarray]:
    pts=points[np.isfinite(points['rho']) & np.isfinite(points['accuracy']) & (points['rho']>0)].copy()
    pts=dedupe_rows(pts,regime)
    x=np.concatenate(([0.0],pts['rho'].to_numpy(float)))
    y=np.concatenate(([1.0],pts['accuracy'].to_numpy(float)))
    if regime=='adversarial' and envelope:
        y=np.minimum.accumulate(y)
    if not np.all(np.diff(x)>0):
        raise ValueError('rho not strictly increasing')
    return x,y

def integrate_nauc(x,y,end):
    if x[-1]+1e-15<end: raise ValueError(f'support {x[-1]} < endpoint {end}')
    inside=x<end
    xc=np.concatenate((x[inside],[end]))
    yc=np.concatenate((y[inside],[np.interp(end,x,y)]))
    return float(np.trapezoid(yc,xc)/end)

def threshold_crossing(x,y,q=0.5):
    # assumes monotone lower envelope, includes clean anchor
    yenv=np.minimum.accumulate(y)
    idx=np.where(yenv<=q)[0]
    if len(idx)==0:
        return {'status':'NOT_REACHED','rho':None,'left_rho':None,'right_rho':None}
    j=int(idx[0])
    if j==0:
        return {'status':'AT_OR_BELOW_CLEAN','rho':0.0,'left_rho':0.0,'right_rho':0.0}
    x0,x1=float(x[j-1]),float(x[j]); y0,y1=float(yenv[j-1]),float(yenv[j])
    if near(x0,x1) or abs(y1-y0)<1e-15:
        rho=x1
    else:
        rho=x0+(q-y0)*(x1-x0)/(y1-y0)
    return {'status':'ESTIMATED','rho':float(rho),'left_rho':x0,'right_rho':x1}

def paired_bootstrap_ci(vals,seed):
    vals=np.asarray(vals,float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(vals),size=(BOOTSTRAP_REPS,len(vals)))
    means=vals[idx].mean(axis=1)
    return tuple(map(float,np.quantile(means,[.025,.975])))

def exact_sign_flip(vals):
    vals=np.asarray(vals,float); obs=abs(vals.mean())
    signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(vals))))
    stats=np.abs((signs*vals[None,:]).mean(axis=1))
    return float(np.mean(stats>=obs-1e-15))

def exact_friedman(matrix):
    matrix=np.asarray(matrix,float); n,k=matrix.shape
    if any(len(np.unique(row))!=k for row in matrix):
        return {'status':'TIES_PRESENT','statistic':None,'exact_permutation_p_value':None}
    ranks=np.vstack([np.argsort(np.argsort(row))+1 for row in matrix]).astype(int)
    sums=ranks.sum(axis=0)
    def stat(s): return float(12/(n*k*(k+1))*np.sum(np.asarray(s,float)**2)-3*n*(k+1))
    obs=stat(sums); perms=list(itertools.permutations(range(1,k+1))); states={(0,)*k:1}
    for _ in range(n):
        nxt=collections.defaultdict(int)
        for s,c in states.items():
            for p in perms: nxt[tuple(s[i]+p[i] for i in range(k))]+=c
        states=nxt
    tail=sum(c for s,c in states.items() if stat(s)>=obs-1e-12)
    total=len(perms)**n
    asym=friedmanchisquare(*[matrix[:,j] for j in range(k)])
    return {'status':'PASS','statistic':obs,'exact_permutation_p_value':float(tail/total),'exact_tail_count':int(tail),'n_total_permutations':int(total),'asymptotic_p_value':float(asym.pvalue)}

def validate_vits(root:Path):
    expected_steps={'learned':1600,'sinusoidal':200,'rope':200,'alibi':200}
    rows_attack=[]; rows_noise=[]; manifest=[]; clean_rows=[]
    split_hashes=set(); sample_hashes=set(); checkpoint_hashes=[]
    sessions=sorted(root.glob('session_*'))
    if len(sessions)!=6: raise ValueError(f'expected 6 sessions got {len(sessions)}')
    for sess in sessions:
        completion=json.loads((sess/'SESSION_COMPLETE.json').read_text())
        protocol=json.loads((sess/'SESSION_PROTOCOL.json').read_text())
        # minimum completeness gate; actual files checked below
        for p in sorted(sess.glob('*.json')):
            manifest.append({'path':str(p.relative_to(root)),'sha256':sha256(p),'bytes':p.stat().st_size})
        noise_path=sess/'noise_all_families.json'
        nd=json.loads(noise_path.read_text())
        nm=nd['metadata']; split_hashes.add(nm['split']['split_sha256']); sample_hashes.add(nm['split']['sample_order_sha256'])
        if nm['dataset']!='imagenet' or nm['n_total_images']!=5000: raise ValueError('bad vits dataset')
        if (nm['split']['n_calibration'],nm['split']['n_attack'],nm['split']['n_eval'])!=(256,1280,3464): raise ValueError('bad vits split')
        for field in ['calibration_attack_overlap','calibration_eval_overlap','attack_eval_overlap']:
            if nm['split'][field]!=0: raise ValueError('split overlap')
        for fam in FAMILIES:
            famd=nd['results'][fam]
            if len(famd)!=1: raise ValueError('noise family seed count')
            seed=int(next(iter(famd)))
            sr=famd[str(seed)]
            clean=float(sr['clean_acc_eval']); clean_rows.append({'architecture':'vits','family':fam,'seed':seed,'clean_accuracy':clean,'checkpoint_sha256':sr['checkpoint_sha256']})
            checkpoint_hashes.append(sr['checkpoint_sha256'])
            draws=sr['noise']['draws']
            if len(draws)!=10: raise ValueError('noise draws !=10')
            for draw in draws:
                for si,point in enumerate(draw['points'].values()):
                    b=float(point['budget'])
                    if b==0: continue
                    rows_noise.append({'architecture':'vits','family':fam,'seed':seed,'draw_seed':int(draw['seed']),'budget':b,'rho':float(point['rho']['rho_rel']),'accuracy':float(point['normalized_accuracy']),'source':str(noise_path.relative_to(root)),'source_index':si})
        for fam in FAMILIES:
            ap=sess/f'attacks_{fam}.json'; d=json.loads(ap.read_text()); m=d['metadata']
            split_hashes.add(m['split']['split_sha256']); sample_hashes.add(m['split']['sample_order_sha256'])
            if m['config']['pgd_steps']!=expected_steps[fam] or m['config']['pgd_restarts']!=5 or not np.isclose(m['config']['pgd_alpha_ratio'],.05): raise ValueError(f'bad attack config {sess.name}/{fam}')
            sr=d['results'][fam][str(int(next(iter(d['results'][fam]))))]; seed=int(next(iter(d['results'][fam])))
            if sr['checkpoint_sha256'] not in checkpoint_hashes: raise ValueError('checkpoint mismatch noise/attack')
            for si,point in enumerate(sr['attacks']['pgd_pe'].values()):
                b=float(point['budget'])
                if b==0: continue
                rr=point['restart_records']
                if len(rr)!=5: raise ValueError('restart count')
                selected=float(point['selected_attack_loss']); mx=max(float(x['attack_loss']) for x in rr)
                if not np.isclose(selected,mx,atol=1e-12,rtol=1e-12): raise ValueError('selected loss mismatch')
                rows_attack.append({'architecture':'vits','family':fam,'seed':seed,'budget':b,'rho':float(point['rho']['rho_rel']),'accuracy':float(point['normalized_accuracy']),'attack_loss':selected,'source':str(ap.relative_to(root)),'source_index':si})
    if len(split_hashes)!=1 or len(sample_hashes)!=1: raise ValueError('inconsistent split hashes')
    # noise budget pairing
    raw_noise=pd.DataFrame(rows_noise).drop_duplicates()
    dup=raw_noise.duplicated(['architecture','family','seed','draw_seed','budget'],keep=False)
    if dup.any(): raise ValueError('duplicate vits noise records')
    noise=(raw_noise.groupby(['architecture','family','seed','budget'],as_index=False)
           .agg(rho=('rho','mean'),accuracy=('accuracy','mean'),n_draws=('draw_seed','nunique'),rho_sd=('rho',lambda x:float(np.std(x,ddof=1))),accuracy_sd=('accuracy',lambda x:float(np.std(x,ddof=1))),sources=('source',lambda x:'|'.join(sorted(set(x)))))
           .sort_values(['family','seed','budget']))
    if not (noise.n_draws==10).all(): raise ValueError('vits noise aggregation incomplete')
    noise['source_index']=noise.groupby(['family','seed']).cumcount()
    attack=pd.DataFrame(rows_attack).sort_values(['family','seed','budget'])
    keys={(f,s) for f in FAMILIES for s in SEEDS}
    if set(map(tuple,noise[['family','seed']].drop_duplicates().to_numpy()))!=keys: raise ValueError('missing vits noise curve')
    if set(map(tuple,attack[['family','seed']].drop_duplicates().to_numpy()))!=keys: raise ValueError('missing vits attack curve')
    report={'status':'PASS','sessions':len(sessions),'attack_points':len(attack),'noise_raw_records':len(raw_noise),'noise_budget_points':len(noise),'split_sha256':next(iter(split_hashes)),'sample_order_sha256':next(iter(sample_hashes)),'manifest_files':len(manifest),'unique_checkpoint_hashes':len(set(checkpoint_hashes))}
    return attack,noise,pd.DataFrame(clean_rows).drop_duplicates(),pd.DataFrame(manifest),report

def load_vitb(v18_outputs:Path):
    adv=pd.read_csv(v18_outputs/'adversarial_native_budget_points_v18.csv')
    noise=pd.read_csv(v18_outputs/'noise_budget_paired_points_v18.csv')
    adv=adv[adv.dataset.eq('imagenet')].copy(); noise=noise[noise.dataset.eq('imagenet')].copy()
    adv=adv.rename(columns={'pe_family':'family'}) if 'pe_family' in adv.columns else adv
    noise=noise.rename(columns={'pe_family':'family'}) if 'pe_family' in noise.columns else noise
    # normalize fields expected by the new builder
    for df in [adv,noise]:
        df['architecture']='vitb'
        if 'accuracy' not in df.columns and 'normalized_accuracy' in df.columns: df['accuracy']=df['normalized_accuracy']
        df['source_index']=df.groupby(['family','seed']).cumcount()
    if 'attack_loss' not in adv.columns: adv['attack_loss']=np.nan
    clean=pd.DataFrame([{'architecture':'vitb','family':f,'seed':s,'clean_accuracy':np.nan,'checkpoint_sha256':''} for f in FAMILIES for s in SEEDS])
    return adv[['architecture','family','seed','budget','rho','accuracy','attack_loss','source_index']].copy(), noise[['architecture','family','seed','budget','rho','accuracy','source_index']].copy(), clean

def build(adv_all,noise_all,out:Path):
    curves={}; curve_rows=[]; support_rows=[]; rho50_rows=[]
    for arch in ['vitb','vits']:
      for fam in FAMILIES:
       for seed in SEEDS:
        na=noise_all.query('architecture==@arch and family==@fam and seed==@seed')
        aa=adv_all.query('architecture==@arch and family==@fam and seed==@seed')
        xn,yn=construct_curve(na,'noise',False); xa,ya=construct_curve(aa,'adversarial',True); xr,yr=construct_curve(aa,'adversarial',False)
        curves[(arch,fam,seed,'noise')]=(xn,yn); curves[(arch,fam,seed,'adversarial')]=(xa,ya); curves[(arch,fam,seed,'adversarial_raw')]=(xr,yr)
        for regime,x,y,env in [('noise',xn,yn,False),('adversarial',xa,ya,True),('adversarial_raw',xr,yr,False)]:
            support_rows.append({'architecture':arch,'family':fam,'seed':seed,'regime':regime,'max_rho':float(x[-1]),'n_points':len(x)})
            if regime!='adversarial_raw':
                tc=threshold_crossing(x,y,.5); rho50_rows.append({'architecture':arch,'family':fam,'seed':seed,'regime':regime,**tc})
            for i,(xx,yy) in enumerate(zip(x,y)):
                curve_rows.append({'architecture':arch,'family':fam,'seed':seed,'regime':regime,'point_index':i,'rho':float(xx),'normalized_accuracy':float(yy),'clean_anchor':i==0,'lower_envelope':env})
    supports=pd.DataFrame(support_rows)
    rho_common_arch={a:float(supports.query("architecture==@a and regime!='adversarial_raw'").max_rho.min()) for a in ['vitb','vits']}
    rho_common_cross=float(supports.query("regime!='adversarial_raw'").max_rho.min())
    rho_common_family={f:float(supports.query("family==@f and regime!='adversarial_raw'").max_rho.min()) for f in FAMILIES}
    endpoints=[]
    for scope,end in [('cross',rho_common_cross),('vitb',rho_common_arch['vitb']),('vits',rho_common_arch['vits'])]:
        for mult in [1.,.9,.75]: endpoints.append((scope,mult,end*mult))
    seed_rows=[]
    for arch in ['vitb','vits']:
     for fam in FAMILIES:
      for seed in SEEDS:
       for scope,mult,end in endpoints:
        if scope not in ('cross',arch): continue
        xn,yn=curves[(arch,fam,seed,'noise')]; xa,ya=curves[(arch,fam,seed,'adversarial')]; xr,yr=curves[(arch,fam,seed,'adversarial_raw')]
        nn=integrate_nauc(xn,yn,end); aa=integrate_nauc(xa,ya,end); rr=integrate_nauc(xr,yr,end)
        seed_rows.append({'scope':scope,'endpoint_multiplier':mult,'endpoint':end,'architecture':arch,'family':fam,'seed':seed,'noise_nauc':nn,'adversarial_nauc':aa,'adversarial_raw_nauc':rr,'gap':nn-aa,'raw_gap':nn-rr})
    seed_df=pd.DataFrame(seed_rows)
    agg=[]
    for keys,g in seed_df.groupby(['scope','endpoint_multiplier','endpoint','architecture','family']):
        scope,mult,end,arch,fam=keys
        for metric in ['noise_nauc','adversarial_nauc','gap']:
            vals=g.sort_values('seed')[metric].to_numpy(float); lo,hi=paired_bootstrap_ci(vals,BOOTSTRAP_SEED+len(agg))
            agg.append({'scope':scope,'endpoint_multiplier':mult,'endpoint':end,'architecture':arch,'family':fam,'metric':metric,'n_seeds':len(vals),'mean':float(vals.mean()),'sd':float(vals.std(ddof=1)),'ci95_low':lo,'ci95_high':hi,'positive_seeds':int((vals>0).sum()),'exact_sign_flip_p':exact_sign_flip(vals)})
    agg_df=pd.DataFrame(agg)
    contrasts=[]
    cross=seed_df.query("scope=='cross'")
    for mult in [1.,.9,.75]:
      for fam in FAMILIES:
       b=cross.query('endpoint_multiplier==@mult and architecture=="vitb" and family==@fam').sort_values('seed')
       s=cross.query('endpoint_multiplier==@mult and architecture=="vits" and family==@fam').sort_values('seed')
       for metric in ['noise_nauc','adversarial_nauc','gap']:
        vals=s[metric].to_numpy()-b[metric].to_numpy(); lo,hi=paired_bootstrap_ci(vals,BOOTSTRAP_SEED+1000+len(contrasts))
        contrasts.append({'endpoint_multiplier':mult,'endpoint':rho_common_cross*mult,'family':fam,'metric':metric,'contrast':'vits_minus_vitb','mean':float(vals.mean()),'sd':float(vals.std(ddof=1)),'ci95_low':lo,'ci95_high':hi,'positive_seeds':int((vals>0).sum()),'exact_sign_flip_p':exact_sign_flip(vals)})
    contrast_df=pd.DataFrame(contrasts)
    friedman={}
    for arch in ['vitb','vits']:
      for mult in [1.,.9,.75]:
       piv=cross.query('architecture==@arch and endpoint_multiplier==@mult').pivot(index='seed',columns='family',values='gap').reindex(index=SEEDS,columns=FAMILIES)
       friedman[f'{arch}_gap_{mult:g}']=exact_friedman(piv.to_numpy())
    # ranks
    ranks=[]
    for arch in ['vitb','vits']:
      for mult in [1.,.9,.75]:
       sub=cross.query('architecture==@arch and endpoint_multiplier==@mult').groupby('family',as_index=False)[['noise_nauc','adversarial_nauc','gap']].mean()
       for metric in ['noise_nauc','adversarial_nauc']:
        tmp=sub.sort_values(metric,ascending=False)
        for rank,(_,r) in enumerate(tmp.iterrows(),1): ranks.append({'architecture':arch,'endpoint_multiplier':mult,'metric':metric,'rank':rank,'family':r.family,'mean':r[metric]})
       tmp=sub.sort_values('gap',ascending=True) # smaller gap = more coupling / less adversarial deficit
       for rank,(_,r) in enumerate(tmp.iterrows(),1): ranks.append({'architecture':arch,'endpoint_multiplier':mult,'metric':'gap_smallest_first','rank':rank,'family':r.family,'mean':r['gap']})
    rank_df=pd.DataFrame(ranks)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(out/'v19_imagenet_curves_long.csv',index=False)
    supports.to_csv(out/'v19_curve_support_maxima.csv',index=False)
    seed_df.to_csv(out/'v19_support_aware_seed_level_nauc.csv',index=False)
    agg_df.to_csv(out/'v19_support_aware_aggregate_nauc.csv',index=False)
    contrast_df.to_csv(out/'v19_cross_architecture_contrasts.csv',index=False)
    pd.DataFrame(rho50_rows).to_csv(out/'v19_rho50_thresholds.csv',index=False)
    rank_df.to_csv(out/'v19_family_rankings.csv',index=False)
    report={'version':'CANONICAL v19','status':'PASS','policy':{'tau_abs':TAU_ABS,'tau_rel':TAU_REL,'canonical_only':True,'D019_D020_support_extension':False,'noise_aggregation':'budget-paired mean over 10 fixed directions','adversarial_selection':'loss-selected best of 5 at each native budget','adversarial_envelope':'cumulative minimum after achieved-rho ordering','no_extrapolation':True},'rho_common':{'vitb':rho_common_arch['vitb'],'vits':rho_common_arch['vits'],'cross_architecture':rho_common_cross,'family_cross_architecture':rho_common_family},'n_curves_canonical':96,'friedman':friedman}
    (out/'V19_SUPPORT_AWARE_BUILD_REPORT.json').write_text(json.dumps(report,indent=2))
    return report,seed_df,agg_df,contrast_df,pd.DataFrame(curve_rows),supports,pd.DataFrame(rho50_rows)

def write_tables(out,report,agg,contrast):
    end=report['rho_common']['cross_architecture']
    primary=agg.query("scope=='cross' and endpoint_multiplier==1.0")
    lines=[
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        f'\\caption{{Cross-architecture ImageNet-100 normalized AUC over the locked common-support interval $\\rho_{{\\mathrm{{rel}}}}\\in[0,{end:.5f}]$. Values are mean $\\pm$ sample SD across the same six training seeds. The primary estimand uses the prespecified canonical native-budget grids; post-lock task-loss points and direct-displacement points are reported separately.}}',
        '\\label{tab:v19-cross-architecture-nauc}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{llccc}',
        '\\toprule',
        'Architecture & PE family & Noise nAUC & Adversarial nAUC & Gap \\\\',
        '\\midrule',
    ]
    for ai,arch in enumerate(['vitb','vits']):
        for fi,fam in enumerate(FAMILIES):
            def val(metric):
                r=primary[(primary['architecture']==arch)&(primary['family']==fam)&(primary['metric']==metric)].iloc[0]
                return f'${r["mean"]:.4f} \\pm {r["sd"]:.4f}$'
            al=ARCH_LABELS[arch] if fi==0 else ''
            lines.append(f'{al} & {FAMILY_LABELS[fam]} & {val("noise_nauc")} & {val("adversarial_nauc")} & {val("gap")} \\\\')
        if ai==0:
            lines.append('\\midrule')
    lines += ['\\bottomrule','\\end{tabular}%','}','\\end{table*}','']
    (out/'table_v19_cross_architecture_nauc.tex').write_text('\n'.join(lines), encoding='utf-8')

    c=contrast.query('endpoint_multiplier==1.0')
    lines=[
        '\\begin{table}[t]',
        '\\centering',
        '\\small',
        f'\\caption{{Seed-aligned descriptive ViT-S/16 minus ViT-B/16 nAUC contrasts over $\\rho_{{\\mathrm{{rel}}}}\\in[0,{end:.5f}]$. Positive values favor ViT-S/16; equal seed labels are not strict training pairs.}}',
        '\\label{tab:v19-architecture-contrasts}',
        '\\begin{tabular}{lrrr}',
        '\\toprule',
        'PE family & Noise & Adversarial & Gap \\\\',
        '\\midrule',
    ]
    for fam in FAMILIES:
        vals=[]
        for metric in ['noise_nauc','adversarial_nauc','gap']:
            r=c[(c['family']==fam)&(c['metric']==metric)].iloc[0]
            vals.append(f'${r["mean"]:+.4f} \\pm {r["sd"]:.4f}$')
        lines.append(f'{FAMILY_LABELS[fam]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\')
    lines += ['\\bottomrule','\\end{tabular}','\\end{table}','']
    (out/'table_v19_architecture_contrasts.tex').write_text('\n'.join(lines), encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v18-outputs',required=True); ap.add_argument('--vits-root',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); out=Path(a.output_dir)
    va,vn,vc,manifest,vreport=validate_vits(Path(a.vits_root)); ba,bn,bc=load_vitb(Path(a.v18_outputs))
    adv=pd.concat([ba,va],ignore_index=True); noise=pd.concat([bn,vn],ignore_index=True)
    report,seed,agg,contrast,curves,supports,rho50=build(adv,noise,out)
    vc.to_csv(out/'v19_vits_clean_accuracy.csv',index=False); manifest.to_csv(out/'v19_vits_source_manifest.csv',index=False); (out/'V19_VITS_INPUT_INTEGRITY_REPORT.json').write_text(json.dumps(vreport,indent=2))
    write_tables(out,report,agg,contrast)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
