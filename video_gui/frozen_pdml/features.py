"""Shared offline feature extraction for CARE-PD and video. No learned parameters here."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter,resample_poly
from .common import ARTIFACTS,dump_json,sha256
from .protocol import identity

ANGLES=['arm_L','arm_R','elbow_L','elbow_R','thigh_L','thigh_R','knee_L','knee_R']
NAMES=ANGLES+[x+'_velocity' for x in ANGLES]+['wrist_forward_L','wrist_forward_R','ankle_forward_L','ankle_forward_R','ankle_separation','shoulder_pelvis_twist']
REQUIRED=[1,2,3,4,5,6,11,12,13,14,15,16]
SIDES={'L':(4,5,6,11,12,13),'R':(1,2,3,14,15,16)}

def columns(feature_set):return list(range({'F1':8,'F2':16,'F3':22}[feature_set]))
def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
def unit(a):
    n=np.linalg.norm(a,axis=-1,keepdims=True)
    return np.divide(a,n,out=np.full_like(a,np.nan),where=n>1e-8)

def resample_skeleton(seq,times,cfg):
    seq=np.asarray(seq,float).copy();times=np.asarray(times,float)
    if seq.ndim!=3 or seq.shape[1:]!=(17,3) or len(times)!=len(seq):raise ValueError('Expected (T,17,3) and T timestamps')
    if len(seq)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid timestamps')
    times=times-times[0]
    # Treat any missing required coordinate as invalid frame; interpolate only short interior gaps.
    valid=np.isfinite(seq[:,REQUIRED]).all(axis=(1,2));original=valid.copy()
    for start,end in runs(~valid):
        if start>0 and end<len(seq) and times[end]-times[start-1] <= cfg.max_gap_s + np.median(np.diff(times))+1e-7:
            for j in REQUIRED:
                for k in range(3):seq[start:end,j,k]=np.interp(times[start:end],times[[start-1,end]],seq[[start-1,end],j,k])
            valid[start:end]=True
    grid=np.arange(int(np.floor(times[-1]*cfg.hz+1e-6))+1)/cfg.hz
    result=np.full((len(grid),17,3),np.nan);quality=np.ones(len(grid),bool)
    for start,end in runs(valid):
        if end-start<2:continue
        t=times[start:end];a=seq[start:end];idx=np.flatnonzero((grid>=t[0]-1e-8)&(grid<=t[-1]+1e-8))
        if not len(idx):continue
        # Polyphase anti-aliasing for regular high-rate runs. Irregular runs first use a uniform native grid.
        native=1/np.median(np.diff(t))
        if native>cfg.hz*1.01:
            native_t=t[0]+np.arange(int(np.floor((t[-1]-t[0])*native))+1)/native
            flat=a.reshape(len(a),-1)
            uniform=np.stack([np.interp(native_t,t,flat[:,k]) for k in range(flat.shape[1])],axis=1)
            ratio=Fraction(cfg.hz/native).limit_denominator(1000)
            filtered=resample_poly(uniform,ratio.numerator,ratio.denominator,axis=0,padtype='line')
            ft=t[0]+np.arange(len(filtered))/(native*ratio.numerator/ratio.denominator)
            flat_out=np.stack([np.interp(grid[idx],ft,filtered[:,k]) for k in range(filtered.shape[1])],axis=1)
        else:
            flat=a.reshape(len(a),-1);flat_out=np.stack([np.interp(grid[idx],t,flat[:,k]) for k in range(flat.shape[1])],axis=1)
        result[idx]=flat_out.reshape(-1,17,3)
        # Conservatively mark resampled frames influenced by a missing raw frame.
        right=np.clip(np.searchsorted(times,grid[idx]),0,len(times)-1);left=np.maximum(right-1,0)
        quality[idx]=~(original[left]&original[right])
    return result,grid,quality

def kinematic_signals(seq,cfg):
    p=(seq[:,1]+seq[:,4])/2;s=(seq[:,11]+seq[:,14])/2
    up=unit(s-p);lateral=seq[:,4]-seq[:,1]
    lateral=unit(lateral-(lateral*up).sum(1)[:,None]*up)
    forward=unit(np.cross(lateral,up));down=-up
    def signed(v):return np.degrees(np.arctan2((v*forward).sum(1),(v*down).sum(1)))
    def flex(a,b,c):
        cosine=(unit(a-b)*unit(c-b)).sum(1)
        return 180-np.degrees(np.arccos(np.clip(cosine,-1,1)))
    values={};lengths=[]
    for side,(hip,knee,ankle,shoulder,elbow,wrist) in SIDES.items():
        values['arm_'+side]=signed(seq[:,elbow]-seq[:,shoulder])
        values['elbow_'+side]=flex(seq[:,shoulder],seq[:,elbow],seq[:,wrist])
        values['thigh_'+side]=signed(seq[:,knee]-seq[:,hip])
        values['knee_'+side]=flex(seq[:,hip],seq[:,knee],seq[:,ankle])
        lengths.append(np.linalg.norm(seq[:,hip]-seq[:,knee],axis=1)+np.linalg.norm(seq[:,knee]-seq[:,ankle],axis=1))
    length_series=np.mean(lengths,axis=0)
    if not np.isfinite(length_series).any():raise ValueError('Invalid leg scale: no finite required joints')
    scale=np.nanmedian(length_series)
    if not np.isfinite(scale) or scale<=1e-8:raise ValueError('Invalid leg scale')
    extras=[]
    for landmark in [13,16,6,3]:extras.append(((seq[:,landmark]-p)*forward).sum(1)/scale)
    extras.append(np.linalg.norm(seq[:,6]-seq[:,3],axis=1)/scale)
    shoulder_axis=seq[:,11]-seq[:,14];shoulder_axis=unit(shoulder_axis-(shoulder_axis*up).sum(1)[:,None]*up)
    twist=np.degrees(np.arctan2((np.cross(lateral,shoulder_axis)*up).sum(1),(lateral*shoulder_axis).sum(1)))
    raw=np.column_stack([values[n] for n in ANGLES]+extras+[twist])
    output=np.full((len(seq),22),np.nan)
    for start,end in runs(np.isfinite(raw).all(1)):
        if end-start<cfg.smooth_window:continue
        a=raw[start:end].copy()
        for k in [0,1,4,5,13]:
            a[:,k]=np.degrees(np.unwrap(np.radians(a[:,k])))
            # Preserve neutral orientation while preventing arbitrary full-turn offsets.
            a[:,k]-=360*round(float(np.median(a[:,k]))/360)
        sm=savgol_filter(a,cfg.smooth_window,cfg.smooth_poly,axis=0,mode='interp')
        vel=savgol_filter(a[:,:8],cfg.smooth_window,cfg.smooth_poly,deriv=1,delta=1/cfg.hz,axis=0,mode='interp')
        output[start:end]=np.column_stack([sm[:,:8],vel,sm[:,8:]])
    return output

def extract_windows(seq,times,cfg):
    seq,t,imputed=resample_skeleton(seq,times,cfg)
    signal=kinematic_signals(seq,cfg);windows=[];accepted=[];rejected=[]
    for start in range(0,len(signal)-cfg.window+1,cfg.stride):
        end=start+cfg.window;a=signal[start:end];fraction=float(imputed[start:end].mean())
        reason='nonfinite_or_geometry' if not np.isfinite(a).all() else ('too_much_interpolation' if fraction>cfg.max_missing else '')
        row=dict(start_frame=start,end_frame=end,start_s=float(t[start]),interpolated_fraction=fraction)
        if reason:rejected.append(dict(row,reason=reason))
        else:windows.append(a.astype('float32'));accepted.append(row)
    if not accepted and not rejected:rejected.append(dict(start_frame=0,end_frame=len(signal),reason='too_short'))
    return np.stack(windows) if windows else np.empty((0,cfg.window,22),np.float32),accepted,rejected,signal

def build_features(audit_dir,cfg,base=None):
    audit_dir=Path(audit_dir);source=audit_dir/'skeleton_manifest.csv';manifest=pd.read_csv(source)
    cache_id=identity(dict(manifest=sha256(source),config=cfg.dict(),code=sha256(__file__)))
    out=Path(base or ARTIFACTS/'features')/cache_id
    if (out/'complete.json').exists():return out
    out.mkdir(parents=True,exist_ok=True);all_x=[];meta=[];reject=[];clips=[]
    with np.load(manifest.npz_path.iloc[0],allow_pickle=False) as z:
        for r in manifest.to_dict('records'):
            try:
                seq=z[r['sequence_key']];x,accepted,bad,signal=extract_windows(seq,np.arange(len(seq))/r['effective_fps'],cfg)
                signal_dir=out/'signals';signal_dir.mkdir(exist_ok=True)
                key=identity(r['sample_id']);np.savez_compressed(signal_dir/f'{key}.npz',signals=signal)
                for a,w in zip(accepted,x):meta.append(dict(r,**a,window_id=f'{r["sample_id"]}::{a["start_frame"]}'));all_x.append(w)
                for b in bad:reject.append(dict(sample_id=r['sample_id'],label=r['label'],**b))
                clips.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],label=r['label'],windows=len(accepted),rejected=len(bad),signal_file=f'signals/{key}.npz'))
            except (ValueError,FloatingPointError) as e:
                reject.append(dict(sample_id=r['sample_id'],label=r['label'],reason=str(e)))
                clips.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],label=r['label'],windows=0,rejected=1,signal_file=''))
    if not all_x:raise ValueError('No valid windows; inspect quality first')
    np.save(out/'windows.npy',np.stack(all_x));pd.DataFrame(meta).to_csv(out/'windows.csv',index=False)
    pd.DataFrame(reject,columns=['sample_id','label','start_frame','end_frame','reason','interpolated_fraction']).to_csv(out/'exclusions.csv',index=False)
    pd.DataFrame(clips).to_csv(out/'coverage.csv',index=False)
    dump_json(out/'protocol.json',cfg.dict());dump_json(out/'feature_schema.json',dict(version='2.1',names=NAMES,sets={s:[NAMES[i] for i in columns(s)] for s in ['F1','F2','F3']},units=['degree']*8+['degree/second']*8+['leg_length_ratio']*5+['degree']))
    dump_json(out/'complete.json',dict(cache_id=cache_id,audit_dir=str(audit_dir),windows=len(meta),clips=len(clips),window_sha256=sha256(out/'windows.npy'),metadata_sha256=sha256(out/'windows.csv')))
    return out

def load_features(path):
    path=Path(path);info=json.loads((path/'complete.json').read_text())
    for filename,key in [('windows.npy','window_sha256'),('windows.csv','metadata_sha256')]:
        if sha256(path/filename)!=info[key]:raise ValueError('Feature cache modified: '+filename)
    return np.load(path/'windows.npy',mmap_mode='r'),pd.read_csv(path/'windows.csv')

def summarize(x,feature_set):
    ids=columns(feature_set);a=np.asarray(x)[:,:,ids];names=[NAMES[i] for i in ids]
    blocks=[a.mean(1),a.std(1),np.median(a,1),np.percentile(a,5,axis=1),np.percentile(a,95,axis=1),np.sqrt((a*a).mean(1))]
    labels=[n+'__'+stat for stat in ['mean','std','median','p05','p95','rms'] for n in names]
    corr=[]
    for row in a:
        left=row[:-1]-row[:-1].mean(0);right=row[1:]-row[1:].mean(0);den=np.sqrt((left*left).sum(0)*(right*right).sum(0))
        corr.append(np.divide((left*right).sum(0),den,out=np.full(row.shape[1],np.nan),where=den>1e-10))
    blocks.append(np.array(corr));labels += [n+'__lag1' for n in names]
    pairs=[(0,1),(4,5),(6,7),(0,5),(1,4)];pair_values=[]
    for row in a:
        vals=[]
        for l,r in pairs:
            options=[]
            for lag in range(-min(15,len(row)//3),min(15,len(row)//3)+1):
                u=row[max(0,lag):min(len(row),len(row)+lag),l];v=row[max(0,-lag):min(len(row),len(row)-lag),r]
                if np.std(u)>1e-8 and np.std(v)>1e-8:options.append((float(np.corrcoef(u,v)[0,1]),lag/30))
            vals.extend(max(options,key=lambda t:abs(t[0])) if options else [np.nan,np.nan])
        pair_values.append(vals)
    blocks.append(np.array(pair_values));labels += [f'{NAMES[l]}_{NAMES[r]}__{k}' for l,r in pairs for k in ['xcorr','lag_s']]
    return np.column_stack(blocks),labels
