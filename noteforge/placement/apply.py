"""Sample a learned layout for a rhythm draft and compare jumps with humans and the rule layout."""
import json,sys,zipfile
import numpy as np, torch, yaml
_here=__import__('pathlib').Path(__file__).resolve().parent; sys.path[:0]=[str(_here.parent),str(_here.parent/'scripts'),str(_here)]
from pathlib import Path
from noteforge.placement.model import M, feats, BINS, NB
from noteforge.events.phira import parse_phira
from noteforge.events.phira_postprocess import clean_types, limit_fingers
from noteforge.events.phira_rpe import plain_background, rpe
KIND={'tap':0,'hold':1,'flick':2,'drag':3}
def layout(ev,net,T=.8,seed=0):
    ev=sorted(ev,key=lambda e:(e['onset_us'],e['type']!='hold'))
    if not ev:
        return []
    a=np.array([(e['onset_us']/1e6,e['end_us']/1e6,KIND[e['type']],0.) for e in ev],np.float32).reshape(-1,4)
    f=torch.as_tensor(feats(a))[None]
    g=torch.Generator().manual_seed(seed); st=None; prev=NB; xs=[]; near=[]
    c=(BINS[:-1]+BINS[1:])/2
    with torch.no_grad():
        for i in range(len(ev)):
            lg,st=net(f[:,i:i+1],torch.tensor([[prev]]),st); lg=lg[0,0]/T
            t=a[i,0]
            # two notes struck together cannot share a spot
            mask=[abs(c[j]-x) < 135 for j in range(NB) for x in [0]] and None
            blocked=[x for tt,x in near if t-tt<.03]
            if blocked:
                for x in blocked: lg[torch.as_tensor(np.abs(c-x)<180)]-=50
            b=torch.multinomial(lg.softmax(-1),1,generator=g).item(); prev=b
            x=float(c[b]+torch.empty(1).uniform_(-20,20,generator=g).item())
            xs.append(x); near=[(tt,xx) for tt,xx in near if t-tt<.03]+[(t,x)]
    return [dict(id=f'event:{i}',t_us=int(e['onset_us']),end_us=int(e['end_us']),type=e['type'],x=round(x,1)) for i,(e,x) in enumerate(zip(ev,xs))]
def stats(L):
    Ls=sorted(L,key=lambda p:p['t_us']); x=np.array([p['x'] for p in Ls]); t=np.array([p['t_us'] for p in Ls],float); dt=np.diff(t)/1e6; dx=np.abs(np.diff(x))
    return 'x sd %.0f jump median %.0f '%(x.std(),np.median(dx))+str({f'{a}-{b}':(int(np.median(dx[(dt>=a)&(dt<b)])),round(float(np.mean(dx[(dt>=a)&(dt<b)]>600)),3),round(float(np.mean(dx[(dt>=a)&(dt<b)]<30)),2)) for a,b in((.01,.1),(.1,.2),(.2,.35),(.35,.6),(.6,2)) if ((dt>=a)&(dt<b)).sum()})
