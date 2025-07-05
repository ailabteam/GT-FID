#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_dl.py – Deep-learning baselines (GRU, CNN-LSTM, Transformer) for host-based IDS
ChatGPT • July 2025
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
)

# -------------------------------------------------- Utils
def set_seed(sd: int = 42) -> None:
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(y, p, thr=0.5):
    y_hat = (p >= thr).astype(int)
    return dict(
        acc  = accuracy_score(y, y_hat),
        precision = precision_score(y, y_hat, zero_division=0),
        recall    = recall_score(y, y_hat, zero_division=0),
        f1        = f1_score(y, y_hat, zero_division=0),
        auroc     = roc_auc_score(y, p),
    )

# -------------------------------------------------- Dataset
class SysDataset(Dataset):
    def __init__(self, path, seq_field="sequence", sliding=None):
        self.samples=[]
        for ln in open(path):
            j=json.loads(ln)
            seq=j.get(seq_field) or j.get("sequence") or j.get("seq")
            if seq is None: continue
            lab=j["label"]
            if sliding:
                win,stride=sliding
                for i in range(0,max(1,len(seq)-win+1),stride):
                    sub=seq[i:i+win]
                    if len(sub)<win: sub+= [0]*(win-len(sub))
                    self.samples.append((sub,lab))
            else: self.samples.append((seq,lab))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s,l=self.samples[i]
        return torch.tensor(s,dtype=torch.long),l

def collate(batch):
    seqs,labels=zip(*batch)
    lens=torch.tensor([len(x) for x in seqs])
    seqs=pad_sequence(seqs,batch_first=True)
    return seqs,lens,torch.tensor(labels,dtype=torch.float32)

def loader(ds,bs,workers,sampler=None,shuffle=False):
    return DataLoader(ds,bs,shuffle if sampler is None else False,
                      sampler=sampler,num_workers=workers,
                      collate_fn=collate,pin_memory=torch.cuda.is_available())

# -------------------------------------------------- Models
class GRU(nn.Module):
    def __init__(self,vocab,embed=128,hidden=256,layers=2,dp=0.3):
        super().__init__()
        self.emb=nn.Embedding(len(vocab),embed,padding_idx=0)

        self.gru = nn.GRU(
    input_size=embed,
    hidden_size=hidden,
    num_layers=layers,
    bidirectional=True,
    batch_first=True,
    dropout=dp if layers > 1 else 0,
)


        self.dp=nn.Dropout(dp)
        self.out=nn.Linear(hidden*2,1)
    def forward(self,x,l):
        x=self.emb(x)
        pack=pack_padded_sequence(x,l.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(pack)
        h=torch.cat([h[-2],h[-1]],1)
        return self.out(self.dp(h)).squeeze(1)

class CNNLSTM(nn.Module):
    def __init__(self,vocab,filters=(256,256,256),ks=(3,5,7),embed=128,hidden=256,dp=0.3,cnn_dp=0.2):
        super().__init__()
        self.emb=nn.Embedding(len(vocab),embed,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(embed,f,k,padding=k//2) for f,k in zip(filters,ks)])
        self.cdp=nn.Dropout(cnn_dp)
        self.lstm=nn.LSTM(sum(filters),hidden,bidirectional=True,batch_first=True)
        self.dp=nn.Dropout(dp); self.out=nn.Linear(hidden*2,1)
    def forward(self,x,l):
        x=self.emb(x).transpose(1,2)
        x=torch.cat([F.relu(c(x)) for c in self.convs],1)
        x=self.cdp(x).transpose(1,2)
        pack=pack_padded_sequence(x,l.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_) = self.lstm(pack)
        h=torch.cat([h[-2],h[-1]],1)
        return self.out(self.dp(h)).squeeze(1)

class PosEnc(nn.Module):
    def __init__(s,d,ml=5000):
        super().__init__()
        pe=torch.zeros(ml,d); pos=torch.arange(ml).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2)*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        s.register_buffer("pe",pe.unsqueeze(0))
    def forward(s,x): return x+s.pe[:,:x.size(1)]

class Transformer(nn.Module):
    def __init__(self,vocab,hidden=256,layers=4,heads=4,dp=0.1,max_len=512):
        super().__init__()
        self.emb=nn.Embedding(len(vocab),hidden,padding_idx=0)
        self.pos=PosEnc(hidden,max_len)
        enc=nn.TransformerEncoderLayer(hidden,heads,hidden*4,dp,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.dp=nn.Dropout(dp); self.out=nn.Linear(hidden,1)
    def forward(self,x,l):
        mask=(x==0)
        h=self.enc(self.pos(self.emb(x)),src_key_padding_mask=mask)
        h=h.masked_fill(mask.unsqueeze(-1),0).sum(1)/l.clamp(min=1).unsqueeze(1)
        return self.out(self.dp(h)).squeeze(1)

# -------------------------------------------------- Train / evaluate
def run(loader,model,crit,opt=None,device="cpu"):
    train=opt is not None
    model.train() if train else model.eval()
    Ys,Ps,loss_tot= [],[],0
    for seq,lens,lab in loader:
        seq,lens,lab=seq.to(device),lens.to(device),lab.to(device)
        logit=model(seq,lens); loss=crit(logit,lab)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_tot+=loss.item()*len(lab)
        Ys.extend(lab.cpu()); Ps.extend(torch.sigmoid(logit).detach().cpu())
    Ys,Ps=np.array(Ys),np.array(Ps)
    return loss_tot/len(loader.dataset), metrics(Ys,Ps)

def best_threshold(val_loader,model,device):
    y,p=[],[]
    with torch.no_grad():
        for s,l,lab in val_loader:
            s,l=s.to(device),l.to(device)
            p.extend(torch.sigmoid(model(s,l)).cpu()); y.extend(lab)
    y,p=np.array(y),np.array(p)
    prec,rec,thr=precision_recall_curve(y,p)
    f1=2*prec*rec/(prec+rec+1e-8)
    idx=f1.argmax(); return thr[idx]

# -------------------------------------------------- Main
def main():
    pa=argparse.ArgumentParser("Deep baseline trainer/evaluator")
    pa.add_argument("--data",required=True); pa.add_argument("--vocab",required=True)
    pa.add_argument("--seq_field",default="sequence")
    pa.add_argument("--model",choices=["gru","cnn_lstm","transformer"],default="gru")
    pa.add_argument("--hidden",type=int,default=256); pa.add_argument("--layers",type=int,default=2)
    pa.add_argument("--heads",type=int,default=4);   pa.add_argument("--dropout",type=float,default=0.3)
    pa.add_argument("--filters",nargs="+",type=int,default=[256,256,256])
    pa.add_argument("--kernel_sizes",nargs="+",type=int,default=[3,5,7])
    pa.add_argument("--cnn_dropout",type=float,default=0.2)
    pa.add_argument("--max_len",type=int,default=512); pa.add_argument("--attn_dropout",type=float,default=0.1)
    pa.add_argument("--batch_size",type=int,default=64); pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--lr",type=float,default=1e-3)
    pa.add_argument("--pos_weight",type=float); pa.add_argument("--sampler",choices=["none","weighted"],default="none")
    pa.add_argument("--scheduler",choices=["none","step","cosine"],default="none")
    pa.add_argument("--step_size",type=int,default=10); pa.add_argument("--gamma",type=float,default=0.5)
    pa.add_argument("--warmup",type=int,default=0)
    pa.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--num_workers",type=int,default=0)
    pa.add_argument("--stratify",action="store_true"); pa.add_argument("--test_split",type=float,default=0.15)
    pa.add_argument("--sliding",nargs=2,type=int)
    pa.add_argument("--seed",type=int,default=42)
    # extra utilities
    pa.add_argument("--load"); pa.add_argument("--find_threshold",action="store_true")
    args=pa.parse_args(); set_seed(args.seed); dev=torch.device(args.device)

    full=SysDataset(args.data,args.seq_field,tuple(args.sliding) if args.sliding else None)
    idx=np.arange(len(full)); labels=[l for _,l in full]
    from sklearn.model_selection import train_test_split
    train_idx,test_idx=train_test_split(idx,test_size=args.test_split,random_state=args.seed,
                                        stratify=labels if args.stratify else None)
    train_idx,val_idx=train_test_split(train_idx,test_size=args.test_split,random_state=args.seed,
                                       stratify=np.array(labels)[train_idx] if args.stratify else None)
    tr,va,te=[torch.utils.data.Subset(full,i) for i in (train_idx,val_idx,test_idx)]

    sampler=None
    if args.sampler=="weighted":
        freq=Counter([l for _,l in tr]); w=[1/freq[l] for _,l in tr]
        sampler=WeightedRandomSampler(w,len(w),True)
    L=loader; trainL=L(tr,args.batch_size,args.num_workers,sampler,shuffle=True)
    valL  =L(va,args.batch_size,args.num_workers); testL=L(te,args.batch_size,args.num_workers)

    vocab=json.load(open(args.vocab))
    kw=dict(vocab=vocab,hidden=args.hidden)
    if args.model=="gru":
        net=GRU(**kw,layers=args.layers,dp=args.dropout).to(dev)
    elif args.model=="cnn_lstm":
        net=CNNLSTM(**kw,filters=args.filters,kernel_sizes=args.kernel_sizes,
                    embed=args.hidden//2,hidden=args.hidden,dp=args.dropout,cnn_dp=args.cnn_dropout).to(dev)
    else:
        net=Transformer(**kw,layers=args.layers,heads=args.heads,dp=args.dropout,max_len=args.max_len).to(dev)

    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    if args.scheduler=="step":
        sched=torch.optim.lr_scheduler.StepLR(opt,args.step_size,args.gamma)
    elif args.scheduler=="cosine":
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs-args.warmup)
    else: sched=None

    pos_w=torch.tensor(args.pos_weight,device=dev) if args.pos_weight else None
    lossFn=nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ck=Path("checkpoints"); ck.mkdir(exist_ok=True)
    if args.load:
        net.load_state_dict(torch.load(args.load,map_location=dev))
    else:
        best=0
        for ep in range(1,args.epochs+1):
            tr_loss,_=run(trainL,net,lossFn,opt,dev)
            if sched:
                if args.scheduler=="cosine" and ep<=args.warmup:
                    for g in opt.param_groups:g["lr"]=args.lr*ep/args.warmup
                else:sched.step()
            _,mv=run(valL,net,lossFn,None,dev)
            print(f"Epoch {ep:03d}: train-loss {tr_loss:.4f} | val-f1 {mv['f1']:.3f} best {best:.3f}")
            if mv["f1"]>best:
                best=mv["f1"]; torch.save(net.state_dict(),ck/f"{args.model}.pt")

        net.load_state_dict(torch.load(ck/f"{args.model}.pt",map_location=dev))

    # ---- Threshold tuning ----
    thr=0.5
    if args.find_threshold:
        thr=best_threshold(valL,net,dev)
        print(f"★ Optimal threshold on val = {thr:.2f}")

    _,mt=run(testL,net,lossFn,None,dev)
    mtThr=metrics(np.array([l for _,l in te]),
                  np.concatenate([
    torch.sigmoid(net(s.to(dev), l.to(dev))).detach().cpu().numpy()
    for s, l, _ in testL
]),thr)
    print("Test:",
          f"acc {mtThr['acc']:.3f}",
          f"precision {mtThr['precision']:.3f}",
          f"recall {mtThr['recall']:.3f}",
          f"f1 {mtThr['f1']:.3f}",
          f"auroc {mtThr['auroc']:.3f}")
    cm=confusion_matrix([l for _,l in te],
                        (np.concatenate([
    torch.sigmoid(net(s.to(dev), l.to(dev))).detach().cpu().numpy()
    for s, l, _ in testL
])>=thr).astype(int))
    print("Confusion matrix:\n",cm)

if __name__=="__main__":
    main()

