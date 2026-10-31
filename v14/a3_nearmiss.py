"""A3 follow-up (descriptive): in the strict run's never-stored docs, which stored fact did the top row come
from (= the definition whose value is the mind's read answer), and how similar is its key to the asked key?
    python v14/a3_nearmiss.py   (reads logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl)
"""
import sys,json,random,zlib,re,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; MAIN=Path('/home/gowrav/Development/qllm2')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v16.bench import build
from memory_probes.spread import filler_tokens
from v7.data import get_chat_tokenizer
tok=get_chat_tokenizer(); fill=filler_tokens(str(MAIN/'.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
rows=[json.loads(l) for l in open(MAIN/'logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl')]
L=32768; filler=fill if L+2<len(fill) else np.concatenate([fill]*(L//len(fill)+2))
DEF=re.compile(r" Definition: ([A-Za-z]+(?: [A-Za-z]+)?) refers to (\w+)\.")
def sim(a,b,keys):
    if keys=='synthetic':  # 2 syllables of 3 letters
        s=[a[:3]==b[:3], a[3:6]==b[3:6]]; ch=len(set(a)&set(b))/max(1,len(set(a)|set(b)))
        return dict(first=int(s[0]),second=int(s[1]),any_syl=int(any(s) or a[:3]==b[3:6] or a[3:6]==b[:3]),char_j=round(ch,2))
    wa,wb=set(a.lower().split()),set(b.lower().split())
    return dict(shared_word=int(bool(wa&wb)),last_word=int(a.split()[-1].lower()==b.split()[-1].lower()),char_j=round(len(set(a)&set(b))/max(1,len(set(a)|set(b))),2))
for keys in ('synthetic','natural'):
    rng=random.Random(22000*1000+L+zlib.crc32(b'neg')%997)
    R={r['doc']:r for r in rows if r['task']=='neg' and r['keys']==keys}
    fet=collections.Counter(); base=collections.Counter(); nf=nb=0; amb=0; nodef=0; ex=[]
    for d in range(40):
        ids,ans,qlen,meta=build(rng,tok,filler,L,'neg',8,keys=keys)
        text=tok.decode(ids[:len(ids)-qlen]); defs=DEF.findall(text)
        pc=R[d]['P_cands'][0]; v=tok.decode([pc['answer']]).strip()
        m=[k for k,vv in defs if vv==v]
        for k,_ in defs:
            for a,b in sim(meta['key'],k,keys).items(): base[a]+=b
            nb+=1
        if not m: nodef+=1; continue
        if len(m)>1: amb+=1
        k=m[-1]
        for a,b in sim(meta['key'],k,keys).items(): fet[a]+=b
        nf+=1
        if len(ex)<6: ex.append((meta['key'],k,round(pc['sim'],3),pc['gain']))
    print(keys,'neg docs 40; top-1 read answer matches a stored definition value:',nf,'(ambiguous',amb,') no def value:',nodef)
    print('  fetched-row key vs asked key (mean):',{a:round(b/max(1,nf),3) for a,b in fet.items()})
    print('  any stored key vs asked key (base): ',{a:round(b/max(1,nb),3) for a,b in base.items()})
    print('  examples:',[e[:2] for e in ex])
