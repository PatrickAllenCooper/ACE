import math
from statistics import NormalDist
N=NormalDist()
def z(p): return N.inv_cdf(p)
# November paired log final posterior-risk contrasts (t, diff) at n=24 -> SD
for name,d,t in [("local_info",-0.055,-2.6),("coverage",-0.319,-8.1),("random",-0.347,-6.6),("ranker",-0.523,-9.2)]:
    se=abs(d/t); print(f"{name}: SE={se:.4f} SD={se*math.sqrt(24):.3f}")
m=3; alpha=0.05; beta=0.2
k=(z(1-alpha/(2*m))+z(1-beta))**2
print("z_holm",round(z(1-alpha/(2*m)),3),"k",round(k,2))
for red in (0.10,0.15,0.20):
    dl=math.log(1/(1-red))
    for ss in (0.10,0.20,0.25,0.28):
        for R in (4,8):
            for sr in (0.5,):
                v=ss**2+sr**2/R
                print(f"red={red:.2f} delta={dl:.4f} sig_s={ss} R={R} sig_r={sr} n={k*v/dl**2:.1f}")
# E5 gate precision (n=64,R=4, sig_s=.25,sig_r=.5)
se=math.sqrt(0.25**2+0.5**2/4)/8; print("E5 SE",round(se,4),"CI half",round(1.96*se,4))
# E1 dev precision (n=32,R=8)
se=math.sqrt(0.25**2+0.5**2/8)/math.sqrt(32); print("E1 SE",round(se,4),"CI half",round(1.96*se,4))
# E7 MDE n=64,R=4 Holm-3
v=0.25**2+0.5**2/4; d=math.sqrt(k*v/64); print("E7 MDE log",round(d,4),"reduction",round(1-math.exp(-d),3))
# hardware online MDE
for n in (12,24,48):
    d=(z(0.975)+z(0.8))*0.65/math.sqrt(n); print("hw blocks",n,"MDE log",round(d,3),"reduction",round(1-math.exp(-d),3))
# hardware blocks needed for 15% at sd .65, two-sided .05
d=math.log(1/0.85); print("blocks for 15%:", ((z(0.975)+z(0.8))*0.65/d)**2)
# chamber arithmetic
eur=0.014
E2=4*15+2*10*10+20*10+3*2*10+200; print("E2 s",E2)
pool=25*2*12*14; desc=5*12*180; powd=3*24*180; ov=700
for nm,on in (("descriptive",desc),("powered",powd)):
    tot=pool+on+ov; print(nm,"pool",pool,"online",on,"total s",tot,"h",round(tot/3600,2),"EUR",round(tot*eur,1),"+30%",round(tot*eur*1.3,1),"commercial",round(tot*0.038,1))
e10=25*1*12*14+2*6*180+300; print("E10 s",e10,"EUR",round(e10*eur,1),"+30%",round(e10*eur*1.3,1))
# brief check: 4 policies x 30 episodes x 16 interventions x 10 s
b=4*30*16*10; print("brief check s",b,"h",round(b/3600,2),"EUR",round(b*eur,1))
# compute: E6 decisions
llm=7*128*4*20; num=9*128*4*20; roll=128*4*20
print("E6 llm passes",llm,"h@0.4s",round(llm*0.4/3600,1))
print("E6 numerical CPU-h",round((num*0.5+roll*7)/3600,1))
# E5 GPU estimates
sft=30000*2
for per in (0.1,0.25):
    print("SFT examples",sft,"slice-h",round(sft*per/3600,2))
gr=100*64*20; print("GRPO decisions",gr,"gen slice-h@.05",round(gr*0.05/3600,2),"bwd slice-h@.1",round(gr*0.1/3600,2))
# Qwen3-0.6B forward FLOPs ~1k tokens
print("FLOPs per decision ~",2*0.6e9*1000)
