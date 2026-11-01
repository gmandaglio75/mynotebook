import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.optimize import curve_fit

def lineare(B,x): return B[0]+B[1]*x
def esponenziale(B,x): return B[0]*np.exp(B[1]*x)
def esegui(nome,fodr,fcf,beta0,titolo,pdf):
    x,y,sx,sy=np.loadtxt(nome,delimiter=',',skiprows=1,unpack=True)
    out=ODR(RealData(x,y,sx=sx,sy=sy),Model(fodr),beta0=beta0).run()
    py,_=curve_fit(fcf,x,y,p0=beta0,sigma=sy,absolute_sigma=True,maxfev=20000)
    xx=np.linspace(x.min(),x.max(),500)
    plt.errorbar(x,y,xerr=sx,yerr=sy,fmt='o',ms=3,label='Dati')
    plt.plot(xx,fodr(out.beta,xx),label='ODR: errori x e y'); plt.plot(xx,fcf(xx,*py),'--',label='curve_fit: solo errori y')
    plt.text(.03,.97,'ODR: '+', '.join(f'p{i}={v:.4g}+/-{e:.2g}' for i,(v,e) in enumerate(zip(out.beta,out.sd_beta))),transform=plt.gca().transAxes,va='top')
    plt.xlabel('x'); plt.ylabel('y'); plt.title(titolo); plt.legend(); plt.tight_layout(); plt.savefig(pdf); plt.show()
esegui('dati_04a_lineare_errori.csv',lineare,lambda x,q,m:q+m*x,[1,1],'Lineare con errori x-y','soluzione_04a.pdf')
esegui('dati_04b_esponenziale_errori.csv',esponenziale,lambda x,A,k:A*np.exp(k*x),[1,.2],'Esponenziale con errori x-y','soluzione_04b.pdf')
