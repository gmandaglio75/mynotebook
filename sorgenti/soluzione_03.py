import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lineare(x,q,m): 
    return q+m*x
def esponenziale(x,A,k): 
    return A*np.exp(k*x)
def esegui(nome,modello,p0,titolo,pdf):
    x,y=np.loadtxt(nome,delimiter=',',skiprows=1,unpack=True)
    popt,pcov=curve_fit(modello,x,y,p0=p0)
    perr=np.sqrt(np.diag(pcov))
    r=y-modello(x,*popt)
    xx=np.linspace(x.min(),x.max(),500)
    fig,(ax,ar)=plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
    ax.plot(x,y,'o',label='Dati')
    ax.plot(xx,modello(xx,*popt),label='Fit')
    ax.legend()
    ax.set_ylabel('y')
    ax.set_title(titolo)
    ax.text(.03,.95,'\n'.join(f'p{i} = {v:.4g} +/- {e:.2g}' for i,(v,e) in enumerate(zip(popt,perr))),transform=ax.transAxes,va='top')
    ar.axhline(0,color='black')
    ar.plot(x,r,'o')
    ar.set_xlabel('x')
    ar.set_ylabel('residui')
    fig.tight_layout()
    fig.savefig(pdf)
    plt.show()
esegui('dati_03a_lineare.csv',lineare,[1,1],'Fit lineare','soluzione_03a.pdf')
esegui('dati_03b_esponenziale.csv',esponenziale,[1,.2],'Fit esponenziale','soluzione_03b.pdf')
