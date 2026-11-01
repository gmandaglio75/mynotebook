import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoide(x,y0,L,x0,s): return y0+L/(1+np.exp(-(x-x0)/s))
x,y=np.loadtxt('dati_05_sigmoide.csv',delimiter=',',skiprows=1,unpack=True)
p0=[y.min(),y.max()-y.min(),20,2]
popt,pcov=curve_fit(sigmoide,x,y,p0=p0,bounds=([-np.inf,0,10,.05],[np.inf,np.inf,30,20]),maxfev=20000)
perr=np.sqrt(np.diag(pcov)); r=y-sigmoide(x,*popt); xx=np.linspace(10,30,600)
fig,(ax,ar)=plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
ax.plot(x,y,'o',label='Dati'); ax.plot(xx,sigmoide(xx,*popt),label='Fit sigmoidale'); ax.legend(); ax.set_ylabel('y'); ax.set_title('Esercizio 5')
ax.text(.03,.96,'\n'.join(f'{n} = {v:.3f} +/- {e:.3f}' for n,v,e in zip(['y0','L','x0','s'],popt,perr)),transform=ax.transAxes,va='top')
ar.axhline(0,color='black'); ar.plot(x,r,'o'); ar.set_xlabel('x'); ar.set_ylabel('residui')
fig.tight_layout(); fig.savefig('soluzione_05.pdf'); plt.show()
