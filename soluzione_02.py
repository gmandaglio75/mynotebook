import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

maschera = dati > 0
dati[maschera]=np.loadtxt("datigpa.txt")
n,bordi=np.histogram(dati,bins=80)
x=(bordi[:-1]+bordi[1:])/2
bw=bordi[1]-bordi[0]
err=np.sqrt(np.maximum(n,1))
xref=(bordi[0]+bordi[-1])/2

def modello(x,N,mu,sigma,c0,c2):
    seg=N*bw/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((x-mu)/sigma)**2)
    fondo=c0+c2*(x-xref)**2
    return seg+fondo
p0=[10000,3,1.5,50,-1]
popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(bordi[0],bordi[-1],1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x"); plt.ylabel("Conteggi per bin"); plt.title("Esercizio 2")
plt.text(.03,.96,f"mu = {popt[1]:.3f} +/- {perr[1]:.3f}\nsigma = {popt[2]:.3f} +/- {perr[2]:.3f}\nc2 = {popt[4]:.3f}",transform=plt.gca().transAxes,va="top")
plt.legend(); plt.tight_layout(); plt.savefig("soluzione_02.pdf"); plt.show()
