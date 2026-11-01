import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
dati=np.loadtxt("cornuto.txt")
x=dati[:,0]
n=dati[:,1]
err=dati[:,3]

def modello(x,N,mu,sigma,N2,mu2,sigma2,c0,c1,c2):
    sig1=N*np.exp(-0.5*((x-mu)/sigma)**2)
    sig2=N2*np.exp(-0.5*((x-mu2)/sigma2)**2)
    fondo=c0+c1*x+c2*x**2
    return sig1+sig2+fondo

p0=[1,1,1,1,1,1,1,1,1]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x") 
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()

p0=[605,1,1,1,1,1,1,1,1]
popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()



p0=[605,1,1,1,1,1,1,1,-10]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()

p0=[605,1,1,1,7,1,1,1,-10]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()

p0=[605,3,1,1,7,1,1,1,-10]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()

p0=[605,3,1,100,7,1,1,1,-10]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()
p0=[605,3,1.2,300,7,0.2,-8,10,-100]

popt,pcov=curve_fit(modello,x,n,p0=p0,sigma=err,absolute_sigma=True,maxfev=50000)
print(popt)
perr=np.sqrt(np.diag(pcov))
xx=np.linspace(0,10,1000)
plt.errorbar(x,n,yerr=err,fmt="o",ms=3,label="Dati")
plt.plot(xx,modello(xx,*popt),label="Gaussiana + fondo parabolico")
plt.xlabel("x")
plt.ylabel("Conteggi per bin")
plt.title("Esercizio 2")
plt.text(
    .23, .46,
    f"$\mu_1$ = {popt[1]:.3f} $\pm$ {perr[1]:.3f}\n"
    f"$\sigma_1$ = {popt[2]:.3f} $\pm$ {perr[2]:.3f}\n"
    f"$\mu_2$ = {popt[4]:.3f} $\pm$ {perr[4]:.3f}\n"
    f"$\sigma_2$ = {popt[5]:.3f} $\pm$ {perr[5]:.3f}",
    transform=plt.gca().transAxes,
    va="top"
    )
plt.legend()
plt.tight_layout()
plt.show()
