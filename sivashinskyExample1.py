#
# Python code for example 1
#
# Written by Reza Abazari on August 01, 2024. 
# Copyright 2010 by Reza Abazari. All Right Reserved.
# e-mail(s): abazari-r@uma.ac.ir, abazri.r@gmail.com.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft2, ifft2
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d 
import pandas as pd
from numpy import linalg as LA


plt.show(block=False)
plt.close('all')
alpha = 1        # parameter of Sivashinsky equation
dx = 1           # Sample spacing (inverse of the sampling rate). Defaults to 1

u1half = np.empty((5,8,8), dtype=np.float32)
u3half = np.empty((5,8,8), dtype=np.float32)
u6half = np.empty((5,8,8), dtype=np.float32)
u10half = np.empty((5,8,8), dtype=np.float32)

N = [8,16,32,64,128]
Nsteps = 2500
dt = 0.02
MassErr = np.empty((16*(250)+1), dtype=np.float32)
MassErr[0] = 0

for l in range(5): # l = 0 for N=8 and l=1 for N=16  and ... to l=4 for N=128
    print('level=',l, 'Nsteps=',Nsteps,'dt=',dt)
    
    u_hat = np.empty((N[l],N[l]), dtype=np.complex64)
    dfdu_hat = np.empty((N[l],N[l]), dtype=np.complex64)
    u = np.empty((Nsteps+1,N[l],N[l]), dtype=np.float32)
    
    L = N[l]*dx
    
    # Example 1: Benchmark Cross initial condition via heaviside function
    x = np.arange(N[l])
    y = np.arange(N[l])
    x, y = np.meshgrid(x, y) 
    u[0] = (-1)*(np.heaviside(x-round(N[l]/5),0)-np.heaviside(x-round(4*N[l]/5),0))*(np.heaviside(y-round(2*N[l]/5),0)-np.heaviside(y-round(3*N[l]/5),0))\
    + (-1)*(np.heaviside(x-round(2*N[l]/5),0)-np.heaviside(x-round(3*N[l]/5),0))*(np.heaviside(y-round(3*N[l]/5),0)-np.heaviside(y-round(4*N[l]/5),0))\
    + (-1)*(np.heaviside(x-round(2*N[l]/5),0)-np.heaviside(x-round(3*N[l]/5),0))*(np.heaviside(y-round(1*N[l]/5),0)-np.heaviside(y-round(2*N[l]/5),0))+0.5
    
    if l == 4:
        # figure 1 x 2
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(u[0],cmap='RdBu_r')
        plt.colorbar(cmap='RdBu_r')
        plt.xlabel('x', fontsize=16) 
        plt.ylabel('y', fontsize=16)
        plt.title('t={}'.format(0*dt))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(x, y, u[0], rstride=1, cstride=1, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)
        plt.xlabel('x', fontsize=16) 
        plt.ylabel('y', fontsize=16)
        plt.show()
        # Mass of initial condition
        Mass_u0 = u[0].sum()/(N[l]**2)
    
    
    kx = ky = np.fft.fftfreq(N[l], d=dx)*2*np.pi
    K = np.array(np.meshgrid(kx , ky ,indexing ='ij'), dtype=np.float32)
    K2 = np.sum(K*K,axis=0, dtype=np.float32)
    
    # The anti-aliasing factor  
    kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
    dealias = np.array((np.abs(K[0]) < kmax_dealias )*(np.abs(K[1]) < kmax_dealias ),dtype =bool)
    
    def dfdu(u):
        return 1/2*u**2-2*u
    
    u_hat[:] = fft2(u[0])
    
    for i in range(1,Nsteps+1):
        dfdu_hat[:] = fft2(dfdu(u[i-1])) # the FT of the derivative
        dfdu_hat *= dealias # dealising
        u_hat[:] = (u_hat-dt*K2*dfdu_hat)/(1+dt*K2**2+dt*alpha) # updating in time
        u[i] = ifft2(u_hat).real # inverse fourier transform
        if i == 25*(2**l):
            u1half[l,:,:]=u[i,::2**l,::2**l]
            
        if i == 75*(2**l):
            u3half[l,:,:]=u[i,::2**l,::2**l]
            
        if i== 150*(2**l):
            u6half[l,:,:]=u[i,::2**l,::2**l]
            
        if i == 250*(2**l):
            u10half[l,:,:]=u[i,::2**l,::2**l]
        
        if l == 4:
            if i < 250*(2**(l))+1:
                Mass_u = u[i].sum()/(N[l]**2)
                MassErr[i] = Mass_u - Mass_u0
            if i in (400,1200,2400,4000,8000,12000,16000,24000,32000,40000):
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(u[i],cmap='RdBu_r')
                plt.colorbar(cmap='RdBu_r')
                plt.xlabel('x', fontsize=16) 
                plt.ylabel('y', fontsize=16)
                plt.title('t={}'.format(i*dt))
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                X = np.arange(N[l])
                Y = np.arange(N[l])
                X, Y = np.meshgrid(X, Y)
                ax.plot_surface(X, Y, u[i], rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
                plt.xlabel('x', fontsize=16) 
                plt.ylabel('y', fontsize=16)
                plt.show()

    Nsteps = Nsteps*2
    dt = dt/2
    del u  # deleting the computed u[] in previous loop to keep performance

# Mass Error
t =  np.linspace(0, 5, 16*250+1)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = plt.gca()
ax.plot(t,MassErr, label=r'$\int_{\Omega}u(.,t)d\Omega-\int_{\Omega}u(.,0)d\Omega$')
ax.legend(fontsize=16)
plt.xlabel('t', fontsize=16) 
plt.ylabel('Mass Error', fontsize=16) 
plt.show()

# L^2 Error with convergence rate
ErrL2 = np.zeros((4,9))
ErrL2[:,0] =np.transpose([16,32,64,128])

for j in range(1,5):
    ErrL2[j-1,1] = np.sqrt(((u1half[j,:,:]-u1half[j-1,:,:])**2).sum())/N[j]
    ErrL2[j-1,3] = np.sqrt(((u3half[j,:,:]-u3half[j-1,:,:])**2).sum())/N[j]
    ErrL2[j-1,5] = np.sqrt(((u6half[j,:,:]-u6half[j-1,:,:])**2).sum())/N[j]
    ErrL2[j-1,7] = np.sqrt(((u10half[j,:,:]-u10half[j-1,:,:])**2).sum())/N[j]

for r in range(1,4):
    for c in range(1,5):
        ErrL2[r,2*c] = np.log(ErrL2[r-1,2*c-1]/ErrL2[r,2*c-1])/np.log(2)

# L^2 error as Table
print('-'*30, 'L^2 Error','-'*30)
print(ErrL2) 
#----------------------------------------------------------




