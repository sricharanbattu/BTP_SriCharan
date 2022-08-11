# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:11:32 2020

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt


class LMS_BF:
    def __init__(self,h_true,g_true,hcap_init,gcap_init,noise_var):
        self.h_true = h_true
        self.g_true = g_true
        self.f_true = np.kron(g_true,h_true)
        self.h_cap = hcap_init
        self.g_cap = gcap_init
        self.f_cap = np.kron(gcap_init,hcap_init)
        self.noise_std = np.sqrt(noise_var)
        self.L = h_true.shape[0]
        self.M = g_true.shape[0]
        self.IL = np.identity(self.L)
        self.IM = np.identity(self.M)
        
    def update_filt(self,xvec,mu_h,mu_g):
        y = np.dot(self.f_true,xvec)
        noise = np.random.normal(0,self.noise_std,1)
        d = y+noise
        
        
        xgcap = np.dot(np.kron(self.g_cap.reshape(-1,1),self.IL).transpose(),xvec)
        xhcap = np.dot(np.kron(self.IM,self.h_cap.reshape(-1,1)).transpose(),xvec)
        #ycap = np.dot(self.f_cap,xvec)
        
        egcap = d - np.dot(self.h_cap,xgcap)
        ehcap = d - np.dot(self.g_cap,xhcap)
        
        self.h_cap = self.h_cap + mu_h*egcap*xgcap
        self.g_cap = self.g_cap + mu_g*ehcap*xhcap 
        self.f_cap = np.kron(self.g_cap,self.h_cap)
        
    def get_NPMg(self):
        return 10*np.log10(1 - (np.square(np.dot(self.g_true,self.g_cap))/(np.dot(self.g_true,self.g_true)*np.dot(self.g_cap,self.g_cap))))
    
    def get_NPMh(self):
        return 10*np.log10(1 - (np.square(np.dot(self.h_true,self.h_cap))/(np.dot(self.h_true,self.h_true)*np.dot(self.h_cap,self.h_cap))))
    
    def get_NMf(self):
        return 10*np.log10(np.square(np.linalg.norm(self.f_true-self.f_cap))/np.square(np.linalg.norm(self.f_true)))
 

class NLMS_BF:
    def __init__(self,h_true,g_true,hcap_init,gcap_init,noise_var):
        self.h_true = h_true
        self.g_true = g_true
        self.f_true = np.kron(g_true,h_true)
        self.h_cap = hcap_init
        self.g_cap = gcap_init
        self.f_cap = np.kron(gcap_init,hcap_init)
        self.noise_std = np.sqrt(noise_var)
        self.L = h_true.shape[0]
        self.M = g_true.shape[0]
        self.IL = np.identity(self.L)
        self.IM = np.identity(self.M)
        
    def update_filt(self,xvec,alpha_h,alpha_g,delta_h,delta_g):
        y = np.dot(self.f_true,xvec)
        noise = np.random.normal(0,self.noise_std,1)
        d = y+noise
        
        
        xgcap = np.dot(np.kron(self.g_cap.reshape(-1,1),self.IL).transpose(),xvec)
        xhcap = np.dot(np.kron(self.IM,self.h_cap.reshape(-1,1)).transpose(),xvec)
        #ycap = np.dot(self.f_cap,xvec)
        
        egcap = d - np.dot(self.h_cap,xgcap)
        ehcap = d - np.dot(self.g_cap,xhcap)
        
        self.h_cap = self.h_cap + (alpha_h*egcap*xgcap/(np.dot(xgcap,xgcap)+delta_h))
        self.g_cap = self.g_cap + (alpha_g*ehcap*xhcap/(np.dot(xhcap,xhcap)+delta_g))
        self.f_cap = np.kron(self.g_cap,self.h_cap)
        
    def get_NPMg(self):
        return 10*np.log10(1 - (np.square(np.dot(self.g_true,self.g_cap))/(np.dot(self.g_true,self.g_true)*np.dot(self.g_cap,self.g_cap))))
    
    def get_NPMh(self):
        return 10*np.log10(1 - (np.square(np.dot(self.h_true,self.h_cap))/(np.dot(self.h_true,self.h_true)*np.dot(self.h_cap,self.h_cap))))
    
    def get_NMf(self):
        return 10*np.log10(np.square(np.linalg.norm(self.f_true-self.f_cap))/np.square(np.linalg.norm(self.f_true)))
               


class ZALMS_BF:
    def __init__(self,h_true,g_true,hcap_init,gcap_init,noise_var):
        self.h_true = h_true
        self.g_true = g_true
        self.f_true = np.kron(g_true,h_true)
        self.h_cap = hcap_init
        self.g_cap = gcap_init
        self.f_cap = np.kron(gcap_init,hcap_init)
        self.noise_std = np.sqrt(noise_var)
        self.L = h_true.shape[0]
        self.M = g_true.shape[0]
        self.IL = np.identity(self.L)
        self.IM = np.identity(self.M)
        
    def update_filt(self,xvec,mu_h,mu_g,rho_h):
        y = np.dot(self.f_true,xvec)
        noise = np.random.normal(0,self.noise_std,1)
        d = y+noise
        
        
        xgcap = np.dot(np.kron(self.g_cap.reshape(-1,1),self.IL).transpose(),xvec)
        xhcap = np.dot(np.kron(self.IM,self.h_cap.reshape(-1,1)).transpose(),xvec)
        #ycap = np.dot(self.f_cap,xvec)
        
        egcap = d - np.dot(self.h_cap,xgcap)
        ehcap = d - np.dot(self.g_cap,xhcap)
        
        self.h_cap = self.h_cap + mu_h*egcap*xgcap - rho_h*np.sign(self.h_cap)
        self.g_cap = self.g_cap + mu_g*ehcap*xhcap 
        self.f_cap = np.kron(self.g_cap,self.h_cap) 
        
    def get_NPMg(self):
        return 10*np.log10(1 - (np.square(np.dot(self.g_true,self.g_cap))/(np.dot(self.g_true,self.g_true)*np.dot(self.g_cap,self.g_cap))))
    
    def get_NPMh(self):
        return 10*np.log10(1 - (np.square(np.dot(self.h_true,self.h_cap))/(np.dot(self.h_true,self.h_true)*np.dot(self.h_cap,self.h_cap))))
    
    def get_NMf(self):
        return 10*np.log10(np.square(np.linalg.norm(self.f_true-self.f_cap))/np.square(np.linalg.norm(self.f_true)))


class PNLMS_BF:
    def __init__(self,h_true,g_true,hcap_init,gcap_init,noise_var):
        self.h_true = h_true
        self.g_true = g_true
        self.f_true = np.kron(g_true,h_true)
        self.h_cap = hcap_init
        self.g_cap = gcap_init
        self.f_cap = np.kron(gcap_init,hcap_init)
        self.noise_std = np.sqrt(noise_var)
        self.L = h_true.shape[0]
        self.M = g_true.shape[0]
        self.IL = np.identity(self.L)
        self.IM = np.identity(self.M)
        
    def update_filt(self,xvec,alpha_h,alpha_g,delta_h,delta_g,rho_h,deltap_h):
        y = np.dot(self.f_true,xvec)
        noise = np.random.normal(0,self.noise_std,1)
        d = y+noise
        
        abs_hcap = np.abs(self.h_cap)
        maxi_triv = np.maximum(rho_h*np.max(abs_hcap),deltap_h)
        gamma_h = np.maximum(maxi_triv,abs_hcap)
        G = gamma_h /np.sum(gamma_h)
        
        delta_PNLMS_h = delta_h/self.L
        
        
        
        xgcap = np.dot(np.kron(self.g_cap.reshape(-1,1),self.IL).transpose(),xvec)
        xhcap = np.dot(np.kron(self.IM,self.h_cap.reshape(-1,1)).transpose(),xvec)
        #ycap = np.dot(self.f_cap,xvec)
        
        egcap = d - np.dot(self.h_cap,xgcap)
        ehcap = d - np.dot(self.g_cap,xhcap)
        
        factor = np.sum(G*xgcap*xgcap) + delta_PNLMS_h

        self.h_cap = self.h_cap + (alpha_h*egcap*G*xgcap)/factor
        self.g_cap = self.g_cap + (alpha_g*ehcap*xhcap/(np.dot(xhcap,xhcap)+delta_g))
        self.f_cap = np.kron(self.g_cap,self.h_cap)
        
    def get_NPMg(self):
        return 10*np.log10(1 - (np.square(np.dot(self.g_true,self.g_cap))/(np.dot(self.g_true,self.g_true)*np.dot(self.g_cap,self.g_cap))))
    
    def get_NPMh(self):
        return 10*np.log10(1 - (np.square(np.dot(self.h_true,self.h_cap))/(np.dot(self.h_true,self.h_true)*np.dot(self.h_cap,self.h_cap))))
    
    def get_NMf(self):
        return 10*np.log10(np.square(np.linalg.norm(self.f_true-self.f_cap))/np.square(np.linalg.norm(self.f_true)))
 


class IPNLMS_BF:
    def __init__(self,h_true,g_true,hcap_init,gcap_init,noise_var):
        self.h_true = h_true
        self.g_true = g_true
        self.f_true = np.kron(g_true,h_true)
        self.h_cap = hcap_init
        self.g_cap = gcap_init
        self.f_cap = np.kron(gcap_init,hcap_init)
        self.noise_std = np.sqrt(noise_var)
        self.L = h_true.shape[0]
        self.M = g_true.shape[0]
        self.IL = np.identity(self.L)
        self.IM = np.identity(self.M)
        
    def update_filt(self,xvec,alpha_h,alpha_g,delta_h,delta_g,alpha,eps):
        y = np.dot(self.f_true,xvec)
        noise = np.random.normal(0,self.noise_std,1)
        d = y+noise
        
        
        xgcap = np.dot(np.kron(self.g_cap.reshape(-1,1),self.IL).transpose(),xvec)
        xhcap = np.dot(np.kron(self.IM,self.h_cap.reshape(-1,1)).transpose(),xvec)
        #ycap = np.dot(self.f_cap,xvec)
        
        egcap = d - np.dot(self.h_cap,xgcap)
        ehcap = d - np.dot(self.g_cap,xhcap)
        
        abs_hcap = np.abs(self.h_cap)
        L = self.L
        norm1_hcap = np.sum(abs_hcap)
        K = (1-alpha)/(2*L) + ((1+alpha)/(eps + 2*norm1_hcap))*abs_hcap
        delta_IPNLMS = delta_h/(2*L)
        factor = np.sum(K*xgcap*xgcap) + delta_IPNLMS
        
        self.h_cap = self.h_cap + (alpha_h*egcap*xgcap*K/factor)
        self.g_cap = self.g_cap + (alpha_g*ehcap*xhcap/(np.dot(xhcap,xhcap)+delta_g))
        self.f_cap = np.kron(self.g_cap,self.h_cap)
        
    def get_NPMg(self):
        return 10*np.log10(1 - (np.square(np.dot(self.g_true,self.g_cap))/(np.dot(self.g_true,self.g_true)*np.dot(self.g_cap,self.g_cap))))
    
    def get_NPMh(self):
        return 10*np.log10(1 - (np.square(np.dot(self.h_true,self.h_cap))/(np.dot(self.h_true,self.h_true)*np.dot(self.h_cap,self.h_cap))))
    
    def get_NMf(self):
        return 10*np.log10(np.square(np.linalg.norm(self.f_true-self.f_cap))/np.square(np.linalg.norm(self.f_true)))
                    
L= 20
M= 20
data_var = 1
data_std = np.sqrt(data_var)
noise_var = 0.01
h_true = np.zeros(L)
h_true[int(L/2)]=1
h_true[int(L/4)]=1
g_true = np.random.normal(0,1,M)
h_cap = np.zeros(L)
h_cap[0]=1
g_cap = np.ones(M)*(1/M)
#g_cap = np.zeros(M)
filt1=IPNLMS_BF(h_true,g_true,h_cap,g_cap,noise_var)
n_sim1 = 60000
NPM_h = np.zeros(n_sim1)
NPM_g = np.zeros(n_sim1)
mu_h = 0.01
mu_g = 0.01
rho_h = 0.01
delta_h=0.01
delta_g=0.01
delta_ph=0.01
alpha=0.5
eps=0.001


for i in range(0,n_sim1):
    xvec = np.random.normal(0,data_std,L*M)
    filt1.update_filt(xvec,mu_h,mu_g,delta_h,delta_g,alpha,eps)
    NPM_h[i]=filt1.get_NPMh()
    NPM_g[i]=filt1.get_NPMg()
    
x = np.linspace(0,n_sim1,n_sim1)
plt.plot(x,NPM_h,'red',label='NPM_h')   
plt.plot(x,NPM_g,'green',label='NPM_g')
plt.show()
    
    
        
        
        
        