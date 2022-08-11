# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:55:35 2020

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class LMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
        
    def update_filt(self,data,mu):
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + mu*error*data 
        return 10*np.log10(error*error)
    
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))
    
    
class NLMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
        
    def update_filt(self,data,mu,delta):
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + (mu*error*data)/(np.sum(data*data) + delta)
        return 10*np.log10(error*error)
    
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))
    
    

class ZA_LMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
        
    def update_filt(self,data,mu,rho):
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + mu*error*data - rho*np.sign(self.filt_now)
        return 10*np.log10(error*error)
    
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))

class RZA_LMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
        
    def update_filt(self,data,mu,rho,eps):
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + mu*error*data - rho*(np.sign(self.filt_now) / (1 +eps* np.abs(self.filt_now)))
        return 10*np.log10(error*error)
    
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))
        
        
class PNLMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
    def update_filt(self,data,mu,rho,delta_NLMS,delta_p):
        abs_filt_now = np.abs(self.filt_now)
        maxi_triv = np.maximum(rho*np.max(abs_filt_now),delta_p)
        gamma = np.maximum(maxi_triv,abs_filt_now)
        G = gamma /np.sum(gamma)
        L = self.filt_size
        delta_PNLMS = delta_NLMS/L
        factor = np.sum(G*data*data) + delta_PNLMS
        
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + (mu*error*G*data)/factor
        
        return 10*np.log10(error*error)
        
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))
        
class IPNLMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
    def update_filt(self,data,mu,alpha,delta_NLMS,eps):
        abs_filt_now = np.abs(self.filt_now)
        L = self.filt_size
        norm1_filt_now = np.sum(abs_filt_now)
        K = (1-alpha)/(2*L) + ((1+alpha)/(eps + 2*norm1_filt_now))*abs_filt_now
        delta_IPNLMS = delta_NLMS/(2*L)
        factor = np.sum(K*data*data) + delta_IPNLMS
        
        noise = np.random.normal(0,self.noise_std,1)
        true_out = np.dot(self.filt_true,data) + noise 
        virtual_out = np.dot(self.filt_now,data)
        error = true_out - virtual_out
        self.filt_now = self.filt_now + (mu*error*K*data)/factor
        
        return 10*np.log10(error*error)
    
    def get_misalignment(self):
        return 20*np.log10(np.linalg.norm(self.filt_true-self.filt_now)/np.linalg.norm(self.filt_true))
    
"""class MPNLMS:
    def __init__(self,filter_coef,noise_var):
        self.filt_true = filter_coef
        
        self.noise_std=np.sqrt(noise_var)
        self.filt_size = filter_coef.shape[0]
        self.filt_now = np.zeros(self.filt_size)
        
    def update_filt(self,data,mu,rho,delta_NLMS,eps):
        abs_filt_now = np.abs(self.filt_now)
        L = self.filt_size
        filt_now_f = np.log(1+ mu*abs_filt_now)/np.log(1+mu)
        
"""        
    

#@jit(nopython=True)
def simulation1():
    mu = 0.05
    rho = 0.0005
    eps = 10
    data_var = 1
    data_std = np.sqrt(data_var)
    noise_var = 0.001
    noise_std = np.sqrt(noise_var)
    
    
    
    
    
    
    no_experiments = 1000
    no_iterations = 1500
    no_iterations1 = 500
    no_iterations2 = 1000
    no_iterations3 = 1500
    
    MSE_LMS = np.zeros(no_iterations)
    MSE_ZA_LMS = np.zeros(no_iterations)
    MSE_RZA_LMS = np.zeros(no_iterations)
    MSE_PNLMS = np.zeros(no_iterations)
    MSE_IPNLMS = np.zeros(no_iterations)
    
    n=16
    
    for j in range(0,no_experiments):
        filt_coef = np.zeros(n)
        filt_coef[4]=1
        filter1 = LMS(filt_coef,noise_var)
        filter2 = ZA_LMS(filt_coef,noise_var)
        filter3 = RZA_LMS(filt_coef,noise_var)
        
        for i in range(0,no_iterations1):
            data = np.random.normal(0,data_std,n)
            
            MSE_LMS[i]+= filter1.update_filt(data,mu)
            MSE_ZA_LMS[i]+= filter2.update_filt(data,mu,rho)
            MSE_RZA_LMS[i]+= filter3.update_filt(data,mu,rho,eps)
            
        filt_coef = np.zeros(n)
        for k in range(1,n,2):
            filt_coef[k]=1
            
        filter1 = LMS(filt_coef,noise_var)
        filter2 = ZA_LMS(filt_coef,noise_var)
        filter3 = RZA_LMS(filt_coef,noise_var)
            
        for i in range(no_iterations1,no_iterations2):
            data = np.random.normal(0,data_std,n)
            
            MSE_LMS[i]+= filter1.update_filt(data,mu)
            MSE_ZA_LMS[i]+= filter2.update_filt(data,mu,rho)
            MSE_RZA_LMS[i]+= filter3.update_filt(data,mu,rho,eps)
            
        filt_coef = np.zeros(n)
        for k in range(1,n,2):
            filt_coef[k]=1
        for k in range(0,n,2):
            filt_coef[k]=-1
            
        filter1 = LMS(filt_coef,noise_var)
        filter2 = ZA_LMS(filt_coef,noise_var)
        filter3 = RZA_LMS(filt_coef,noise_var)
            
        for i in range(no_iterations2,no_iterations3):
            data = np.random.normal(0,data_std,n)
            
            MSE_LMS[i]+= filter1.update_filt(data,mu)
            MSE_ZA_LMS[i]+= filter2.update_filt(data,mu,rho)
            MSE_RZA_LMS[i]+= filter3.update_filt(data,mu,rho,eps)
            
            
    
    
    MSE_LMS = MSE_LMS / no_experiments
    MSE_ZA_LMS = MSE_ZA_LMS / no_experiments
    MSE_RZA_LMS = MSE_RZA_LMS / no_experiments
    
    x = np.linspace(0,no_iterations,no_iterations)
    plt.plot(x,MSE_LMS,'red',label="LMS")   
    plt.plot(x,MSE_ZA_LMS,'green',label="ZA_LMS")    
    plt.plot(x,MSE_RZA_LMS,'blue',label="RZA_LMS") 
    #plt.plot(x,MSE_PNLMS,'black')  
    plt.ylabel("MSD")
    plt.legend()
    plt.show()

 

def simulation2():
    n=64
    mu=0.2
    alpha=0.5
    delta_p=0.01
    rho=0.01
    eps = 10
    data_var = 1
    data_std = np.sqrt(data_var)
    noise_var = 0.001
    
    delta_NLMS= data_var
    
    no_experiments = 1
    no_iterations = 60000
    no_iterations1 = 20000
    no_iterations2 = 40000
    no_iterations3 = 60000
    
    MSE_NLMS = np.zeros(no_iterations)
    MSE_PNLMS = np.zeros(no_iterations)
    MSE_IPNLMS = np.zeros(no_iterations)
    
    for j in range(0,no_experiments):
        ########too sparse
        filt_coef = np.zeros(n)
        filt_coef[4]=1
        filter1 = NLMS(filt_coef,noise_var)
        filter2 = PNLMS(filt_coef,noise_var)
        filter3 = IPNLMS(filt_coef,noise_var)
        
        for i in range(0,no_iterations1):
            data = np.random.normal(0,data_std,n)
            
            filter1.update_filt(data,mu,delta_NLMS)
            MSE_NLMS[i]+=filter1.get_misalignment()
            filter2.update_filt(data,mu,rho,delta_NLMS,delta_p)
            MSE_PNLMS[i]+=filter2.get_misalignment()
            filter3.update_filt(data,mu,alpha,delta_NLMS,eps)
            MSE_IPNLMS[i]+=filter3.get_misalignment()
            
            
        #############moderately sparse    
        filt_coef = np.zeros(n)
        for k in range(1,n,2):
            filt_coef[k]=1
            
        filter1 = NLMS(filt_coef,noise_var)
        filter2 = PNLMS(filt_coef,noise_var)
        filter3 = IPNLMS(filt_coef,noise_var)
            
        for i in range(no_iterations1,no_iterations2):
            data = np.random.normal(0,data_std,n)
            
            filter1.update_filt(data,mu,delta_NLMS)
            MSE_NLMS[i]+=filter1.get_misalignment()
            filter2.update_filt(data,mu,rho,delta_NLMS,delta_p)
            MSE_PNLMS[i]+=filter2.get_misalignment()
            filter3.update_filt(data,mu,alpha,delta_NLMS,eps)
            MSE_IPNLMS[i]+=filter3.get_misalignment()
        
        
        
        ###########dispersive filter
        filt_coef = np.zeros(n)
        for k in range(1,n,2):
            filt_coef[k]=1
        for k in range(0,n,2):
            filt_coef[k]=-1
            
        filter1 = NLMS(filt_coef,noise_var)
        filter2 = PNLMS(filt_coef,noise_var)
        filter3 = IPNLMS(filt_coef,noise_var)
            
        for i in range(no_iterations2,no_iterations3):
            data = np.random.normal(0,data_std,n)
            
            filter1.update_filt(data,mu,delta_NLMS)
            MSE_NLMS[i]+=filter1.get_misalignment()
            filter2.update_filt(data,mu,rho,delta_NLMS,delta_p)
            MSE_PNLMS[i]+=filter2.get_misalignment()
            filter3.update_filt(data,mu,alpha,delta_NLMS,eps)
            MSE_IPNLMS[i]+=filter3.get_misalignment()
            
            
    
    
    MSE_NLMS = MSE_NLMS/no_experiments
    MSE_PNLMS = MSE_PNLMS/no_experiments
    MSE_IPLMS = MSE_IPNLMS/no_experiments
    
    x = np.linspace(0,no_iterations,no_iterations)
    plt.plot(x,MSE_NLMS,'red',label='NLMS')   
    plt.plot(x,MSE_PNLMS,'green',label='PNLMS')    
    plt.plot(x,MSE_IPNLMS,'blue',label='IPNLMS')  
    plt.xlabel('iteration number')
    plt.ylabel('misadjustment') 
    plt.legend()
    plt.show()
         
    
simulation1()    
    
