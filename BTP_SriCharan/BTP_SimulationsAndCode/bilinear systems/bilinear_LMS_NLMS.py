# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:43:33 2020

@author: LENOVO
"""


import numpy as np
import matplotlib.pyplot as plt
file='./octave file/LMS.txt';
a=np.genfromtxt(file,delimiter=',');
N=np.shape(a)[0]-1
x=np.linspace(0,N,N);
y1=a[1:,0]
y2=a[1:,1]
mh_inf=a[0,0];
mg_inf=a[0,1];
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='blue')
plt.plot([0,N-1],[mh_inf,mh_inf],color='red');
plt.plot([0,N-1],[mg_inf,mg_inf],color='blue');

file='./NLMS.txt';
a=np.genfromtxt(file,delimiter=',');
N=np.shape(a)[0]-1
x=np.linspace(0,N,N);
y1=a[1:,0]
y2=a[1:,1]
mh_inf=a[0,0];
mg_inf=a[0,1];
plt.plot(x,y1,color='green')
plt.plot(x,y2,color='black')
plt.plot([0,N-1],[mh_inf,mh_inf],color='green');
plt.plot([0,N-1],[mg_inf,mg_inf],color='black');