
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:54:08 2022

@author: NIKHIL
"""

## let's import the relevant libraries
import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os
import math
import matplotlib.gridspec as Gridspec
from pyDOE import *
device = torch.device("cpu")





# def normalize(x):
#     # print("x",x)
#     # xnorm=(x-torch.min(x))/(torch.max(x)-torch.min(x))
#     xnorm=(x-0)/(10-0)
#     # print("xnorm",xnorm)
    
#     return xnorm

# The Psi_t function
def Psi(xt):
#Psi_t = lambda x:  ((x-l)**2) * N(x)
    # cc=x/xn
    #print(cc)
    Psi=N(xt)
    # print("Psi_t",Psi.shape)
    return  Psi
def Psi1(xtb1,xtb2):
#Psi_t = lambda x:  ((x-l)**2) * N(x)
    # cc=x/xn
    #print(cc)
    Psi_1=N(xtb1)
    Psi_2=N(xtb2)
    # print("Psi_t",Psi_t)
    return  Psi_1,Psi_2
# The right hand side function
#f = lambda x, Psi: 2*x
def f(x):
  return 0
# The loss function
def residue_loss(xt):
    # print("xt",xt)
    xt.requires_grad = True
    outputs = Psi(xt)
    # print("outputs",outputs.shape)
    Psi_t = torch.autograd.grad(outputs, xt, grad_outputs=torch.ones_like(outputs),
                                  create_graph=True)[0]
    # print("psi_t",Psi_t)
    # print("psi_t[:,1]",Psi_t[:,1])
    Psi_x = torch.autograd.grad(outputs, xt, grad_outputs=torch.ones_like(outputs),
                                  create_graph=True)[0]
    # print("psi_x",Psi_x)
    Psi_xx = torch.autograd.grad(Psi_x, xt, grad_outputs=torch.ones_like(Psi_x),
                                  create_graph=True)[0]
    out_psix=torch.zeros(len(outputs),1)
    # print("psixx",Psi_xx)
    for i in range(len(outputs)):
        out_psix[i,:]=outputs[i,:]*Psi_x[i,0]
    # print("out_psix",out_psix.shape)  
    # print("(0.01/np.pi)*Psi_xx[:,0]",((0.01/np.pi)*Psi_xx[:,0]).reshape(len(outputs),1))
    # residue=(Psi_t[:,1] +outputs*Psi_x[:,0]-((0.01/np.pi)*Psi_xx[:,0] ) ** 2
    residue=(Psi_t[:,1].reshape(len(outputs),1) +out_psix-((0.01/np.pi)*Psi_xx[:,0]).reshape(len(outputs),1)) ** 2
    # print("outputs*psi_x",Psi_x[:,0]*outputs)
    # print("res",residue)
    tm=torch.mean(residue)
    # print("tmean",tm)                             
    return tm
def boundary_loss(xtb1,xtb2):
    xtb1.requires_grad=True
    xtb2.requires_grad=True
    output1,output2=Psi1(xtb1,xtb2)
    loss2=torch.mean(output1**2+output2**2)
    return loss2
def initial_loss(xi):
    xi.requires_grad=True
    output_ic=Psi(xi)
    # print("output_ic",output_ic.shape)
    rhs_ic=torch.sin(math.pi*xi[:,0]).reshape(len(xi),1)
    # k=torch.transpose(K)
    # print("k,",rhs_ic.shape)
    
    loss3= torch.mean((output_ic+ rhs_ic)**2)
    # print("loss3",loss3)
    return loss3

# =============================================================================
# We need to initialize the network
# =============================================================================
N = nn.Sequential(nn.Linear(2, 20,bias=True),
                  nn.Tanh(), 
                  nn.Linear(20,20,bias=True)
                  ,nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,1, bias=True))
# adam = torch.optim.Adam(N.parameters(), lr=0.001)

# The batch size you want to use (how many points to use per iteration)
n_c = 10000
n_b=1000
lb_x,ub_x,lb_t,ub_t=-1,1,0,1
x_c=torch.tensor((ub_x-lb_x)*lhs(1,n_c)-1)
x_b=torch.tensor((ub_x-lb_x)*lhs(1,n_b)-1)
# x=(ub_x-lb_x)*torch.rand(n_batch,1)-1
# t=(ub_t-lb_t)*torch.rand(n_batch,1)-1
t_c=torch.tensor((ub_t-lb_t)*lhs(1,n_c))
t_b=torch.tensor((ub_t-lb_t)*lhs(1,n_b))
xt=torch.cat((x_c,t_c),axis=1)

#boundary data
x1=torch.ones(n_b,1)
x2=-1*torch.ones(n_b,1)
xtb1=torch.cat((x1,t_b),axis=1)
xtb2=torch.cat((x2,t_b),axis=1)

#initial condition data
t1=torch.zeros(n_b,1)
xi=torch.cat((x_b,t1),axis=1)
# The maximum number of iterations to do
epochs =5000
alpha=1
lbfgs=torch.optim.LBFGS(N.parameters(), lr=0.001, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
for epoch in range(epochs):
    running_loss = 0.0
    xi=xi[torch.randperm(xi.size()[0])]
    xt=xt[torch.randperm(xt.size()[0])]
    xtb1=xtb1[torch.randperm(xtb1.size()[0])]
    xtb2=xtb2[torch.randperm(xtb2.size()[0])]
    def closure():
    # Randomly pick n_batch random x's:
    
    # x_mean=torch.mean(x)
    # x_var=torch.var(x)
    # x_norm=normalize(x)
    #x_norm=x
    #print(x_norm)
    # Zero-out the gradient buffers
    # adam.zero_grad()
        
        lbfgs.zero_grad()
    # Evaluate the loss
        loss1 = residue_loss(xt.float())
        loss2= boundary_loss(xtb1.float(),xtb2.float())
        loss3=alpha*initial_loss(xi.float())
        l=(loss1+loss2+loss3)
        # Calculate the gradients
        l.backward()
        # Print the iteration number
        # if epoch % 100 == 99:
        #     print(epoch+1)
        #     print("residue loss=",loss1.item(),"bc loss=",loss2.item(),"ic loss=",loss3.item(),"overall_loss=",l.item())
        return l
    # Update the network
    # adam.step()
    lbfgs.step(closure)
    loss = closure()
    running_loss += loss.item()

    print(f"Epoch: {epoch + 1:02}/{epochs} Loss: {running_loss:.5e}")
    
        

 # predict u(t,x) distribution
num_test_samples=1000
t_flat = np.linspace(lb_t, ub_t, num_test_samples)
x_flat = np.linspace(lb_x, ub_x, num_test_samples)
tTEST, xTEST = np.meshgrid(t_flat, x_flat)
txTEST = torch.tensor(np.stack([tTEST.flatten(), xTEST.flatten()], axis=-1))
u = N(txTEST.float())
u = u.reshape(tTEST.shape)

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(7,4))
gs = Gridspec.GridSpec(2, 3)
plt.subplot(gs[0, :])
plt.pcolormesh(tTEST, xTEST, u.detach().numpy(), cmap='rainbow')
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(-1, 1)
# plot u(t=const, x) cross-sections
t_cross_sections = [0.25, 0.5, 0.75]
for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    tx1 = torch.tensor(np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1))
    u = N(tx1.float())
    plt.plot(x_flat, u.detach().numpy())
    plt.title('t={}'.format(t_cs))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
plt.tight_layout()
plt.show()
