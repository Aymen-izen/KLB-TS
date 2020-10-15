import numpy as np
import math
import torch
import itertools
from scipy.special import lambertw as W
import warnings
warnings.filterwarnings("ignore")

# def Wbar(x):
#     return float(-W(-np.exp(-x),k=-1))
# def x1(delta,t):
#     return np.max(np.array([1+np.log(4/delta),1+np.log((2+2*np.log(t))/delta),Wbar(1+np.log(4*np.log(t)/delta))]))


def x(delta,n,m): # Threshold function for KL concentration from Johansson et al 2020
    return np.log(1/delta) + (m-1) + (m-1)*np.log(1+n/(m-1))


# useful numerical functions 
def hInv(x): # see Proposition 15 in (Kaufmann and Koolen 2018)
    return float(-W(-np.exp(-x),k=-1))
def hTilde(x):
    if x >= hInv(1/np.log(1.5)):
        a = hInv(x)
        return a*np.exp(1/a)
    else:
        return 1.5*(x-np.log(np.log(1.5)))
def tau(x):  # Threshold function for KL concentration of rewards, taken from Theorem 14 in (Kaufmann and Koolen 2018)
    z = hInv(1+x)+2*np.log((math.pi**2)/3)
    return 2*hTilde(z/2)


def y(delta,n): # Threshold function for KL concentration of rewards, taken from Theorem 14 in (Kaufmann and Koolen 2018)
    d = 4
    c = 3
    x = np.log(1/delta)
    return c*np.log(d+np.log(n)) + x + np.log(1+x+np.sqrt(2*x)) # x + np.log(1+x+np.sqrt(2*x)) is an approximation of tau(x) in their paper

def check_condition(pi_hat,N_visits,delta,T1_hat,T2_hat,T3_hat,T4_hat):
    T1_hat = T1_hat
    T2_hat = T2_hat
    T3_hat = T3_hat
    T4_hat = T4_hat
    Ns = N_visits.size()[0]
    Na = N_visits.size()[1]
    delta_prime = delta/(4*(Ns**3)*Na)
    M_12 = - torch.DoubleTensor([float("Inf")])
    M_34 = - torch.DoubleTensor([float("Inf")])
    #print(T1_hat,T2_hat,T3_hat,T4_hat)
    for s,a in itertools.product(range(Ns),range(Na)):
        if a != pi_hat[s]:
            n = N_visits[s,a]
            ThreshR = np.min([y(delta_prime,n),x(delta_prime,n,2)]) # both threshold functions are valid for the rewards so we use the minimum to optimize stopping
            frac = (torch.sqrt(T1_hat[s,a]*ThreshR)+ torch.sqrt(T2_hat[s,a]*x(delta_prime,n,Ns)))/(torch.sqrt(n))
            if frac > M_12:
                M_12 = frac
        else:
            n = N_visits[s,a]
            ThreshR = np.min([y(delta_prime,n),x(delta_prime,n,2)])
            frac = (torch.sqrt(T3_hat*ThreshR)+ torch.sqrt(T4_hat*x(delta_prime,n,Ns)))/(torch.sqrt(n))
            if frac > M_34 :
                M_34 = frac
    
    STOP = (M_12 + M_34 < 1)
    print("stopping term == ",(M_12 + M_34).item())
    return STOP
             
    