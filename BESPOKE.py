import torch
import time
import numpy as np
import itertools
from PI import policy_iteration
from environment import MDP, uniform_initial_estimate,update
from Bespoke_Program import SolveProgram



def n_min(Ns,Na,gamma,delta):
    delta_prime = delta/(4*Ns*Na)
    n = 2*(625**2)* (Ns**2)*Na*(gamma**2)* np.log(4/delta_prime)/((1-gamma)**2) 
    return n

def BESPOKE(phi,delta,accuracy):
    Ns = phi.Ns
    Na = phi.Na
    gamma = phi.gamma
    P = phi.P
    R = phi.R
    epsilon = 1/(1-gamma) # accuracy initialization
    
    # Algorithm's constants
    Cp = 1/625
    t0 = time.time()
    N = n_min(Ns,Na,gamma,delta)
    phi_hat = uniform_initial_estimate(phi, N)
    #print("time to form initial MDP estimate", time.time()-t0)
    pi_hat, V_hat, _ = policy_iteration(phi_hat)
    N_visits = N*torch.ones(Ns,Na)
    t = Ns*Na*N
    while epsilon > accuracy:
        #Optimal value of the convex program
        n_max = 1
        opt, N_samples = SolveProgram(phi_hat,pi_hat, V_hat, n_max, delta, epsilon) 
        while (opt is None) or (opt + 2*Cp*epsilon > epsilon/2) :
            n_max = 2*n_max
            #print("epsilon : ", epsilon, " n_max : ", n_max)
            opt, N_samples = SolveProgram(phi_hat,pi_hat, V_hat, n_max, delta, epsilon) 
        for s,a in itertools.product(range(Ns),range(Na)):
            N = int(N_samples[a + s*Na])
            if N > 0:
                rewards, transitions = phi.multiple_samples(s,a,N)
                phi_hat = update(phi_hat,s,a,rewards,transitions,N,N_visits[s,a])
                N_visits[s,a] += N
                t+= N
        pi_hat, V_hat, _ = policy_iteration(phi_hat)
        epsilon = epsilon/2
    return pi_hat, t