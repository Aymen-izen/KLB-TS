import torch
from environment import MDP, uniform_initial_estimate,update
from PI import policy_iteration
from sampling import AAstar, optimal_allocation, D_tracking, fast_D_tracking
from stopping import check_condition
import itertools
import time

def KLBTS(phi,delta= 1e-3,period = 1e3, speedup=True) :
    Ns = phi.Ns
    Na = phi.Na
    ## GROUNDTRUTH VARIABLES FOR CHECK-IN
    pi, Vr, Qr = policy_iteration(phi)
    omegar,_,T1,T2,T3,T4 = optimal_allocation(phi,pi, Vr, Qr)
    
    
    
    #### BEGINNING OF THE ALGO
    init_samples = 1
    phi_hat = uniform_initial_estimate(phi,init_samples)
    pi_hat, V, Q = policy_iteration(phi_hat)
    omega_hat,_,T1_hat,T2_hat,T3_hat,T4_hat = optimal_allocation(phi_hat,pi_hat, V, Q)
    N_visits = init_samples*torch.ones(Ns,Na,dtype=torch.float64)
    t = init_samples*Ns*Na
    LAST = init_samples*Ns*Na
    STOP = False
    CORRECT = False
    EARLIEST = 0
    t0 = time.time()
    while not STOP:
        if t%period == 0: # update sampling weights after every period to avoid unnecessary computation.
#             print("Period in ", time.time()-t0)
            print("Iteration n° ", t)
#             print("empirical frequencies vs optimal oracle allocation == \n ",torch.sqrt(torch.sum((omegar - N_visits/t)**2)))
#             print("empirical frequencies vs optimal oracle allocation == \n ", omegar - N_visits/t)
#             print("starving proportion", count/t)
#             print("Visits === \n", N_visits)
#             print("omega = \n",omega_hat)
            pi_hat, V_hat, Q_hat = policy_iteration(phi_hat)
            omega_hat,_,T1_hat,T2_hat,T3_hat,T4_hat = optimal_allocation(phi_hat,pi_hat, V_hat, Q_hat)
            #print("convergence of stopping rule terms :  \n", T1_hat/T1,T2_hat/T2,T3_hat/T3,T4_hat/T4)
            if (not CORRECT) and torch.all(torch.eq(pi, pi_hat)):
                EARLIEST = t
                CORRECT = True
#             t1 = time.time()
            STOP = check_condition(pi_hat,N_visits,delta,T1_hat,T2_hat,T3_hat,T4_hat)
#             print("checked in", time.time()-t1)
            t0 = time.time()
        s,a,starved = D_tracking(omega_hat,N_visits,t)
#         print("Tracking in ", time.time()-t1)
        t1 = time.time()
        rewards, transitions = phi.multiple_samples(s,a,1)
#         print("Sampled in ", time.time()-t1)
        t1 = time.time()
        phi_hat = update(phi_hat,s,a,rewards,transitions,1,N_visits[s,a])
#         print("Updated in ", time.time()-t1)
        N_visits[s,a] += 1
        t+=1        
 
    return pi_hat,t,EARLIEST
