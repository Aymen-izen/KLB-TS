# Standard data science libraries
import numpy as np
import torch
import itertools
# Visualization
import time
import matplotlib as mpl
#mpl.use('MacOSX')
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
#plt.style.use('bmh')
from environment import MDP, generate_mdp
from PI import policy_iteration
from sampling import optimal_allocation
from KLB_TS import KLBTS
from BESPOKE import BESPOKE
from IPython.display import display, Math, Latex
import pickle


if __name__== "__main__" :

    Ns = 10
    Na = 5
    gamma = 0.9
    phi = generate_mdp(Ns,Na,gamma=gamma)
    pi, V, Q = policy_iteration(phi)
    _,U,_,_,_,_ = optimal_allocation(phi,pi, V, Q)
    #computing delta_min of phi to reveal it to BESPOKE
    delta_phi = torch.zeros(Ns,Na,dtype=torch.float64)
    delta_phi = V - Q
    non_zero = torch.where(delta_phi !=0.0, delta_phi, (2/(1-gamma))*torch.ones(Ns,Na, dtype=torch.float64)) # looking for non null gaps
    delta_min = torch.min(non_zero) 
    minimax = Ns*Na/((delta_min**2)*((1-gamma)**3))
    print("KLB-TS asymptotic sample complexity (without the log delta) :", int(U.item()))
    print("Minimax sample complexity (without the log delta):" ,int(minimax.item()))


    name2 = "HARD_02" #  remember to change then name
    with open("MDP_"+name2+".pkl","wb") as file:
        pickle.dump(phi,file)

    N_experiments = 10
    DELTAS = [0.1,0.05,0.01,0.005,0.001]
    d = len(DELTAS)
    data={}
    accuracy = 0.9*delta_min # accuracy level of policy to be returned by bespoke (= epsilon)
    data["BESPOKE"] = { "complexity" : np.zeros((d,N_experiments)), "accurate": np.zeros((d,N_experiments))}
    data["KLB-TS"] = { "complexity" : np.zeros((d,N_experiments)), "accurate": np.zeros((d,N_experiments))}
    for k in range(d):
        delta = DELTAS[k]
        for n in range(N_experiments):
            print("Confidence level (delta) = ", delta, " Experiment n°", n+1)
            ##### KLB-TS part######
            t0=time.time()
            pi_hat,t,EARLIEST = KLBTS(phi,delta,period = 1e3)
            print("KLB-TS run time = ", time.time()-t0)
            data["KLB-TS"]["complexity"][k][n] = t
    #         data["KLB-TS"]["EARLIEST"][k][n] = EARLIEST
            if torch.all(torch.eq(pi, pi_hat)) :
                data["KLB-TS"]["accurate"][k][n] = 1 

            ###### BESPOKE part ######
            t0=time.time()
            pi_hat,t = BESPOKE(phi,delta,accuracy) # note that we have revealed delta_min to BESPOKE
            print("BESPOKE run time = ", time.time()-t0)
            data["BESPOKE"]["complexity"][k][n] = t

            if torch.all(torch.eq(pi, pi_hat)) :
                data["BESPOKE"]["accurate"][k][n] = 1  

    ## SAVE EXPERIMENT DATA
    data_file = open("experiment_"+name2+".pkl", "wb")
    pickle.dump(data, data_file)
    data_file.close()