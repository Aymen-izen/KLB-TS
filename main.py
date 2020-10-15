# Standard data science libraries
import numpy as np
import torch
import itertools
# Visualization
import time
#plt.style.use('bmh')
from environment import MDP, generate_mdp
from PI import policy_iteration
from sampling import optimal_allocation
from KLB_TS import KLBTS
from BESPOKE import BESPOKE
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", help="name of the .pkl file containing the MDP object")
parser.add_argument("-o","--output", help="name of the .pkl file where the experiments results will be stored")
parser.add_argument("-n","--simulations", help="number of simulations per confidence level (for averaging sample complexity)")
args = parser.parse_args()

if __name__== "__main__" :
    
    name2 = "HARD_02"
    mdp_file = open(args.filename+".pkl", 'rb')
    phi = pickle.load(mdp_file)
    Ns = phi.Ns
    Na = phi.Na
    gamma = phi.gamma
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


    N_simulations = int(args.simulations)
    DELTAS = [0.1,0.05,0.01,0.005,0.001]
    d = len(DELTAS)
    data={}
    accuracy = 0.9*delta_min # accuracy level of policy to be returned by bespoke (= epsilon)
    data["BESPOKE"] = { "complexity" : np.zeros((d,N_simulations)), "accurate": np.zeros((d,N_simulations))}
    data["KLB-TS"] = { "complexity" : np.zeros((d,N_simulations)), "accurate": np.zeros((d,N_simulations))}
    for k in range(d):
        delta = DELTAS[k]
        for n in range(N_simulations):
            print("Confidence level (delta) = ", delta, " Simulation n°", n+1)
            ###### BESPOKE part ###### 
            t0=time.time()
#             pi_hat,t = BESPOKE(phi,delta,accuracy) # note that we have revealed delta_min to BESPOKE
            delta_prime = delta/(4*Ns*Na)
            n_min = 2*(625**2)* (Ns**2)*Na*(gamma**2)* np.log(4/delta_prime)/((1-gamma)**2) 
            print("t == ", n_min)
            #to gain time, we only count BESPOKE's minimum number of samples as it's sample complexity (see lemma 16 in BESPOKE's paper)
            data["BESPOKE"]["complexity"][k][n] = n_min

#             if torch.all(torch.eq(pi, pi_hat)) :
            data["BESPOKE"]["accurate"][k][n] = 1  
            print("BESPOKE run time = ", time.time()-t0)
            
            ##### KLB-TS part######
            t0=time.time()
            pi_hat,t,EARLIEST = KLBTS(phi,delta,period = 1e3)
            print("KLB-TS run time = ", time.time()-t0)
            data["KLB-TS"]["complexity"][k][n] = t
    #         data["KLB-TS"]["EARLIEST"][k][n] = EARLIEST
            if torch.all(torch.eq(pi, pi_hat)) :
                data["KLB-TS"]["accurate"][k][n] = 1 


    ## SAVE EXPERIMENT DATA
    data_file = open(args.output+".pkl", "wb")
    pickle.dump(data, data_file)
    data_file.close()
