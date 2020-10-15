import numpy as np
import torch
import itertools
from PI import policy_iteration
import cvxpy as cp

## COMPUTING IMPORTANCE INDICES OF STATE-ACTION PAIRS USED IN THE FORMULA OF THE OPTIMAL ALLOCATION VECTOR
def AAstar(mdp,pi, V, Q) :
    Ns = mdp.Ns
    Na = mdp.Na
    gamma = mdp.gamma
    P = mdp.P
    R = mdp.R
    
    # matrix of gaps rounded to 6 decimals 
    delta_phi =  torch.round((V-Q) * 10**6) / (10**6) 
    non_zero = torch.where(delta_phi !=0, delta_phi, (2/(1-gamma))*torch.ones(Ns,Na,dtype=torch.float64)) # looking for non null gaps
    delta_min = torch.min(non_zero) # minimum gap
    if delta_min==0: # pathological case where all gaps are zero
        delta_min = 1/(1-gamma)
    
    #  matrix of Variance of next-state value function
    X = V.reshape(1,1,Ns).repeat(Ns,Na,1)
    Y = torch.mul(P,X)
    Var = torch.mul(P,X**2).sum(dim=2) - Y.sum(dim=2)**2
    #matrix of maximum deviation of next-state value function
    MD = torch.max(V.max()-Y.sum(dim=2),Y.sum(dim=2) -V.min())
    
    
    # Bound Terms
    T1 =  2/(delta_phi**2) # matrix of the first term T_1(s,a) used in the sampling and stopping rules 
    T1 = torch.where(T1 != float("inf"), T1,torch.zeros(Ns,Na, dtype=torch.float64))
    T1 = torch.where(torch.isnan(T1),torch.zeros(Ns,Na,dtype=torch.float64),T1)
    Z = torch.stack((16*Var/(delta_phi**2),6*(MD/delta_phi)**(4/3)))
    T2 = torch.max(Z,dim=0).values # matrix of the second term T_2(s,a) used in the sampling and stopping rules
    T2 = torch.where(T2 != float("inf"), T2,torch.zeros(Ns,Na, dtype=torch.float64))
    T2 = torch.where(torch.isnan(T2),torch.zeros(Ns,Na, dtype=torch.float64),T2)
    T3 = 2/(((1-gamma)*delta_min)**2) # Third term  used in the sampling and stopping rules   
    
    # Fourth term used in the sampling and stopping rules 
    
    Var_max =  torch.DoubleTensor([0.]) # maximum variance of optimal (state,action) pairs
    MD_max = torch.DoubleTensor([0.]) # maximum maximum-deviation of optimal (state,action) pairs
    for s in range(Ns):  
        if Var[s,pi[s]] > Var_max :
            Var_max = Var[s,pi[s]]
        if MD[s,pi[s]] > MD_max :
            MD_max = MD[s,pi[s]]
    
    V1 = torch.max(torch.DoubleTensor([27/((delta_min**2)*((1-gamma)**3)), 8/(delta_min*((1-gamma)**2.5)), 14*(MD_max/(delta_min*(1-gamma)))**(4/3)]))
    V2 = torch.max(torch.DoubleTensor([16*Var_max/((delta_min*(1-gamma))**2),6*(MD_max/(delta_min*(1-gamma)))**(4/3)]))
    T4 = torch.min(V1,V2) 
    
    A = T1 + T2 #matrix of state-action relevance indices, useful to compute optimal sampling weights
    A_star = T3 + T4
    A_star = A_star*Ns
    return A, A_star,T1,T2,T3,T4


def optimal_allocation(mdp,pi, V, Q) : 
    Ns = mdp.Ns
    Na = mdp.Na
    gamma = mdp.gamma
    P = mdp.P
    R = mdp.R
    A,A_star,T1,T2,T3,T4 = AAstar(mdp,pi, V, Q)
#     A = torch.where(A != float("inf"), A, torch.zeros(Ns,Na))
#     A = torch.where(torch.isnan(A),torch.zeros(Ns,Na), A)
    SUM1 =  A.sum()# sum the A_sa but remember to remove infinity/Nan values of optimal pairs 
    if SUM1 ==0: # pathological case where all gaps are zero hence A==0 too (we replaced infty values by zero in T1 and T2)
        omega = (1/(Ns*Na))*torch.ones(Ns,Na,dtype=torch.float64)
        return omega, torch.DoubleTensor([float("inf")]),T1,T2,torch.DoubleTensor([float("inf")]),T4 
        
    SUM2 = torch.sqrt( SUM1*A_star)
    D = SUM1 + SUM2 #DENOMINATOR IN THE FORMULA OF OPTIMAL OMEGA
    omega = torch.zeros(Ns,Na, dtype=torch.float64)
    for s,a in itertools.product(range(Ns),range(Na)):
        omega[s,a] = A[s,a]/D
        if A[s,a] == 0: 
            #this means that delta_phi[s,a] is zero because the pair (s,a) is optimal
            omega[s,a] = SUM2/(D*Ns)
            
            
    U = SUM1 + A_star + 2*SUM2 #THEORETICAL UPPER-BOUND OF THE Algorithm (see proof of Corollary 1)
    U = 4*U
    return omega,U,T1,T2,T3,T4

#D_tracking rule (#In practice yield the same results as C-tracking)
def D_tracking(omega,N_visits, t):
    Ns = N_visits.size()[0]
    Na = N_visits.size()[1]
    Nsa = Ns * Na
    starved = False 
    for s,a in itertools.product(range(Ns),range(Na)):
        if N_visits[s,a] < t**(1/2) - Nsa/2 : # if some pair is starving, sample from it
            starved = True
            return s,a, starved
    
    # Otherwise sample from the pair whose number of visits is far behind its cumulated weights
    y = t*omega - N_visits
    idx = torch.where(y==torch.max(y), torch.ones(Ns,Na),torch.zeros(Ns,Na))
    idx = idx.nonzero()
    try:
        idx = idx[np.random.randint(0,idx.shape[0])] #if the argmax is not unique, break the tie randomly 
    except:
        print(omega)
        print(y)
        print(torch.where(y==torch.max(y), torch.ones(Ns,Na),torch.zeros(Ns,Na)))
        print(idx)
        print(idx.shape)
    s = idx[0].item()
    a = idx[1].item()
    return s,a,starved

