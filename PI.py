###POLICY ITERATION FUNCTIONS TO COMPUTE THE EMPIRICAL OPTIMAL POLICY
import torch
import itertools
def Greedy(R,gamma,P,V) :
    Ns = R.size()[0]
    Na = R.size()[1]
    Q = torch.zeros(Ns,Na,dtype=torch.float64)  
    for s,a in itertools.product(range(Ns),range(Na)):
        Q[s,a]= R[s,a] + gamma*torch.mm(P[s,a].reshape(1,-1),V).item()
    V, pi = torch.max(Q,dim=1)
    V = V.reshape(-1,1)
    return pi,V,Q
def Value(pi,R,gamma,P) :
    Ns = R.size()[0]
    P_pi = torch.zeros(Ns,Ns,dtype=torch.float64)
    R_pi = torch.zeros(Ns,1,dtype=torch.float64)
    for s in range(Ns):
        R_pi[s] = R[s,pi[s]]
        for s_prime in range(Ns):
            P_pi[s,s_prime] = P[s,pi[s],s_prime] 
    V = torch.solve(R_pi,torch.eye(Ns,dtype=torch.float64) -gamma*P_pi).solution
    return V
def policy_iteration(mdp) :
    Ns = mdp.Ns
    Na = mdp.Na
    gamma = mdp.gamma
    P = mdp.P
    R = mdp.R
    pi = torch.randint(low=0, high=Na,size=(Ns,))
    V = Value(pi,R,gamma,P)
    while True :
        pi_1,V_1,Q = Greedy(R,gamma,P,V)
        if (pi_1 == pi).all() or torch.abs(V_1 - V).max()< 1e-5:
            pi = pi_1
            V = V_1
            return pi, V, Q
        pi = pi_1
        V =  V_1 #Value(pi,R,gamma,P)