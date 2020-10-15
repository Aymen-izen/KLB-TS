###SIMPLE MDP ENVIRONMENT
import torch
import numpy as np
from torch.distributions import Bernoulli
import itertools
class MDP:
#Infinite horizon discounted MDP   
    def __init__(self,Ns,Na,P,R,gamma=0.9):
        self.Ns = Ns
        self.Na = Na
        assert P.shape == (Ns,Na,Ns)
        self.P = P
        assert R.shape == (Ns,Na)
        self.R = R
        self.gamma = gamma
        self.state = torch.multinomial(torch.ones(Ns),1).item()
        
    
    def sample(self,state,action):
        reward = Bernoulli(self.R[state,action].item()).sample()
        next_state = torch.multinomial(self.P[state,action],1).item()
        self.state = int(next_state)
        return  reward,self.state
    
    def multiple_samples(self,state,action,N):
        rewards = np.random.choice([0,1], size=N, p=[1-self.R[state,action].item(),self.R[state,action].item()])
        next_states = np.random.choice(range(self.Ns),size=N, p=self.P[state,action].numpy())
        unique, counts = np.unique(next_states, return_counts=True)
        dic = dict(zip(unique, counts))
        transitions = []
        for s_prime in range(self.Ns):
            if s_prime in dic.keys():
                transitions.append(dic[s_prime])
            else:
                transitions.append(0)
        return  torch.DoubleTensor(rewards).mean(), torch.DoubleTensor(transitions)
    
    def reset(self,visits=None):
        self.state = int(torch.multinomial(torch.ones(Ns),1).item())
        if not (visits is None):
            self.state = int(torch.multinomial(1/visits,1).item())
        return self.state
    
##FUNCTIONS FOR GENERATING RANDOM MDPs 
def Random_transitions(Ns,Na) : 
    P = torch.zeros(Na,Ns,Ns, dtype=torch.float64)
    for a in range(Na):
        Pa = torch.rand(Ns,Ns, dtype=torch.float64)
        Pa = Pa/torch.sum(Pa,axis=1)[:,None]
        P[a] = Pa
    P = P.permute(1,0,2)
    return P
def Random_rewards(Ns,Na):
    R = torch.rand(Ns,Na, dtype=torch.float64)
    return R
def generate_mdp(Ns,Na,gamma=0.9) :
    P = Random_transitions(Ns,Na)
    R = Random_rewards(Ns,Na)
    mdp = MDP(Ns,Na,P,R,gamma)
    return mdp

#we draw N samples from every state-action pair to construct an initial empirical mdp
def uniform_initial_estimate(mdp,N):
    Ns = mdp.Ns
    Na = mdp.Na
    gamma = mdp.gamma
    R = torch.zeros(Ns, Na,dtype=torch.float64)
    P = torch.zeros(Ns,Na,Ns,dtype=torch.float64)
    for s,a in itertools.product(range(Ns),range(Na)):
        rewards, transitions =  mdp.multiple_samples(s,a,N)
        R[s,a] = rewards.mean()
        P[s,a] =  transitions/N
    phi_hat = MDP(Ns,Na,P,R,gamma)
    return phi_hat

#UPDATING EMPIRICAL MDP FROM SAMPLES
def update(mdp,s,a,rewards,transitions,N,N_visits) :
    Ns = mdp.Ns
    Na = mdp.Na
    gamma = mdp.gamma
    P = mdp.P
    R = mdp.R
    R[s,a] = (R[s,a]*N_visits+rewards.sum())/(N_visits+N)
    P[s,a,:] = (P[s,a,:]*N_visits+transitions)/(N_visits+N)
    new_mdp = MDP(Ns,Na,P,R,gamma)
    return new_mdp
    