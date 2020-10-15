import torch
import numpy as np
import cvxpy as cp
from PI import policy_iteration
import itertools



def c(delta,Ns,Na,n):
    delta_prime = delta/(4*Ns*Na*(n**2))
    return np.log(4/delta_prime)

def Equation219(N_samples,Ns,Na,gamma): # equation 219 (to be multiplied by c_n later)
    one = np.ones((Ns*Na,1))
    Eq219 = (7/(3*(1-gamma)))*cp.inv_pos(N_samples-one)
    return Eq219

def Equation218(N_samples,Ns,Na,R,P,gamma,V,epsilon): # right hand side of equation 218 (to be multiplied by sqrt(c_n) later)
    Eq218 = cp.Variable((Ns*Na,1))
    R = R.numpy()
    factor = np.zeros((Ns*Na,1))
    X = V.reshape(1,1,Ns).repeat(Ns,Na,1)
    Y = torch.mul(P,X)
    Var = torch.abs(torch.mul(P,X**2).sum(dim=2) - Y.sum(dim=2)**2) # we use torch.abs() because when the variance is null, sometimes the result is -x \times 10**(-7)
    Var = np.array(Var.numpy())
    for s,a in itertools.product(range(Ns),range(Na)):
        factor[a+s*Na,0] = np.sqrt(2*R[s,a]*(1-R[s,a])) + gamma*np.sqrt(2*Var[s,a]) + gamma*np.sqrt(2)*epsilon
        if np.isnan(factor[a+s*Na,0]):
            print("Some term in Equation (218) of BESPOKE is Nan")
            print("R", R)
            print("P", P)
            print("V",V)
            print("Var", Var)
            return
#     print("factor",factor)       
    Eq218 = cp.multiply(factor,cp.inv_pos(cp.sqrt(N_samples)))
    return Eq218
    
def CIplusBK(N_samples, n_max,Ns,Na,R,P,gamma,V,delta,epsilon): # \widehat{CI}^{k+1} + B_k of equation218
    d = Ns*Na+Ns+1 
    Eq218 = Equation218(N_samples,Ns,Na,R,P,gamma,V,epsilon)
    Eq219 = Equation219(N_samples,Ns,Na,gamma)
    CIBK = np.sqrt(c(delta,Ns,Na,n_max))*Eq218 + c(delta,Ns,Na,n_max)*Eq219
    return CIBK
    
def SolveProgram(phi, pi ,V, n_max,delta,epsilon):
    Ns = phi.Ns
    Na = phi.Na
    gamma = phi.gamma
    P = phi.P
    R = phi.R
    pi, V, _ = policy_iteration(phi)
    #constant
    Cp = 1/625
    # dimension of the problem
    d = Ns*Na+Ns+1 
    # Variables 
    N_samples = cp.Variable((Ns*Na,1),nonneg=True) # already contains constraint N_samples >= 0 
    y = cp.Variable((Ns+2,1))
    
    # Problem parameters
    A = np.zeros((Ns+2,d))
    b = np.zeros((Ns+2,1))
    
    # setting A matrix of equation 217
    E = np.zeros((Ns,Ns*Na)) # marginalization matrix
    for i in range(Ns):
        E[i, i*Na:i*Na+Na] = np.ones((1,Na))
    P_k = np.zeros((Ns*Na,Ns))
    R_k = np.zeros((Ns*Na,1))
    V_k = np.zeros((Ns,1))
    for s in range(Ns):
        V_k[s] = V[s].numpy()
        for a in range(Na):
            R_k[a+s*Na] = R[s,a].numpy()
            for s_prime in range(Ns) :
                P_k[a+s*Na,s_prime] = P[s,a,s_prime].numpy()
        
    A[:Ns,:Ns*Na] = E - gamma*np.transpose(P_k)
    A[:Ns,Ns*Na:Ns*Na+Ns] = -np.eye(Ns)
    A[Ns,Ns*Na:Ns*Na+Ns] = np.ones((1,Ns))
    A[Ns+1, :Ns*Na] = np.transpose(R_k)
    A[Ns+1, Ns*Na:Ns*Na+Ns] = -np.transpose(V_k)
    A[Ns+1,Ns*Na+Ns] = -1
    
    # setting b vector of equation 217
    b[Ns,0] = 1
    b[Ns+1,0] = -20*epsilon
    
    # setting c_vector of equation 217
    # for technical cvxpy considerations (DCP rules) ,we define it as free variable
    # then set the costraint c_vec[:Ns*Na,0] = CIBK later
    c_vec = cp.Variable((d,1), nonneg=True)
    #CIBK = CIplusBK(N_samples, n_max,Ns,Na,R,P,gamma,V,delta,epsilon) 
    Eq218 = Equation218(N_samples,Ns,Na,R,P,gamma,V,epsilon)
    Eq219 = Equation219(N_samples,Ns,Na,gamma)
    CIBK = np.sqrt(c(delta,Ns,Na,n_max))*Eq218 + c(delta,Ns,Na,n_max)*Eq219
    # Objective
    objective = np.transpose(b)@y
    #Constraints
    constraints = [cp.sum(N_samples)-n_max <= 0, CIBK -c_vec[:Ns*Na] <= 0 , c_vec - np.transpose(A)@y <= 0]
    # problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    opt = prob.solve(solver=cp.SCS)
#     if opt is None:
#         print("Nsamples",N_samples.value)
#         print("Eq218",Equation218(N_samples,Ns,Na,R,P,gamma,V,epsilon))
#         print("Eq219",Equation219(N_samples,Ns,Na,gamma))
    solution = N_samples.value
    return opt, solution