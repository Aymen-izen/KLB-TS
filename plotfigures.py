import pickle
import numpy as np
from PI import policy_iteration
from sampling import optimal_allocation
from IPython.display import Math, Latex
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)


def load_compare_save(mdp_filename, experiment_filename, DELTAS, ALG_NAMES, figure_name, log_scale=True):
    mdp_file = open(mdp_filename, 'rb')
    phi = pickle.load(mdp_file)
    mdp_file.close()
    ### LOAD Experiment results
    data_file = open(experiment_filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    #Theoritical U(\phi) 
    Ns = phi.Ns
    Na = phi.Na
    gamma = phi.gamma
    pi, V, Q = policy_iteration(phi)
    _,U,_,_,_,_ = optimal_allocation(phi,pi, V, Q)
    plt.figure(figsize=(10,8))
    confidences = np.array([np.log10(1/delta) for delta in DELTAS])
    if log_scale:
        for alg_name in ALG_NAMES:
            mean_SC = np.mean(np.log(data[alg_name]["complexity"]), axis=1)
            std_SC = np.std(np.log(data[alg_name]["complexity"]), axis=1)
            plt.plot(confidences, mean_SC, label=alg_name)#,color="b")
            plt.fill_between(confidences, mean_SC - 2 * std_SC, mean_SC + 2 * std_SC, alpha=0.15)
        theoretic_SC = np.log(np.array([U*np.log(1/delta) for delta in DELTAS]))
        plt.plot(confidences,theoretic_SC, label ="asymptotic bound", color ="r")
        plt.xlabel(r'$\log(1/\delta)$',fontsize=38)
        plt.ylabel(r'$\log(\textrm{Sample Complexity})$',fontsize=36)
        plt.legend(fontsize=32)
        plt.savefig(figure_name+".png") #eps", format='eps', dpi=1200)
        plt.show()
    else:
        plt.rc('xtick',labelsize=30)
        plt.rc('ytick',labelsize=20)
        mean_SC = np.mean(data["KLB-TS"]["complexity"], axis=1)
        std_SC = np.std(data["KLB-TS"]["complexity"], axis=1)
        plt.plot(confidences, mean_SC, label="KLB-TS")#,color="b")
        plt.ticklabel_format(axis="y", style="sci")
        plt.fill_between(confidences, mean_SC - 2 * std_SC, mean_SC + 2 * std_SC, alpha=0.15)
        theoretic_SC = np.array([U*np.log(1/delta) for delta in DELTAS])
        plt.plot(confidences,theoretic_SC, label ="asymptotic bound", color ="r")
        plt.ticklabel_format(axis="y", style="sci")
        plt.xlabel(r'$\log(1/\delta)$',fontsize=32)
        plt.ylabel(r'$\textrm{Sample Complexity}$',fontsize=32)
        plt.legend(fontsize=32)
        plt.savefig(figure_name+".png") #eps", format='eps', dpi=1200)
        plt.show()
        return

def BESPOKE_SCvsNmin(mdp_filename, experiment_filename, DELTAS, figure_name):
    from BESPOKE import n_min
    mdp_file = open(mdp_filename, 'rb')
    phi = pickle.load(mdp_file)
    mdp_file.close()
    ### LOAD Experiment results
    data_file = open(experiment_filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    #Theoritical U(\phi) 
    Ns = phi.Ns
    Na = phi.Na
    gamma = phi.gamma
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    plt.figure(figsize=(10,8))
    confidences = np.log(np.array([1/delta for delta in DELTAS]))
    mean_SC = np.mean(data["BESPOKE"]["complexity"], axis=1)
    std_SC = np.std(data["BESPOKE"]["complexity"], axis=1)
    Initial_SC = np.array([n_min(Ns,Na,gamma,delta) for delta in DELTAS])
    plt.plot(confidences,-np.log(1-Initial_SC/mean_SC), label ="BESPOKE Min samples vs Sample Complexity", color ="r")
    plt.xlabel(r'$\log(1/\delta)$',fontsize=32)
    plt.ylabel(r'$-\log(1- \frac{n_{min}}{\displaystyle{\tau}})$',fontsize=32)
    plt.savefig(figure_name+".png") #eps", format='eps', dpi=1200)
    plt.show()
