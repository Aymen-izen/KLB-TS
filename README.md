# Implementation of KLB-TS algorithm (see https://arxiv.org/abs/2009.13405)


## Reproducing experiments:
The pickle folder contains .pkl files of the two MDPs used in the paper experiments. It also contains the .pkl file of classical example RIVERSWIM (see https://ieeexplore.ieee.org/document/1374179) However reverswim has a constant <img src="https://latex.codecogs.com/png.latex?\dpi{100} U(\phi) \sim 7 \times 10^8"/>, it might take some run time.
The notebook Experiments.ipynb shows how to generate random MDPs of a given size and how to run KLB-TS or BESPOKE on them.

## main.py:
You can run the command:   \textbf{python main.py -f path_to_mdp_file -o path_to_results_file -n N} to compare KLB-TS and BESPOKE. 
Change "path_to_mdp_file" to the path of the .pkl file where the mdp object is stored. 
The results will be stored at "path_to_results_file.pkl". 
"N" stands for the number of simulations per confidence level, used to compute the expected sample complexity.

Example:  \textbf{python main.py -f./pickle/MDP_EASY -o ./pickle/results_EASY -n 10}

## KLB-TS:
sampling.py: contains functions to compute the optimal sample allocation and the terms <img src="https://latex.codecogs.com/png.latex?\dpi{100}\(T_i)_{1%20\leq%20i%20\leq%204}"/> used in the stopping rule. We implemented the D-tracking (similar guarantees to C-tracking but much easier to implement :p, see http://proceedings.mlr.press/v49/garivier16a.pdf ) 
stopping.py: checks if the stopping condition is verified
environment.py: contains functions to generate random MDPs, construct initial estimate of the MDP, sample from the MDP and update MDP estimate after collecting observations.  





