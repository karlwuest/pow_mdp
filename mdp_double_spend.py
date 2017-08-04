import mdptoolbox
import numpy as np
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import Queue
from matplotlib.colors import LogNorm
import seaborn
seaborn.set(font_scale=2.3)
seaborn.set_style("whitegrid")
import sys

class State:
    def __init__(self, l_a, l_h, b_e, match="relevant"):
        self.length_a = l_a
        self.length_h = l_h
        self.blocks_e = b_e
        self.match = match

    def __hash__(self):
        return hash((self.length_a, self.length_h, self.blocks_e, self.match))

    def __eq__(self, other):
        try:
            return (self.length_a, self.length_h, self.blocks_e, self.match) == (other.length_a, other.length_h, other.blocks_e, other.match)
        except:
            return False

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return "(%d, %d, %d, %s)" % (self.length_a, self.length_h, self.blocks_e, self.match)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def optimal_strategy(p, k, stale, double_spend_value, max_blocks, gamma, cutoff, m_cost, lam=0):
    """
    p: attacker hashrate as fraction of total hashrate
    stale_rate: rate of stale blocks in honest network
    double_spend_value: the value that the attacker gains from a successful double spend, 1 = block reward
    max_blocks: maximum number of total mined blocks considered
    gamma: fraction of honest nodes that a block from the attacker reaches before a block from the honest network (when matching)
    cutoff: maximum length of either chain (needed for finite MDP)
    m_cost: cost of mining (per state transition)
    lam: hashrate of eclipsed miner

    implicitly: 
        q = (1 - p - lam) hashrate of honest network
        q*(1-stale) probability that the honest chain grows (in each step)
        q*stale probability that state stays the same
    """
    states = {}
    states_inverted = {}
    q = 1.0-p-lam
    
    match_cases = ["irrelevant", "relevant", "active"]

    # construct states
    states_counter = 0
    for l_a in xrange(cutoff + 1):
        for l_h in xrange(cutoff + 1):
            for b_e in xrange(l_a+1):
                if lam == 0 and b_e > 0:
                    break
                for match in match_cases:
                    state = State(l_a, l_h, b_e, match)
                    states[states_counter] = state
                    states_inverted[state] = states_counter
                    states_counter += 1

    # exit state
    exit_idx = states_counter
    states_counter += 1

    # build transition matrices
    P_adopt    = np.zeros(shape=(states_counter, states_counter))
    P_override = np.zeros(shape=(states_counter, states_counter))
    P_match    = np.zeros(shape=(states_counter, states_counter))
    P_wait     = np.zeros(shape=(states_counter, states_counter))
    P_exit     = np.zeros(shape=(states_counter, states_counter))
    
    # build reward matrices
    R_adopt    = np.zeros(shape=(states_counter, states_counter))
    R_override = np.zeros(shape=(states_counter, states_counter))
    R_match    = np.zeros(shape=(states_counter, states_counter))
    R_wait     = np.zeros(shape=(states_counter, states_counter))
    R_exit     = np.zeros(shape=(states_counter, states_counter))

    # never leave exit state
    P_exit[exit_idx, exit_idx] = 1
    P_adopt[exit_idx, exit_idx] = 1
    P_override[exit_idx, exit_idx] = 1
    P_match[exit_idx, exit_idx] = 1
    P_wait[exit_idx, exit_idx] = 1
    R_exit[exit_idx, exit_idx] = p - m_cost

    for state_idx, state in states.iteritems():
        l_a = state.length_a
        l_h = state.length_h
        b_e = state.blocks_e
        match = state.match
        
        # exit
        if l_a > k and l_a > l_h:
            P_exit[state_idx, exit_idx] = 1
            R_exit[state_idx, exit_idx] = l_a - b_e + double_spend_value - m_cost
        else:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_exit[state_idx, state_idx] = 1
            R_exit[state_idx, state_idx] = -100

        # adopt
        # attacker mines next block
        P_adopt[state_idx, states_inverted[State(1, 0, 0, "irrelevant")]] = p
        R_adopt[state_idx, states_inverted[State(1, 0, 0, "irrelevant")]] = 0 - m_cost 
        # eclipsed node mines next block
        if lam != 0:
            P_adopt[state_idx, states_inverted[State(1, 0, 1, "irrelevant")]] = lam 
            R_adopt[state_idx, states_inverted[State(1, 0, 1, "irrelevant")]] = 0 - m_cost 
        # network mines next block
        P_adopt[state_idx, states_inverted[State(0, 1, 0, "relevant")]] = q*(1-stale)
        R_adopt[state_idx, states_inverted[State(0, 1, 0, "relevant")]] = 0 - m_cost 
        # network mines stale block
        P_adopt[state_idx, states_inverted[State(0, 0, 0, "irrelevant")]] = q*stale
        R_adopt[state_idx, states_inverted[State(0, 0, 0, "irrelevant")]] = 0 - m_cost
        

        # override
        if l_a > l_h:
            payout = (l_h+1)*(l_a - b_e)//l_a
            new_b_e = b_e - (l_h+1 - payout)
            # attacker mines next block
            P_override[state_idx, states_inverted[State(l_a - l_h, 0, new_b_e, "irrelevant")]] = p
            R_override[state_idx, states_inverted[State(l_a - l_h, 0, new_b_e,"irrelevant")]] = payout - m_cost
            # eclipsed node mines next block
            if lam != 0:
                P_override[state_idx, states_inverted[State(l_a - l_h, 0, new_b_e + 1, "irrelevant")]] = lam
                R_override[state_idx, states_inverted[State(l_a - l_h, 0, new_b_e + 1,"irrelevant")]] = payout - m_cost
            # network mines next block
            P_override[state_idx, states_inverted[State(l_a-l_h-1, 1, new_b_e, "relevant")]] = q*(1 - stale)
            R_override[state_idx, states_inverted[State(l_a-l_h-1, 1, new_b_e, "relevant")]] = payout - m_cost
            # network mines stale block
            P_override[state_idx, states_inverted[State(l_a-l_h-1, 0, new_b_e, "irrelevant")]] = q*stale 
            R_override[state_idx, states_inverted[State(l_a-l_h-1, 0, new_b_e, "irrelevant")]] = payout - m_cost
        else:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_override[state_idx, state_idx] = 1
            R_override[state_idx, state_idx] = -100

        # only allow adopt, override or exit after cutoff
        if l_a == cutoff or l_h == cutoff:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_match[state_idx, state_idx] = 1
            R_match[state_idx, state_idx] = -100
            P_wait[state_idx, state_idx] = 1
            R_wait[state_idx, state_idx] = -100
            continue

        # match
        if match == "relevant" and l_a >= l_h and l_h > 0:
            payout = (l_h)*(l_a - b_e)//l_a
            new_b_e = b_e - (l_h - payout)
            # attacker mines next block
            P_match[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "active")]] = p
            R_match[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "active")]] = 0 - m_cost
            # eclipsed node mines next block
            if lam != 0:
                P_match[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "active")]] = lam
                R_match[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "active")]] = 0 - m_cost
            # network mines next block on chain released by attacker
            P_match[state_idx, states_inverted[State(l_a - l_h, 1, new_b_e, "relevant")]] = gamma*q*(1 - stale)
            R_match[state_idx, states_inverted[State(l_a - l_h, 1, new_b_e, "relevant")]] = payout - m_cost
            # network mines next block on honest chain
            P_match[state_idx, states_inverted[State(l_a, l_h +  1, b_e, "relevant")]] = (1-gamma)*q*(1 - stale)
            R_match[state_idx, states_inverted[State(l_a, l_h +  1, b_e, "relevant")]] = 0 - m_cost
            # network mines stale block
            P_match[state_idx, states_inverted[State(l_a, l_h, b_e, "active")]] = q*stale 
            R_match[state_idx, states_inverted[State(l_a, l_h, b_e, "active")]] = 0 - m_cost 
        else:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_match[state_idx, state_idx] = 1
            R_match[state_idx, state_idx] = -100
            
        # wait
        if match == "active" and l_a >= l_h and l_h >0:
            payout = (l_h)*(l_a - b_e)//l_a
            new_b_e = b_e - (l_h - payout)
            # attacker mines next block
            P_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "active")]] = p
            R_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "active")]] = 0 - m_cost
            # eclipsed node mines next block
            if lam != 0:
                P_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "active")]] = lam
                R_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "active")]] = 0 - m_cost
            # network mines next block on chain released by attacker
            P_wait[state_idx, states_inverted[State(l_a - l_h, 1, new_b_e, "relevant")]] = gamma*q*(1 - stale)
            R_wait[state_idx, states_inverted[State(l_a - l_h, 1, new_b_e, "relevant")]] = payout - m_cost
            # network mines next block on honest chain
            P_wait[state_idx, states_inverted[State(l_a, l_h +  1, b_e, "relevant")]] = (1-gamma)*q*(1 - stale)
            R_wait[state_idx, states_inverted[State(l_a, l_h +  1, b_e, "relevant")]] = 0 - m_cost
            # network mines stale block
            P_wait[state_idx, states_inverted[State(l_a, l_h, b_e, "active")]] = q*stale 
            R_wait[state_idx, states_inverted[State(l_a, l_h, b_e, "active")]] = 0 - m_cost 
        else:
            # attacker mines next block
            P_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "irrelevant")]] = p
            R_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e, "irrelevant")]] = 0 - m_cost 
            # eclipsed node mines next block
            if lam != 0:
                P_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "irrelevant")]] = lam
                R_wait[state_idx, states_inverted[State(l_a + 1, l_h, b_e+1, "irrelevant")]] = 0 - m_cost 
            # network mines next block
            P_wait[state_idx, states_inverted[State(l_a, l_h + 1, b_e, "relevant")]] = q*(1 - stale)
            R_wait[state_idx, states_inverted[State(l_a, l_h + 1, b_e, "relevant")]] = 0 - m_cost
            # network mines stale block
            P_wait[state_idx, states_inverted[State(l_a, l_h, b_e, "irrelevant")]] = q*stale 
            R_wait[state_idx, states_inverted[State(l_a, l_h, b_e, "irrelevant")]] = 0 - m_cost 


    P = [P_wait, P_adopt, P_override, P_match, P_exit]
    R = [R_wait, R_adopt, R_override, R_match, R_exit]
    for i,p in enumerate(P):
        try:
            mdptoolbox.util.checkSquareStochastic(p)
        except:
            print "not stochastic:", i
            #for l in p:
                #print l
    #mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.999, max_blocks)
    #mdp = mdptoolbox.mdp.ValueIteration(P, R, 0.999)
    #mdp = mdptoolbox.mdp.QLearning(P, R, 0.999)
    mdp = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)
    #mdp.setVerbose()
    mdp.run()
    return mdp, states

def state_graph(states, transitions, policy):
    policy_colors = ["blue", "red", "grey", "yellow", "green"]
    G = nx.DiGraph()
    q = Queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        G.add_node(states[state_idx], color=policy_colors[pol], style="filled")
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    G.add_edge(states[state_idx], "exit", label=p)
                else:
                    G.add_edge(states[state_idx], states[i], label=p)
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return G


def state_table(states, transitions, policy, cutoff):
    policy_letter = ["w", "a", "o", "m", "e"]
    q = Queue.Queue()
    table = [[['*']*3]*cutoff]*cutoff
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        state = states[state_idx]
        if state.match == "irrelevant":
            fork_idx = 0
        elif state.match == "relevant":
            fork_idx = 1
        elif state.match == "active":
            fork_idx = 2
        else:
            raise ValueError('Invalid fork label')
        table[state.length_a][state.length_h][fork_idx] = policy_letter[pol]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    pass
                else:
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return table

def print_table(table):
    l = len(table)
    print r"\begin{tabular}{@{}c|"+ l*'c' +r"@{}}"
    print r"\toprule"
    print '& ' + ' & '.join(str(x) for x in range(l)) + r'\\'
    for idx, line in enumerate(table):
        print str(idx) + '& ' +  ' & '.join([''.join(x) for x in line]) + r'\\'
        if idx < l-1:
            print r'\midrule'
    print r"\bottomrule"
    print r"\end{tabular}"

def hashrate_k_plot(stale, gamma, cost, cutoff=20):
    ps = np.arange(0.025, 0.5, 0.025)
    ks = np.arange(0, 13, 1)
    ds = np.zeros([len(ps), len(ks)])
    max_val = 1000000000
    eps = 0.1
    for p_idx, p in enumerate(ps):
        for k_idx, k in enumerate(ks):
            m_cost = cost*p
            double_spend_value = max_val/2
            if k_idx > 0 and ds[p_idx, k_idx-1] > max_val - eps:
                double_spend_value = max_val
            last_value = 0
            diff = max_val/2
            lower = 0
            upper = max_val
            while diff > eps:
                print p, k, double_spend_value
                mdp,states = optimal_strategy(p, k, stale, double_spend_value, None, gamma,cutoff, m_cost)
                G = state_graph(states, mdp.P, mdp.policy)
                diff = abs(last_value - double_spend_value)
                if G.has_node("exit"):
                    print "exit"
                    last_value = double_spend_value
                    upper = double_spend_value
                    double_spend_value -= (double_spend_value - lower)/2.0
                else:
                    last_value = double_spend_value
                    lower = double_spend_value
                    double_spend_value += (upper - double_spend_value)/2.0
            ds[p_idx, k_idx] = last_value
    np.save("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f.npy" % (cutoff, gamma, stale, cost), ds)
    plt.pcolor(ps, ks, ds.T, norm=LogNorm(vmin=ds.min(), vmax=ds.max()))
    cbar = plt.colorbar()
    cbar.set_label("double spend value")
    plt.ylabel("k")
    plt.xlabel("p")
    plt.savefig("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f.png" % (cutoff, gamma, stale, cost))
    plt.close()

def hashrate_lam_plot(stale, gamma, cost, cutoff=20, k=6):
    ps = np.arange(0.025, 0.5, 0.025)
    lams = np.arange(0.0, 0.5, 0.025)
    ds = np.zeros([len(ps), len(lams)])
    max_val = 1000000000
    eps = 0.1
    for p_idx, p in enumerate(ps):
        for lam_idx, lam in enumerate(lams):
            m_cost = cost*p
            double_spend_value = max_val/2
            last_value = 0
            diff = max_val/2
            lower = 0
            upper = max_val
            while diff > eps:
                print p, lam, double_spend_value
                mdp,states = optimal_strategy(p, k, stale, double_spend_value, None, gamma,cutoff, m_cost, lam=lam)
                G = state_graph(states, mdp.P, mdp.policy)
                diff = abs(last_value - double_spend_value)
                if G.has_node("exit"):
                    print "exit"
                    last_value = double_spend_value
                    upper = double_spend_value
                    double_spend_value -= (double_spend_value - lower)/2.0
                else:
                    last_value = double_spend_value
                    lower = double_spend_value
                    double_spend_value += (upper - double_spend_value)/2.0
            ds[p_idx, lam_idx] = last_value
    np.save("hashrate_om_double_spend_co%dk%dg%.2fs%.4fc%.2f.npy" % (cutoff,k, gamma, stale, cost), ds)
    plt.pcolor(ps, lams, ds.T, norm=LogNorm(vmin=ds.min(), vmax=ds.max()))
    cbar = plt.colorbar()
    cbar.set_label("double spend value")
    plt.ylabel("Eclipsed node hashrate $\omega$")
    plt.xlabel(r"Adversary hashrate $\alpha$")
    fig = plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(10,7)
    plt.savefig("hashrate_om_double_spend_co%dk%dg%.2fs%.4fc%.2f.eps" % (cutoff,k, gamma, stale, cost))
    plt.close()

                        
def markov_chain(states, transitions, policy):
    import pykov
    T = pykov.Chain()
    q = Queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    start = pykov.Vector({states[0]:1})
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    T[(states[state_idx], "exit")] = p
                    T[("exit", "exit")] = 1
                else:
                    T[(states[state_idx], states[i])] = p
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return T, start

def markov_matrix(transitions, policy):
    n = len(policy)
    P = np.zeros((n, n))
    q = Queue.Queue()
    visited = [False]*n
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                P[state_idx, i] = p
                if not visited[i]:
                    q.put(i)
                    visited[i] = True
    return P

def exp_blocks_needed(p, k, stale, gamma, double_spend_value, m_cost, cutoff=20):
    mdp, states = optimal_strategy(p, k, stale, double_spend_value, None, gamma, cutoff, m_cost)
    P = markov_matrix(mdp.P, mdp.policy)
    l = len(mdp.policy)
    Q = P[0:l-1,0:l-1]
    I = np.eye(l-1)
    N = np.linalg.inv(I - Q)
    ones = np.ones((l-1,1))
    t = N.dot(ones)
    return t[0]


def main():
    l = len(sys.argv)
    if l >=3:
        cost = float(sys.argv[1])
        gamma = float(sys.argv[2])
    else:
        print "Not enough arguments"
        print "Usage: %s <gamma> <cost>" %sys.argv[0]
        return
    stale = 0.0041
    k = 6
    hashrate_k_plot(stale, gamma, cost, cutoff=20):

if __name__=="__main__":
    main()
