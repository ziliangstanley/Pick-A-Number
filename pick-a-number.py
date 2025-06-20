# Teaser problem about number guessing.

# T = 5, N = 100

# num states = T * N * N * (2 * N)
#
#

import cvxpy as cp
import gurobipy as gp
from collections import defaultdict
from typing import NamedTuple
import math
import numpy as np
from abc import ABC, abstractmethod


# case = 'LARGE'
case = 'MEDIUM'
# case = 'SMALL'
# case = 'TINY'

if case == 'LARGE':

    T = 5
    N = 100
    V = [100, 80, 60, 40, 20]

elif case == 'MEDIUM':

    T = 4
    N = 50
    V = [80, 60, 40, 20]
    # Naive deterministic 3.333333333333333
    # Naive randomized 3.75
    # GT optimal 10.4

elif case == 'SMALL':

    T = 3
    N = 30
    V = [30, 20, 10]
    # Naive deterministic 1.4285714285714284
    # Naive randomized 1.488095238095238
    # GT optimal 3.66666667

elif case == 'TINY':

    T = 1
    N = 5
    V = [10]


class Infoset(NamedTuple):
    t: int
    n_low: int
    n_high: int


class Sequence(NamedTuple):
    infoset: Infoset
    guess: int


class Strategy(ABC):
    @abstractmethod
    def get_behavioral(self, infoset: Infoset):
        pass


class NaiveStrat(Strategy):
    def __init__(self, randomized=True):
        self.randomized = randomized
        super().__init__()

    def get_behavioral(self, infoset: Infoset):
        if infoset.t == T-1:
            guesses = list(range(infoset.n_low, infoset.n_high+1))
            guesses_prob = [1./len(guesses)] * len(guesses)
        else:
            if self.randomized:
                # If even number of choices, pick one of the two center choices uar
                if (infoset.n_high-infoset.n_low+1) % 2 == 0:
                    guesses = [math.floor((infoset.n_high + infoset.n_low) / 2),
                               math.ceil((infoset.n_high + infoset.n_low) / 2)]
                    guesses_prob = [0.5, 0.5]
                else:
                    guesses = [(infoset.n_high + infoset.n_low)//2]
                    guesses_prob = [1.0]
            else:
                # Skewed selection.
                guesses = [(infoset.n_high + infoset.n_low)//2]
                guesses_prob = [1.0]

        return guesses, guesses_prob


class DictStrat(Strategy):
    def __init__(self, d):
        self.d = d
        super().__init__()

    def get_behavioral(self, infoset: Infoset):
        return self.d[infoset]


def eval_strat(strategy: Strategy):
    """
    Evaluate a strategy by computing the expected payoff for each possible true position
    and extracting the worst case payoff.
    """

    P = eval_pos(strategy)

    return min(P)


def eval_pos(strategy: Strategy):
    """
    Evaluate a strategy by computing the expected payoff for each possible true position.
    """
    P = []
    for truth in range(N):
        # Get payoff assuming true posiiton is truth and we are using
        # naive strategy (randomize between midpoints if range is even in size)

        mem = dict()

        def solve(infoset: Infoset):
            if infoset in mem:
                return mem[infoset]

            guesses, guesses_prob = strategy.get_behavioral(infoset)
            assert np.isclose(sum(guesses_prob), 1.0)

            exp_payoff = 0.
            for guess_id in range(len(guesses)):
                guess = guesses[guess_id]
                prob = guesses_prob[guess_id]
                if guess == truth:
                    exp_payoff += prob * V[infoset.t]
                else:
                    if truth > guess and infoset.t < T-1:
                        next_infoset = Infoset(
                            infoset.t+1, guess+1, infoset.n_high)
                        exp_payoff += prob * solve(next_infoset)
                    elif truth < guess and infoset.t < T-1:
                        next_infoset = Infoset(
                            infoset.t+1, infoset.n_low, guess-1)
                        exp_payoff += prob * solve(next_infoset)

            mem[infoset] = exp_payoff
            return exp_payoff
        P.append(solve(Infoset(0, 0, N-1)))

    return P


def solve_using_LP():
    # NOTE: very suboptimally implemented, runs out of memory quickly
    all_infosets = list()
    parent_sequences = defaultdict(lambda: set())
    child_sequences = defaultdict(lambda: set())
    seq_to_variable = dict()
    for t in range(T):
        for n_low in range(N):
            for n_high in range(n_low, N):
                infoset = Infoset(t, n_low, n_high)
                all_infosets.append(infoset)
                for guess in range(n_low, n_high+1):
                    sequence = Sequence(infoset, guess)
                    child_sequences[infoset].add(sequence)
                    assert sequence not in seq_to_variable
                    seq_to_variable[sequence] = cp.Variable(1, nonneg=True)

                    if t < T - 1:
                        # ans is higher
                        if guess < n_high:
                            child_infoset = Infoset(t+1, guess+1, n_high)
                            parent_sequences[child_infoset].add(sequence)

                        # ans is lower
                        if guess > n_low:
                            child_infoset = Infoset(t+1, n_low, guess-1)
                            parent_sequences[child_infoset].add(sequence)

    print('Add flow constraints to LP')
    root_infoset = Infoset(0, 0, N-1)
    flow_cons = list()
    for infoset in all_infosets:
        if infoset.t == 0:
            if infoset == root_infoset:
                # special case at root
                child_sum = cp.sum([seq_to_variable[seq]
                                   for seq in child_sequences[root_infoset]])
                flow_cons.append(child_sum == 1)
            else:
                child_sum = cp.sum([seq_to_variable[seq]
                                   for seq in child_sequences[infoset]])
                flow_cons.append(child_sum == 0)
        else:
            par_sum = cp.sum([seq_to_variable[seq]
                             for seq in parent_sequences[infoset]])
            child_sum = cp.sum([seq_to_variable[seq]
                               for seq in child_sequences[infoset]])
            flow_cons.append(par_sum == child_sum)

    print('Optimality constraints obtained from duality.')
    opt_constr = []

    L = cp.Variable(1,)
    payoffs = [[] for i in range(N)]
    for seq, seq_var in seq_to_variable.items():
        payoffs[seq.guess].append(V[seq.infoset.t] * seq_var)

    for pos in range(N):
        opt_constr.append(L <= cp.sum(payoffs[pos]))

    print('Constructing Problem.')
    prob = cp.Problem(cp.Maximize(L), flow_cons + opt_constr)

    print('Solving LP.')

    env = gp.Env()
    # Use dual simplex. Set to -1 if you want to use concurrent (faster but more memory consuming) and 0 for primal simplex
    env.setParam('method', 1)
    # method=1 for dual simplex
    prob.solve(solver=cp.GUROBI, verbose=True, env=env)

    print(prob.value)

    # Construct behavioral strategy
    beh_strat = dict()
    for infoset in all_infosets:
        # reach_prob[infoset] = sum([seq_to_variable[seq].value for seq in child_sequences[infoset]])
        reach_prob = sum(
            [seq_to_variable[seq].value for seq in child_sequences[infoset]])
        if reach_prob < 1e-9:
            continue
        guesses = []
        probs = []
        for child_seq in child_sequences[infoset]:
            guess = child_seq.guess
            prob = seq_to_variable[child_seq].value/reach_prob
            if prob < 1e-9:
                continue
            guesses.append(guess)
            probs.append(prob)
        beh_strat[infoset] = (guesses, probs)

    # Construct optimal hider strategy
    opt_hiding_locs = np.array([opt_constr[i].dual_value for i in range(N)]).flatten()

    return DictStrat(beh_strat), opt_hiding_locs





print('Naive deterministic', eval_strat(NaiveStrat(randomized=False)))
print('Naive randomized', eval_strat(NaiveStrat(randomized=True)))

gt_optimal_finder, gt_optimal_hiding_locs = solve_using_LP()
print('GT optimal', eval_strat(gt_optimal_finder))

for t in range(T):
    for n_low in range(N):
        for n_high in range(n_low, N):
            try: # Some infosets are not in the strategy
                print(t, n_low, n_high, gt_optimal_finder.get_behavioral(Infoset(t, n_low, n_high)))
            except:
                pass
    print('--------------')