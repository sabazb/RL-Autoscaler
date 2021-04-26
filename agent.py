from config import Config
import random
from action import NineAction, FiveAction
from state import State, ModelBasedState
import copy
import numpy as np


class ModelItem:
    def __init__(self, state, action, cost, nstate):
        self.state = state
        self.action = action
        self.cost = cost
        self.nstate = nstate

    def set(self, cost, nstate):
        self.cost = cost
        self.nstate = nstate


class Model:
    def __init__(self):
        self.n_actions = Config.n_actions
        self.n_states = int(1 / Config.cpu_quantum) * Config.max_replica * (int(1 / Config.util_quantum) + 1)
        self.table = [None for i in range(self.n_actions) for j in range(self.n_states)]

    def get(self, state, action):
        action_idx = action.hash()
        state_idx = state.hash()
        return self.table[action_idx + state_idx * self.n_actions]

    def set(self, state, action, cost, nstate):
        action_idx = action.hash()
        state_idx = state.hash()
        if self.table[action_idx + state_idx * self.n_actions] is None:
            self.table[action_idx + state_idx * self.n_actions] = ModelItem(state, action, cost, nstate)
        else:
            self.table[action_idx + state_idx * self.n_actions].set(cost, nstate)

    def random_model_item(self):
        observed = [item for item in self.table if item is not None]
        model_item = random.choice(observed)
        return model_item


class Qtable:
    def __init__(self):
        self.n_actions = Config.n_actions
        self.n_states = int(1 / Config.cpu_quantum) * Config.max_replica * (int(1 / Config.util_quantum) + 1)
        self.table = [0 for i in range(self.n_actions) for j in range(self.n_states)]

    def get(self, state, action):
        action_idx = action.hash()
        state_idx = state.hash()
        return self.table[action_idx + state_idx * self.n_actions]

    def set(self, state, action, val):
        action_idx = action.hash()
        state_idx = state.hash()
        self.table[action_idx + state_idx * self.n_actions] = val


class QlearningAgent:
    def __init__(self):
        self.alpha = Config.alpha
        self.gamma = Config.gamma
        self.q_table = Qtable()
        self.n_actions = Config.n_actions

    def best_action(self, state):
        min_q = -1
        best_action = None
        action_indices = np.arange(self.n_actions)
        np.random.shuffle(action_indices)
        for i in action_indices:
            if self.n_actions == 9:
                action = NineAction(i)
            else:
                action = FiveAction(i)
            if state.is_valid_action(action):
                q = self.q_table.get(state, action)
                if min_q < 0 or q < min_q:
                    best_action = action
                    min_q = q
        return best_action, min_q

    def take_action(self, state, time):
        p = random.random()
        epsilon = 1. / time
        # epsilon = Config.epsilon
        if p > epsilon:
            action, _ = self.best_action(state)
        else:
            if self.n_actions == 9:
                action = NineAction.random_action()
                while not state.is_valid_action(action):
                    action = NineAction.random_action()
            else:
                action = FiveAction.random_action()
                while not state.is_valid_action(action):
                    action = FiveAction.random_action()
        return action

    @staticmethod
    def observed_cost(new_state, action):
        c_sla = new_state.is_sla_violated
        c_res = float(new_state.k / Config.max_replica) * new_state.cpu
        c_adapt = action.is_reconfig
        return c_sla * Config.w_sla + c_res * Config.w_res + c_adapt * Config.w_adapt

    def update(self, state, time, new_input_tps):
        action = self.take_action(state, time)
        previous_state = copy.deepcopy(state)
        state.update(action)
        state.set_input_tps(new_input_tps)
        cost = self.observed_cost(state, action)
        _, min_q = self.best_action(state)
        new_val = (1 - self.alpha) * self.q_table.get(previous_state, action) + self.alpha * (cost + self.gamma * min_q)
        self.q_table.set(previous_state, action, new_val)
        return action, cost, previous_state


class DynaQ(QlearningAgent):
    def __init__(self):
        super(DynaQ, self).__init__()
        self.model = Model()

    def update(self, state, time, new_input_tps):
        action, cost, previous_state = super(DynaQ, self).update(state, time, new_input_tps)
        self.model.set(previous_state, action, cost, state)
        for i in range(Config.n_dynaq):
            model_item = self.model.random_model_item()
            random_state = model_item.state
            random_action = model_item.action
            random_cost = model_item.cost
            random_nstate = model_item.nstate

            _, min_q = self.best_action(random_nstate)
            new_val = (1 - self.alpha) * self.q_table.get(random_state, random_action) + \
                      self.alpha * (random_cost + self.gamma * min_q)
            self.q_table.set(random_state, action, new_val)

        return action, cost, previous_state


class ModelBasedAgent:
    def __init__(self):
        self.alpha = Config.alpha
        self.gamma = Config.gamma
        self.n_actions = Config.n_actions
        self.n_states = int(1 / Config.cpu_quantum) * Config.max_replica * (int(1 / Config.util_quantum) + 1)

        self.q_table = Qtable()

        self.n_util = (int(1 / Config.util_quantum) + 1)
        self.trans_count = np.zeros((self.n_util, self.n_util))
        for i in range(self.n_util):
            self.trans_count[i, i] = 1

        self.cost_estimates = np.zeros(self.n_states)

    def best_action(self, state):
        min_q = -1
        best_action = None
        action_indices = np.arange(self.n_actions)
        np.random.shuffle(action_indices)
        for i in action_indices:
            if self.n_actions == 9:
                action = NineAction(i)
            else:
                action = FiveAction(i)
            if state.is_valid_action(action):
                q = self.q_table.get(state, action)
                if min_q < 0 or q < min_q:
                    best_action = action
                    min_q = q
        return best_action, min_q

    @staticmethod
    def known_cost(new_state, action):
        c_res = float(new_state.k / Config.max_replica) * new_state.cpu
        c_adapt = action.is_reconfig
        c_known = c_res * Config.w_res + c_adapt * Config.w_adapt
        return c_known

    @staticmethod
    def unobserved_cost(new_state):
        c_sla = new_state.is_sla_violated
        return c_sla * Config.w_sla

    def update_trans_count(self, state, nstate):
        self.trans_count[state.util_quant, nstate.util_quant] += 1

    def update_cost_estimate(self, nstate):
        old_cost = self.cost_estimates[nstate.hash()]
        unknown = self.unobserved_cost(nstate)
        new_cost = (1 - self.alpha) * old_cost + self.alpha * unknown
        self.cost_estimates[nstate.hash()] = new_cost

    def update_q(self, state, action):
        initial_util_quant = state.util_quant
        state.update(action)
        known_cost = self.known_cost(state, action)
        new_val = known_cost
        for util_quant in range(self.n_util):
            if self.trans_count[initial_util_quant, util_quant] == 0:
                continue
            p = self.trans_count[initial_util_quant, util_quant] / sum(self.trans_count[initial_util_quant, :])
            state.set_util_quant(util_quant)
            sla_cost = self.cost_estimates[state.hash()]
            _, min_q = self.best_action(state)
            new_val += p * (sla_cost + self.gamma * min_q)
        return new_val

    def iterate(self):
        for k in range(Config.max_replica):
            for cpu_quant in range(int(1 / Config.cpu_quantum)):
                state = ModelBasedState(k, cpu_quant)
                for action_idx in range(self.n_actions):
                    if self.n_actions == 9:
                        action = NineAction(action_idx)
                    else:
                        action = FiveAction(action_idx)
                    if state.is_valid_action(action):
                        q_val = self.update_q(copy.deepcopy(state), action)
                        self.q_table.set(state, action, q_val)

    def update(self, state, time, new_input_tps):
        previous_state = copy.deepcopy(state)
        state.set_input_tps(new_input_tps)
        action, _ = self.best_action(state)
        state.update(action)

        self.update_cost_estimate(state)
        self.update_trans_count(previous_state, state)
        self.iterate()

        return action, 0, previous_state

