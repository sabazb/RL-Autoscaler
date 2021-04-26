import matplotlib.pyplot as plt
from config import Config
from agent import QlearningAgent, DynaQ, ModelBasedAgent
from state import State, ModelBasedState
import numpy as np
from tqdm import tqdm


class Simulate:
    def __init__(self, agent_type):
        with open('data.txt', 'r') as f:
            self.requests = [int(line) for line in f.readlines()]
        if agent_type == 'Qlearning':
            self.agent = QlearningAgent()
        elif agent_type == "DynaQ":
            self.agent = DynaQ()
        elif agent_type == "ModelBased":
            self.agent = ModelBasedAgent()
        else:
            raise NotImplementedError
        self.state = State(1, 10)
        self.stats = {'SLA Violation': [], 'Response Time': [], 'Adaptation': [],
                      'Utilization': [], 'CPU Share': [], 'Containers': [], }


    def plot_work_load(self, start_time, end_time):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(start_time, end_time), self.requests[start_time:end_time], color='darkorange')
        plt.xlabel('Time')
        plt.ylabel('Request rate')
        plt.show()

    def record_stats(self, action):
        self.stats['SLA Violation'].append(self.state.is_sla_violated)
        self.stats['Response Time'].append(self.state.response_time)
        self.stats['Utilization'].append(self.state.utils)
        self.stats['CPU Share'].append(self.state.cpu)
        self.stats['Containers'].append(self.state.k)
        self.stats['Adaptation'].append(1 if action.is_reconfig else 0)

    def report_stats(self):
        print("SLA Violation: {:.2f}%".format(np.mean(self.stats['SLA Violation']) * 100))
        print("Median Response Time: {:.2f}ms".format(np.median(self.stats['Response Time']) * 1000))
        print("CPU Utilization: {:.2f}%".format(np.mean([util if util <= 1 else 1 for util in self.stats['Utilization']]) * 100))
        print("CPU Share: {:.2f}%".format(np.mean(self.stats['CPU Share']) * 100))
        print("Average Number of Containers: {:.2f}".format(np.mean(self.stats['Containers'])))
        print("Adaptation: {:.2f}%".format(np.mean(self.stats['Adaptation']) * 100))

    def run(self):
        for time in tqdm(range(Config.time_limit)):
            action, _, _ = self.agent.update(self.state, time + 1, self.requests[time])
            self.record_stats(action)

    def plot_stat(self, stat_name='Utilization'):
        stat = self.stats[stat_name]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(len(stat)), stat, color='darkorange')
        plt.xlabel('Time')
        plt.ylabel(stat_name)
        plt.show()

if __name__ == "__main__":
    # simulate = Simulate("ModelBased")
    # simulate = Simulate("DynaQ")
    simulate = Simulate("Qlearning")
    simulate.run()

