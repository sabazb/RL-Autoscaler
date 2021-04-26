from config import Config


class State:
    def __init__(self, k, cpu_quant):
        self.k = k
        self.cpu_quant = cpu_quant
        self.input_tps = 0

    def hash(self):
        n_cpu = int(1 / Config.cpu_quantum)
        n_util = int(1 / Config.util_quantum) + 1
        return int((n_util * (self.k - 1) + self.util_quant) * n_cpu + (self.cpu_quant - 1))

    def set_input_tps(self, input_tps):
        self.input_tps = input_tps

    @property
    def cpu(self):
        return self.cpu_quant * Config.cpu_quantum

    @property
    def util_quant(self):
        if self.utils > 1:
            return int(1 / Config.util_quantum)
        return int(self.utils / Config.util_quantum)

    @property
    def utils(self):
        service_time_mean = 1. / (Config.mu * self.cpu)
        rho = (self.input_tps / self.k) * service_time_mean
        return rho

    @property
    def response_time(self):
        if self.utils >= 1:
            return 999999.

        service_time_mean = 1. / (Config.mu * self.cpu)
        return service_time_mean + self.utils / (2 * (1 - self.utils)) * service_time_mean

    @property
    def is_sla_violated(self):
        return self.response_time > Config.max_response_time

    # def is_sla_violated2(self, utils_quant):
    #     utils = utils_quant * Config.util_quantum
    #     if utils >= 1:
    #         return True
    #     service_time_mean = 1. / (Config.mu * self.cpu)
    #     response_time = service_time_mean + utils / (2 * (1 - utils)) * service_time_mean
    #     return response_time > Config.max_response_time

    # def hash2(self, utils_quant):
    #     n_cpu = int(1 / Config.cpu_quantum)
    #     n_util = int(1 / Config.util_quantum) + 1
    #     return (n_util * (self.k - 1) + utils_quant) * n_cpu + (self.cpu_quant - 1)

    def is_valid_action(self, action):
        delta_k = action.delta_k
        delta_cpu_quant = action.delta_cpu

        new_k = self.k + delta_k
        new_cpu_quant = self.cpu_quant + delta_cpu_quant
        if new_k > Config.max_replica or new_k < 1:
            return False
        if new_cpu_quant < 1 or new_cpu_quant > int(1 / Config.cpu_quantum):
            return False

        return True

    def update(self, action):
        self.cpu_quant += action.delta_cpu
        self.k += action.delta_k


class ModelBasedState:
    def __init__(self, k, cpu_qunt):
        self.k = k
        self.cpu_quant = cpu_qunt
        self.util_quant = 0

    def hash(self):
        n_cpu = int(1 / Config.cpu_quantum)
        n_util = int(1 / Config.util_quantum) + 1
        return (n_util * (self.k - 1) + self.util_quant) * n_cpu + (self.cpu_quant - 1)

    def set_util_quant(self, util_quant):
        self.util_quant = util_quant

    @property
    def cpu(self):
        return self.cpu_quant * Config.cpu_quantum

    @property
    def utils(self):
        return self.utils * Config.util_quantum

    @property
    def response_time(self):
        if self.utils >= 1:
            return 999999.

        service_time_mean = 1. / (Config.mu * self.cpu)
        return service_time_mean + self.utils / (2 * (1 - self.utils)) * service_time_mean

    @property
    def is_sla_violated(self):
        return self.response_time > Config.max_response_time

    def is_valid_action(self, action):
        delta_k = action.delta_k
        delta_cpu = action.delta_cpu

        new_k = self.k + delta_k
        new_cpu_quant = self.cpu_quant + delta_cpu
        if new_k > Config.max_replica or new_k < 1:
            return False
        if new_cpu_quant < 1 or new_cpu_quant > int(1 / Config.cpu_quantum):
            return False

        return True

    def update(self, action):
        self.cpu_quant += action.delta_cpu
        self.k += action.delta_k
