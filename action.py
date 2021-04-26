from config import Config
import random


class Action:
    def __init__(self, idx):
        self.idx = idx
        self.delta_cpu = 0
        self.delta_k = 0

    def hash(self):
        return self.idx

    @property
    def is_reconfig(self):
        return self.delta_cpu != 0


class NineAction(Action):
    OUT_DOWN = 0  # {1, -r} ->  scale out and scale down
    OUT_NULL = 1  # {1, 0} -> scale out
    OUT_UP = 2  # {1, r} -> scale out and scale up
    NULL_DOWN = 3  # {0, -r} -> scale down
    NULL_NULL = 4  # {0, 0} -> no
    NULL_UP = 5  # {0, r} -> scale up
    IN_DOWN = 6  # {-1, -r} -> scale in and scale down
    IN_NULL = 7  # {-1, 0} -> scale in
    IN_UP = 8  # {-1, r} -> scale in and scale up

    def __init__(self, idx):
        super().__init__(idx)

        if idx == self.OUT_DOWN or idx == self.NULL_DOWN or idx == self.IN_DOWN:
            self.delta_cpu = -1

        if idx == self.OUT_UP or idx == self.NULL_UP or idx == self.IN_UP:
            self.delta_cpu = 1

        if idx == self.IN_DOWN or idx == self.IN_NULL or idx == self.IN_UP:
            self.delta_k = -1

        if idx == self.OUT_DOWN or idx == self.OUT_NULL or idx == self.OUT_UP:
            self.delta_k = 1

    @staticmethod
    def random_action():
        rand_idx = random.randint(0, 8)
        return NineAction(rand_idx)


class FiveAction(Action):
    OUT_NULL = 0  # {1, 0} -> scale out
    NULL_NULL = 1  # {0, 0} -> no
    IN_NULL = 2  # {-1, 0} -> scale in
    NULL_DOWN = 3  # {0, -r} -> scale down
    NULL_UP = 4  # {0, r} -> scale up

    def __init__(self, idx):
        super().__init__(idx)

        if idx == self.NULL_DOWN:
            self.delta_cpu = -1

        if idx == self.NULL_UP:
            self.delta_cpu = 1

        if idx == self.IN_NULL:
            self.delta_k = -1

        if idx == self.OUT_NULL:
            self.delta_k = 1

    @staticmethod
    def random_action():
        rand_idx = random.randint(0, 4)
        return FiveAction(rand_idx)
