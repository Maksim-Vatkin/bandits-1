import random


class BernoulliArm:

    def __init__(self, p):
        assert 0 <= p <= 1
        self.p = p

    def draw(self):
        if random.random() < self.p:
            return 1
        else:
            return 0