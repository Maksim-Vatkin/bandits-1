import abc
import math
import random
import operator


def index_max(values):
    index, value = max(enumerate(values), key=operator.itemgetter(1))
    return index


class BanditAlgorithm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def select_arm(self):
        pass

    @abc.abstractmethod
    def update(self, reward):
        pass

    def reset(self):
        self.arms = {i: ArmCounter() for i in range(len(self.arms))}

    def update(self, arm, reward):
        self.arms[arm].update(reward)


class ArmCounter:

    def __init__(self):
        self.history = []
        self.total_pulls = 0
        self.total_rewards = 0

    @property
    def payout(self):
        try:
            return self.total_rewards / self.total_pulls
        # algorithms expect that on the initial pull all the probabilities
        # are zero
        except ZeroDivisionError:
            return 0.

    def update(self, reward):
        self.total_pulls += 1
        self.total_rewards += reward
        self.history.append(reward)

    def __repr__(self):
        return "<ArmCounter %s rewards %s pulls (%s)>" % (self.total_rewards, self.total_pulls, self.payout)


class EpsilonGreedy(BanditAlgorithm):
    """
    Pure A/B testing (exploration) -> epsilon = 100%
    Pure exploitation -> epsilon = 0%
    There exists some middleground

    The first weakness is that, as you
get more certain which of your two logo designs is best, this tendency to explore the
worse design a full 5% of the time will become more wasteful. In the jargon of bandit
algorithms, you’ll be over-exploring. And there’s another problem with a fixed 10%
exploration rule: at the start of your experimentation, you’ll choose options that you
don’t know much about far more rarely than you’d like to because you only try new
options 10% of the time.


    If epsilon is high, we explore a lot and find the best arm quickly, but then we
keep exploring even after it’s not worth doing anymore.

the epsilon-Greedy algorithm
does eventually figure out which arm is best no matter how epsilon is set. But the length
of time required to figure our which arm is best depends a lot on the value of epsilon.
    """

    def __init__(self, epsilon, num_arms, is_annealing=False):
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon
        self.is_annealing = is_annealing
        self.arms = {i: ArmCounter() for i in range(num_arms)}

    def select_arm(self):
        if self.is_annealing:
            t = sum(a.total_pulls for a in self.arms.values()) + 1
            epsilon = 1 / math.log(t + 0.0000001)
        else:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice(list(self.arms.keys()))
        else:
            return index_max([self.arms[i].payout for i in range(len(self.arms))])



class SoftMax(BanditAlgorithm):

    def __init__(self, temperature, num_arms, is_annealing=False):
        assert temperature > 0
        self.temperature = temperature
        self.is_annealing = is_annealing
        self.arms = {i: ArmCounter() for i in range(num_arms)}

    def select_arm(self):
        if self.is_annealing:
            t = sum(a.total_pulls for a in self.arms.values()) + 1
            temperature = 1 / math.log(t + 0.0000001)
        else:
            temperature = self.temperature
        probabilities = {i: math.exp(counter.payout / temperature) for i, counter in self.arms.items()}
        total_probability = sum(probabilities.values())
        normalized_probabilities = {i: p / total_probability for i, p in probabilities.items()}
        return self._draw_in_proportion(normalized_probabilities)

    def _draw_in_proportion(self, probabilities):
        threshold = random.random()

        cumulative = 0
        for arm, prob in probabilities.items():
            cumulative += prob
            if cumulative > threshold:
                return arm
        else:
            return arm




class UCB1(BanditAlgorithm):

    def __init__(self, num_arms):
        self.arms = {i: ArmCounter() for i in range(num_arms)}

    def select_arm(self):
        total_counts = sum(arm.total_pulls for arm in self.arms.values())

        ucb_values = {}
        for i, arm in self.arms.items():
            # must explore every arm at least once
            if arm.total_pulls == 0:
                return i
            bonus = math.sqrt((2 * math.log(total_counts)) / float(arm.total_pulls))
            ucb_values[i] = arm.payout + bonus
        return index_max(ucb_values[i] for i in range(len(ucb_values)))





