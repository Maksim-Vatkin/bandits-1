import pandas as pd


class AlgorithmTester:

    def __init__(self, algorithm, arms, rounds, horizon):
        self.algorithm = algorithm
        self.arms = arms
        self.rounds = rounds
        self.horizon = horizon

        self.history = []

    def test(self):
        for _ in range(self.rounds):
            round_ = []
            self.algorithm.reset()
            for _ in range(self.horizon):
                arm = self.algorithm.select_arm()
                reward = self.arms[arm].draw()
                self.algorithm.update(arm, reward)
                round_.append((arm, reward))
            self.history.append(round_)



def test_algorithm(base_algorithm, arms, rounds=5000, horizon=250):
    results = {}
    for epsilon in (0.1, 0.2, 0.3, 0.4, 0.5):
        algorithm = base_algorithm(epsilon, len(arms))
        tester = AlgorithmTester(algorithm, arms, rounds, horizon)
        tester.test()
        results[epsilon] = track_performance(tester.history)
    return pd.concat(results, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        # 1) probability best arm
        # 2) average reward
        # 3) cumulative reward

def track_performance(history, best_arm=4):
    totals = {}
    for round_ in history:
        num_best_arm = 0
        for i, (arm, reward) in enumerate(round_):
            if i not in totals:
                totals[i] = 0

            totals[i] += reward

    avgs = {k: v / len(history) for k,v in totals.items()}

    best_arms = {}
    df = pd.DataFrame(history)
    for col in df.columns:
        best_arms[col] = df[col].map(lambda x: x[0]).value_counts().to_dict().get(best_arm,0) / len(df)


    return pd.concat(dict(
            cumulative=pd.concat({col: df[col].map(lambda x: x[1])for col in df.columns}, axis=1).cumsum(axis=1).mean(axis=0),
            avg_reward=pd.Series(avgs),
            best_arm=pd.Series(best_arms)
        ), axis=1)
