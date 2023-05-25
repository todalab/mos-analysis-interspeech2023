import numpy as np


class Arm:

    def __init__(self, _id):
        self.id = _id
        self.samples = []

    def n(self):
        return len(self.samples)

    def sample_mean(self):
        if self.n() == 0:
            return 0.0
        return sum(self.samples) / float(self.n())

    def add_sample(self, sample):
        self.samples.append(sample)

    def add_samples(self, samples):
        self.samples.extend(samples)


class Arms:

    @staticmethod
    def create_from_ids(ids):
        return Arms([Arm(i) for i in ids])

    def __init__(self, arms):
        self.arms = arms
        self.arm_map = {arm.id: arm for arm in arms}

    def k(self):
        return len(self.arms)

    def t(self):
        return sum([arm.n() for arm in self.arms])

    def sample(self, decisions, samples):
        assert len(decisions) == len(samples)
        for decision, sample in zip(decisions, samples):
            arm = self.arm_map[decision]
            arm.add_sample(sample)

    def best_arm(self):
        i = np.argmax([arm.sample_mean() for arm in self.arms])
        return self.arms[i]

    def suboptimal_arms(self):
        i = np.argmax([arm.sample_mean() for arm in self.arms])
        return [arm for j, arm in enumerate(self.arms) if j != i]

    def remove(self, _id):
        arm = self.arm_map[_id]
        self.arms.remove(arm)
        self.arm_map.pop(_id)

    def remove_all(self, ids):
        for i in ids:
            self.remove(i)

    def all_sample_mean(self):
        return {key: arm.sample_mean() for key, arm in self.arm_map.items()}

    def all_n_samples(self):
        return {key: arm.n() for key, arm in self.arm_map.items()}

