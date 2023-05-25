import numpy as np


class ArmSystem:

    def __init__(self, _id, samples, min_reward_value, max_reward_value):
        self.id = _id
        self.samples = samples
        self.min_reward_value = min_reward_value
        self.max_reward_value = max_reward_value
        self.sample_ids = list(samples.keys())
        self._n_sample_map = {key: len(samples) for key, samples in self.samples.items()}
        normalized_samples = np.array([self.normalize_reward(s) for sx in self.samples.values() for s in sx])
        self.total_samples = len(normalized_samples)
        self._sample_mean = sum(normalized_samples) / self.total_samples
        self._sample_stddev = np.std(normalized_samples)

    def normalize_reward(self, reward):
        scale = self.max_reward_value - self.min_reward_value
        return (reward - self.min_reward_value) / scale

    def denormalize_reward(self, reward):
        scale = self.max_reward_value - self.min_reward_value
        return reward * scale + self.min_reward_value

    def n_sample_map(self):
        return self._n_sample_map

    def sample_mean(self):
        return self._sample_mean

    def sample_stddev(self):
        return self._sample_stddev

    def denormalized_sample_mean(self):
        return self.denormalize_reward(self._sample_mean)

    def n_sample(self, sample_id):
        return self._n_sample_map[sample_id]

    def random_sample_id(self):
        ns = [self._n_sample_map[i] for i in self.sample_ids]
        denom = sum(ns)
        if denom == 0:
            return None
        p = np.array(ns, dtype=np.float) / denom
        i = np.random.choice(len(self.sample_ids), p=p)
        return self.sample_ids[i]

    def pop_sample(self, sample_id):
        self._n_sample_map[sample_id] -= 1
        reward = self.samples[sample_id].pop()
        return self.normalize_reward(reward)


def sum_nonzero(ns):
    m = min(ns)
    if m == 0:
        return 0
    else:
        return sum(ns)


class OnMemoryEnvironment:

    def __init__(self, systems):
        self.systems = {system.id: system for system in systems}

    def random_reward_from_same_sample_id(self, decisions):
        '''
        :param decisions: no duplicate list
        :return:
        '''
        systems = [self.systems[d] for d in decisions]
        sample_ids = [system.random_sample_id() for system in systems]
        sample_ids = [i for i in sample_ids if i is not None]
        n_samples = [sum_nonzero([system.n_sample(sample_id) for system in systems]) for sample_id in sample_ids]
        if sum(n_samples) == 0:
            return None
        max_sample_pool_id = sample_ids[np.argmax(n_samples)]
        return [system.pop_sample(max_sample_pool_id) for system in systems]

    def random_reward(self, decisions):
        '''
        :param decisions: no duplicate list
        :return:
        '''
        systems = [self.systems[d] for d in decisions]
        sample_ids = [system.random_sample_id() for system in systems]
        if any([i is None for i in sample_ids]):
            return None

        return [system.pop_sample(sample_id) for system, sample_id in zip(systems, sample_ids)]

    def sample_complexity(self, epsilon):
        systems = list(self.systems.values())
        sample_mean = [system.sample_mean() for system in systems]
        best_system_index = np.argmax(sample_mean)
        best_sample_mean = sample_mean[best_system_index]
        sample_mean_wo_best = sample_mean.copy()
        sample_mean_wo_best.pop(best_system_index)
        second_best_system_index = np.argmax(sample_mean_wo_best)
        second_best_sample_mean = sample_mean_wo_best[second_best_system_index]
        sc = np.array(sample_mean_wo_best + [second_best_sample_mean])
        sc = np.sum(np.reciprocal(2 * np.square(best_sample_mean + epsilon - sc)))
        return sc

    def sample_stats(self):
        return {system.id: (system.sample_mean(), system.sample_stddev(), system.total_samples) for system in self.systems.values()}
