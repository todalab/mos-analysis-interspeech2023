import json
import csv
from arm import Arms
from environment import OnMemoryEnvironment, ArmSystem


class VoiceMOSChallenge2022:

    def preprocess(self, input_file_path):
        systems = {}
        with open(input_file_path, newline='', mode='r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                system = row[0]
                sample_id = row[1].split('-')[1].rstrip('.wav')
                score = int(row[2])
                if system not in systems:
                    systems[system] = {}
                if sample_id not in systems[system]:
                    systems[system][sample_id] = []
                systems[system][sample_id].append(score)

        arm_systems = [ArmSystem(system_id,
                                 scores,
                                 min_reward_value=1,
                                 max_reward_value=5) for system_id, scores in systems.items()]
        arm_systems = sorted(arm_systems, key=lambda s: s.sample_mean(), reverse=True)
        environment = OnMemoryEnvironment(arm_systems)
        arms = Arms.create_from_ids([system.id for system in arm_systems])
        return environment, arms
