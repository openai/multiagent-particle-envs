import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

class EnsembleBaseScenario(BaseScenario):
    def __init__(self):
        self.partition = 'rand'
        self.partition_flag = -1
        self.measure_success = False

    def select_agents(self, world):
        good_agents = [agent for agent in world.all_agents if not agent.adversary]
        adversary_agents = [agent for agent in world.all_agents if agent.adversary]
        n_good = world.good_part_n
        n_bad = world.adversary_part_n
        if self.partition == 'rand':
            np.random.shuffle(good_agents)
            np.random.shuffle(adversary_agents)
            world.agents = adversary_agents[:world.num_adversaries] + \
                           good_agents[:(world.num_agents - world.num_adversaries)]
        elif self.partition == 'fix':
            k = np.random.choice(world.partition_n)
            bad_part = adversary_agents[k * n_bad: (k + 1) * n_bad]
            np.random.shuffle(bad_part)
            good_part = good_agents[k * n_good: (k + 1) * n_good]
            np.random.shuffle(good_part)
            world.agents = bad_part[:world.num_adversaries] + good_part[:(world.num_agents - world.num_adversaries)]
        else:
            fix_good = good_agents[:n_good]
            rand_good_all = good_agents[n_good:]
            np.random.shuffle(fix_good)
            fix_bad = adversary_agents[:n_bad]
            rand_bad_all = adversary_agents[n_bad:]
            np.random.shuffle(fix_bad)
            # pick a team from rand-good/bad
            t = np.random.choice(world.partition_n - 1)  # excluding fix-team
            rand_good = rand_good_all[t * n_good: (t+1) * n_good]
            t = np.random.choice(world.partition_n - 1)
            rand_bad = rand_bad_all[t * n_bad: (t+1) * n_bad]
            np.random.shuffle(rand_good)
            np.random.shuffle(rand_bad)
            if self.partition == 'mix':
                k = np.random.choice(world.partition_n)
                if self.partition_flag > -1:  # only use fixed partition
                    k = self.partition_flag
                if k == 0:
                    world.agents = fix_bad[:world.num_adversaries] + fix_good[:(world.num_agents - world.num_adversaries)]
                else:
                    world.agents = rand_bad[:world.num_adversaries] + \
                                   rand_good[:(world.num_agents - world.num_adversaries)]
            else:
                if self.partition_flag > -1:
                    k = self.partition_flag
                else:
                    k = np.random.choice(2)
                if k == 0:
                    world.agents = fix_bad[:world.num_adversaries] + rand_good[:(world.num_agents - world.num_adversaries)]
                else:
                    world.agents = rand_bad[:world.num_adversaries] + fix_good[:(world.num_agents - world.num_adversaries)]
        assert (len(world.agents) == world.num_agents)
