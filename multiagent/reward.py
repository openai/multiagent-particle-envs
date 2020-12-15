import cvxpy as cvx

# import osqp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#### file to calculate the rewards. Meant to be modular:
#### class Rewards should have several different functions by Dec 2019


class Reward:
    def __init__(self, energy_use, prices, min_demand, max_demand):
        """
		Args: 
			energy_use: list returned by Person class signifying energy use 
			prices: list returned by grid signifying cost throughout day 
			min_demand: value computed by Person class signifying minimum energy use long term
			max_demand: value computed by Person class signifying maximum energy use long term
		"""

        self.energy_use = np.array(energy_use)
        self.prices = np.array(prices)
        self._num_timesteps = energy_use.shape[0]
        self.min_demand = np.min(energy_use)  # min_demand
        self.max_demand = np.max(energy_use)  # max_demand
        self.baseline_max_demand = 159.32

        assert round(self.max_demand) == round(
            max_demand
        ), "The max demand that the player is using and the optimization is using is not the same"
        assert round(self.min_demand) == round(
            min_demand
        ), "The min demand that the player is using and the optimization is using is not the same"

        self.total_demand = np.sum(energy_use)

    def ideal_use_calculation(self):
        """
		Computes an optimization of demand according to price 

		returns: np.array of ideal energy demands given a price signal 
		"""

        demands = cvx.Variable(self._num_timesteps)
        min_demand = cvx.Parameter()
        max_demand = cvx.Parameter()
        total_demand = cvx.Parameter()
        prices = cvx.Parameter(self._num_timesteps)

        min_demand = self.min_demand
        max_demand = self.max_demand
        total_demand = self.total_demand

        while max_demand * 10 < total_demand:
            print("multiplying demand to make optimization work")
            print("max_demand: " + str(max_demand))
            print("total_demand: " + str(total_demand))
            print("energy_use")
            print(self.energy_use)
            max_demand *= 1.1

        prices = self.prices
        constraints = [cvx.sum(demands, axis=0, keepdims=True) == total_demand]
        # constraints = [np.ones(self._num_timesteps).T * demands == total_demand]
        for i in range(self._num_timesteps):
            constraints += [demands[i] <= max_demand]
            constraints += [min_demand <= demands[i]]
            # if i != 0:y
            # 	constraints += [cvx.abs(demands[i] - demands[i-1]) <= 100]

        objective = cvx.Minimize(demands.T * prices)
        problem = cvx.Problem(objective, constraints)

        problem.solve(solver=cvx.ECOS, verbose=False)
        return np.array(demands.value)

    def log_cost(self):
        """
		Scales energy_use to be between min and max energy demands (this is repeated 
		in agent.routine_output_trasform), and then returns the simple total cost. 

		"""

        scaler = MinMaxScaler(feature_range=(self.min_demand, self.max_demand))
        scaled_energy = np.squeeze(scaler.fit_transform(self.energy_use.reshape(-1, 1)))

        return -np.log(np.dot(scaled_energy, self.prices))

    def log_cost_regularized(self):
        """
		Scales energy_use to be between min and max energy demands (this is repeated 
		in agent.routine_output_trasform), and then returns the simple total cost. 

		:param: h - the hyperparameter that modifies the penalty on energy demand that's driven too low. 

		"""

        scaler = MinMaxScaler(feature_range=(self.min_demand, self.max_demand))
        ## ?
        scaled_energy = np.squeeze(scaler.fit_transform(self.energy_use.reshape(-1, 1)))

        return -np.log(np.dot(scaled_energy, self.prices)) - 10 * (
            np.sum(self.energy_use) < (10 * (0.5 * self.baseline_max_demand))
        )  # -
        # 10 * ())
        # sigmoid between 10 and 20 so there's a smooth transition
        # - lambd * (difference b/w energy(t)) - put a bound on
        # play with the lipschitz constant
        # [10, 20, 10, 10, 10, ]

    def neg_distance_from_ideal(self, demands):
        """
		args: 
			demands: np.array() of demands from ideal_use_calculation()

		returns: 
			a numerical distance metric, negated
		"""

        return -((demands - self.energy_use) ** 2).sum()

    def cost_distance(self, ideal_demands):
        """
		args: 
			demands: np.array() of demands from ideal_use_calculation()

		returns: 
			a cost-based distance metric, negated
		"""
        current_cost = np.dot(self.prices, self.energy_use)
        ideal_cost = np.dot(self.prices, ideal_demands)

        cost_difference = ideal_cost - current_cost

        return cost_difference

    def log_cost_distance(self, ideal_demands):
        """
		args: 
			demands: np.array() of demands from ideal_use_calculation()

		returns: 
			the log of the cost distance
		"""
        current_cost = np.dot(self.prices, self.energy_use)
        ideal_cost = np.dot(self.prices, ideal_demands)

        cost_difference = ideal_cost - current_cost

        # TODO ENSURE THAT COST DIFFERENCE IS < 0
        if cost_difference < 0:
            return -np.log(-cost_difference)
        else:
            print(
                "WEIRD REWARD ALERT. IDEAL COST >= CURRENT COST. returning reward of 10"
            )
            return 10

    def scaled_cost_distance(self, ideal_demands):
        """
		args: 
			demands: np.array() of demands from ideal_use_calculation()

		returns: 
			a cost-based distance metric normalized by total ideal cost
		"""

        current_cost = np.dot(self.prices, self.energy_use)
        ideal_cost = np.dot(self.prices, ideal_demands)

        cost_difference = ideal_cost - current_cost

        # print("--" * 10)
        # print("ideal cost")
        # print(ideal_cost)
        # print("--" * 10)
        # print("prices")
        # print(self.prices)
        # print("--" * 10)
        # print("current_cost")
        # print(current_cost)

        if cost_difference > 0 or ideal_cost < 0:
            print("--" * 10)
            print("Problem with reward")
            # print("min_demand: " + str(self.min_demand))
            # print("max_demand: " + str(self.max_demand))
            # print("--" * 10)
            print("prices")
            print(self.prices)
            # print("current_cost")
            # print(current_cost)
            # print("--" * 10)
            # print("ideal_cost")
            # print(ideal_cost)
            # print("ideal_demands")
            # print(ideal_demands)
            # print("energy demand")
            # print(self.energy_use)
            print("taking the neg abs value so that it stays the same sign.")

        return -np.abs(cost_difference / ideal_cost)

