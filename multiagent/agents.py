import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#### file to make the simulation of people that we can work with


class Person:
    """ Person (parent?) class -- will define how the person takes in a points signal and puts out an energy signal 
	baseline_energy = a list or dataframe of values. This is data from SinBerBEST 
	points_multiplier = an int which describes how sensitive each person is to points 

	"""

    def __init__(self, baseline_energy_df, points_multiplier=1):
        self.baseline_energy_df = baseline_energy_df
        self.baseline_energy = np.array(self.baseline_energy_df["net_energy_use"])
        self.points_multiplier = points_multiplier

        baseline_min = self.baseline_energy.min()
        baseline_max = self.baseline_energy.max()
        baseline_range = baseline_max - baseline_min

        self.min_demand = np.maximum(0, baseline_min + baseline_range * 0.05)
        self.max_demand = np.maximum(0, baseline_min + baseline_range * 0.95)
        self.silent = True
        self.prev_energy = self.baseline_energy  # basically a placeholder.

    def energy_output_simple_linear(self, points):
        """Determines the energy output of the person, based on the formula:
		
		y[n] = -sum_{rolling window of 5} points + baseline_energy + noise

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 1 hour increments

		"""
        points_df = pd.DataFrame(points)

        points_effect = points_df.rolling(window=5, min_periods=1).mean()

        time = points_effect.shape[0]
        energy_output = []

        for t in range(time):
            temp_energy = (
                self.baseline_energy[t]
                - points_effect.iloc[t] * self.points_multiplier
                + np.random.normal(1)
            )
            energy_output.append(temp_energy)

        return pd.DataFrame(energy_output)

    def pure_linear_signal(self, points, baseline_day=0):
        """
		A linear person. The more points you give them, the less energy they will use
		(within some bounds) for each hour. No rolling effects or anything. The simplest
		signal. 
		"""

        # hack here to always grab the first day from the baseline_energy
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]

        points_effect = np.array(points * self.points_multiplier)
        output = output - points_effect

        # impose bounds/constraints
        output = np.maximum(output, self.min_demand)
        output = np.minimum(output, self.max_demand)
        return output

    def get_min_demand(self):
        return self.min_demand
        # return np.quantile(self.baseline_energy, .05)

    def get_max_demand(self):
        return self.max_demand
        # return np.quantile(self.baseline_energy, .95)


class FixedDemandPerson(Person):
    def __init__(self, baseline_energy_df, points_multiplier=1):
        super().__init__(baseline_energy_df, points_multiplier)

    def demand_from_points(self, points, baseline_day=0):
        # hack here to always grab the first day from the baseline_energy
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        total_demand = np.sum(output)

        points_effect = np.array(points * self.points_multiplier)
        output = output - points_effect

        # scale to keep total_demand (almost) constant
        # almost bc imposing bounds afterwards
        output = output * (total_demand / np.sum(output))

        # impose bounds/constraints
        output = np.maximum(output, self.min_demand)
        output = np.minimum(output, self.max_demand)

        return output

    def adverserial_linear(self, points, baseline_day=0):
        # hack here to always grab the first day from the baseline_energy
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        total_demand = np.sum(output)

        points_effect = np.array(points * self.points_multiplier)
        output = output + points_effect

        # scale to keep total_demand (almost) constant
        # almost bc imposing bounds afterwards
        output = output * (total_demand / np.sum(output))

        # impose bounds/constraints
        output = np.maximum(output, self.min_demand)
        output = np.minimum(output, self.max_demand)

        return output


class DeterministicFunctionPerson(Person):
    def __init__(self, baseline_energy_df, points_multiplier=1, response="t"):
        super().__init__(baseline_energy_df, points_multiplier)
        self.response = response

    def threshold_response_func(self, points):
        points = np.array(points) * self.points_multiplier
        threshold = np.mean(points)
        return [p if p > threshold else 0 for p in points]

    def exponential_response_func(self, points):
        points = np.array(points) * self.points_multiplier
        points_effect = [p ** 2 for p in points]

        return points_effect

    def sin_response_func(self, points):
        points = np.array(points)
        # n = np.max(points)
        # points = [np.sin((float(i)/float(n))*np.pi) for i in points]
        points = [np.sin(float(i) * np.pi) * self.points_multiplier for i in points]
        points = points
        return points

    def linear_response_func(self, points):
        return points * self.points_multiplier

    def routine_output_transform(self, points_effect, baseline_day=0, day_of_week=None):
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        total_demand = np.sum(output)

        # scale to keep total_demand (almost) constant
        # almost bc imposing bounds afterwards
        output = output - points_effect
        # output = output * (total_demand/np.sum(output))

        # impose bounds/constraints
        # output = np.maximum(output, self.min_demand)
        # output = np.minimum(output, self.max_demand)
        # return output

        if day_of_week != None:
            energy_resp = energy_resp * self.day_of_week_multiplier[day_of_week]

        scaler = MinMaxScaler(feature_range=(self.min_demand, self.max_demand))
        scaled_output = scaler.fit_transform(output.reshape(-1, 1))

        return np.squeeze(scaled_output)

    def threshold_response(self, points, day_of_week=None):
        points_effect = self.threshold_response_func(points)
        output = self.routine_output_transform(points_effect, day_of_week=day_of_week)
        return output

    def sin_response(self, points, day_of_week=None):
        points_effect = self.sin_response_func(points)
        output = self.routine_output_transform(points_effect, day_of_week=day_of_week)
        return output

    def exp_response(self, points, day_of_week=None):
        points_effect = self.exponential_response_func(points)
        output = self.routine_output_transform(points_effect, day_of_week=day_of_week)
        return output

    def threshold_exp_response(self, points, day_of_week=None):
        points_effect = self.exponential_response_func(points)
        points_effect = self.threshold_response_func(points_effect)
        output = self.routine_output_transform(points_effect, day_of_week=day_of_week)
        return output

    def linear_response(self, points, day_of_week=None):
        points_effect = points * self.points_multiplier
        output = self.routine_output_transform(points_effect, day_of_week=day_of_week)
        return output

    def get_response(self, points, day_of_week=None):
        if self.response == "t":
            energy_resp = self.threshold_exp_response(points, day_of_week=day_of_week)
        elif self.response == "s":
            energy_resp = self.sin_response(points, day_of_week=day_of_week)
        elif self.response == "l":
            energy_resp = self.linear_response(points, day_of_week=day_of_week)
        elif self.response == "m":
            energy_resp = self.mixed_response(points, day_of_week=day_of_week)
        else:
            raise NotImplementedError
        return energy_resp


class RandomizedFunctionPerson(DeterministicFunctionPerson):
    def __init__(
        self,
        baseline_energy_df,
        points_multiplier=1,
        response="t",
        low=0,
        high=50,
        distr="U",
    ):

        """
		Adds Random Noise to DeterministicFunctionPerson energy output (for D.R. purposes)

		New Args:
			Low = Lower bound for random noise added to energy use
			High = Upper bound "    "      "     "    "    "
			Distr = 'G' for Gaussian noise, 'U' for Uniform random noise (Note: Continuous distr.)

		Note: For design purposes the random noise is updated at the end of each episode
	 """
        # TODO: Multivariate distr??

        super().__init__(
            baseline_energy_df, points_multiplier=points_multiplier, response=response
        )

        distr = distr.upper()
        assert distr in ["G", "U"]

        self.response = response
        self.low = low
        self.high = high if high < self.max_demand else 50
        self.distr = distr

        self.noise = []
        self.update_noise()

    def update_noise(self):
        if self.distr == "G":
            # TODO: Update how to sample from Gausian
            self.noise = np.random.normal(
                loc=(self.low + self.high) / 2, scale=10, size=10
            )

        elif self.distr == "U":
            self.noise = np.random.uniform(low=self.low, high=self.high, size=10)

    def exponential_response_func(self, points):
        points = np.array(points) * self.points_multiplier
        points_effect = [p ** 2 for p in points]

        return points_effect + self.noise

    def sin_response_func(self, points):
        points = np.array(points)
        # n = np.max(points)
        # points = [np.sin((float(i)/float(n))*np.pi) for i in points]
        points = [np.sin(float(i) * np.pi) * self.points_multiplier for i in points]
        points = points
        return points + self.noise

    def linear_response_func(self, points):
        return points * self.points_multiplier + self.noise


# utkarsha's person


class CurtailandShiftPerson(Person):
    def __init__(self, baseline_energy_df, points_multiplier=1):
        super().__init__(baseline_energy_df, points_multiplier)
        self.shiftableLoadFraction = 0.1
        self.shiftByHours = 3
        self.curtailableLoadFraction = 0.1
        self.maxCurtailHours = (
            3  # Person willing to curtail for no more than these hours
        )

    def shiftedLoad(self, points, baseline_day=0, day_of_week=None):
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        points = np.array(points) * self.points_multiplier
        shiftableLoad = self.shiftableLoadFraction * output
        shiftByHours = self.shiftByHours

        # 10 hour day. Rearrange the sum of shiftableLoad into these hours by treating points as the 'price' at that hour
        # Load can be shifted by a max of shiftByHours (default = 3 hours)
        # For each hour, calculate the optimal hour to shift load to within +- 3 hours
        shiftedLoad = np.zeros(10)
        for hour in range(10):
            candidatePrices = points[
                max(hour - shiftByHours, 0) : min(hour + shiftByHours, 9) + 1
            ]
            shiftToHour = max(hour - shiftByHours, 0) + np.argmin(candidatePrices)
            shiftedLoad[shiftToHour] += shiftableLoad[hour]
        return shiftedLoad

    def curtailedLoad(self, points, baseline_day=0, day_of_week=None):
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        points = np.array(points) * self.points_multiplier
        curtailableLoad = self.curtailableLoadFraction * output
        maxPriceHours = np.argsort(points)[0 : self.maxCurtailHours]
        for hour in maxPriceHours:
            curtailableLoad[hour] = 0
        return curtailableLoad

    def get_response(self, points, day_of_week=None):
        baseline_day = 0
        output = np.array(self.baseline_energy)[
            baseline_day * 24 : baseline_day * 24 + 10
        ]
        energy_resp = (
            output * (1 - self.curtailableLoadFraction - self.shiftableLoadFraction)
            + self.curtailedLoad(points)
            + self.shiftedLoad(points)
        )

        self.min_demand = np.maximum(0, min(energy_resp))
        self.max_demand = np.maximum(0, max(energy_resp))

        return energy_resp

