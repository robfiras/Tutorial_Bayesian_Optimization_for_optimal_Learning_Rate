import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor


class BayesianOptimizer:
    """
    This is a Bayesian Optimizer, which takes in a function to optimize, and finds the
    maximum value of a parameter within a bounded search space. It uses Expected Improvement as the
    acquisition function.

    Attributes
    ----------
    f: function
        Function to optimize.

    gp: GaussianProcessRegressor
        Gaussian Process used for regression.

    mode: str
        Either "linear" or "logarithmic".

    bound: list
        List containing the lower and upper bound of the search space. IMPORTANT: If mode is "logarithmic",
        the bound specifies the minimum and maximum exponents!

    size_search_space: int
        Number of evaluation points used for finding the maximum of the acquisition function. Can be interpreted
        as the size of our discrete search space.

    search_space: ndarray
        Vector covering the search space.

    gp_search_space: ndarray
        The search space of GP might be transformed logarithmically depending on the mode, which is why it
        might differ from our defined search space.

    dataset: list
        List containing all data samples used for fitting (empty at the beginning).

    states: list
        List containing the state of each iteration in the optimization process (used for later plotting).
    """

    def __init__(self, f, gp, mode, bound, size_search_space=250):
        if mode not in ["linear", "logarithmic"]:
            raise ValueError("%s mode not supported! Chose either linear or logarithmic." % mode)
        else:
            self.mode = mode
        self.f = f
        self.gp = gp
        self.min = bound[0]
        self.max = bound[1]
        self.size_search_space = size_search_space
        if mode == "linear":
            self.search_space = np.linspace(self.min, self.max, num=size_search_space).reshape(-1, 1)
            self.gp_search_space = self.search_space
        else:
            self.search_space = np.logspace(self.min, self.max, num=size_search_space).reshape(-1, 1)
            self.gp_search_space = np.log10(self.search_space)
        self.dataset = []
        self.states = []

    def _ei(self, c_inc, xi=0.05):
        """
        Expected Improvement (EI) acquisition function used for maximization.

        Parameters
        ----------
        c_inc: float
            Utility of current incumbent.

        xi: float
            Optional exploration parameter.

        Returns
        -------
        util: ndarray
            Utilization given the current Gaussian Process and incumbent
        """
        # calculate the current mean and std for the search space
        mean, std = self.gp.predict(self.gp_search_space, return_std=True)
        std = np.array(std).reshape(-1, 1)

        # calculate the utilization
        a = (mean - c_inc - xi)
        z = a / std
        util = a * norm.cdf(z) + std * norm.pdf(z)
        return util

    def _max_acq(self):
        """
        Calculates the next best incumbent for the current dataset D.

        Returns
        -------
        x_max: float
            Location (x-coordinate) of the next best incumbent

        util_max: float
            Utility of the next best incumbent.

        util: ndarray
            Utility function for the search space.
        """
        # get the value of the current best incumbent
        c_inc = np.max(np.array(self.dataset)[:, 1])

        # calculate the utility function
        util = self._ei(c_inc)

        # check if the utilization is all zero
        if np.all((util == 0.)):
            print("Warning! Utilization function is all zero. Returning a random point for evaluation.")
            x_max = self.search_space.reshape(-1)[np.random.randint(len(self.search_space))]
            util_max = 0.0
        else:
            # get the maximum's location and utility
            x_max = self.search_space.reshape(-1)[util.argmax()]
            util_max = util.max()

        return x_max, util_max, util

    def eval(self, n_iter=10, init_x_max=None):
        """
        Runs n_iter evaluations of function f and optimizes its parameter using Bayesian Optimization.

        Parameters
        ----------
        n_iter: int
            Number of iterations used for optimization

        init_x_max: float
            Initial guess of the parameter. If none, a random initial guess is sampled in the search space.

        Returns
        -------
        best_return_x: float
            Best sample found during optimization

        best_return_param:
            Parameters defining the best function (e.g., torch model).
        """

        # get a random initial value for the incumbent from our search space if not specified
        if not init_x_max:
            x_max = self.search_space[np.random.randint(len(self.search_space))]
            x_max = x_max.item()
        else:
            x_max = init_x_max

        # for storing the best return and some parameters specifying it
        best_return = None
        best_return_x = None
        best_return_param = None

        for i in range(n_iter):

            # print some information
            print("\nBO Iteration %d --> Chosen parameter: %f %s" % (i, x_max,
                                                                     "" if (init_x_max or i != 0) else "(randomly)"))
            # evaluate the function
            y, param = self.f(x_max)

            # store if it is the best
            if not best_return or y > best_return:
                best_return = y
                best_return_x = x_max
                best_return_param = param

            # add the new sample to the dataset
            self.dataset.append([x_max, y])

            # get all the data samples in the dataset
            xs = np.array(self.dataset)[:, 0].reshape(-1, 1)
            ys = np.array(self.dataset)[:, 1].reshape(-1, 1)

            # fit the GP with the updated dataset
            if self.mode == "linear":
                self.gp.fit(xs, ys)
            else:
                self.gp.fit(np.log10(xs), ys)

            # calculate the maximum utilization and its position
            x_max, util_max, util = self._max_acq()

            # save the state for later plotting
            self.states.append({"dataset": self.dataset.copy(),
                                "util": util,
                                "GP": self.gp.predict(self.gp_search_space, return_std=True)})

        return best_return_x, best_return_param

    def save_all_plots(self):
        """
        Saves all plots.
        """
        self.plot_all(show=False, save=True)

    def plot_all(self, show=True, save=True):
        """
        Plots all states/iterations made during optimization until now.

        Parameters
        ----------
        show: bool
            If true, plot is shown directly.

        save: bool
            If true, plot is saved.
        """
        for id, state in enumerate(self.states):
            self.plot_state(state, id, show=False, save=save)
        if show:
            plt.show()

    def plot_iteration(self, it, show=True, save=True):
        """
        Plots a certain iteration of the optimization process.

        Parameters
        ----------
        it: int
            Iteration of the optimization process

        show: bool
            If true, plot is shown directly.

        save: bool
            If true, plot is saved.
        """
        # get the corresponding state
        state = self.states[it]
        self.plot_state(state, it, show=show, save=save)

    def plot_state(self, state, fig_number, show=True, save=True, additional_func=None):
        """
        Plots a state of the optimization process.

        Parameters
        ----------
        state: dict
            Dictionary storing the dataset, utilization and GP describing one state during optimization.

        fig_number: int
            Id of the figure to plot.

        show: bool
            If true, plot is shown directly.

        save: bool
            If true, plot is saved.

        additional_func: (function, name)
            Additional function to plot.
        """

        # reshape search space as this is more convenient for plotting
        search_space = self.search_space.reshape(-1)

        # get all information of the corresponding state
        dataset = state["dataset"]
        util = state["util"].reshape(-1)
        gp = state["GP"]

        # create figure with two plots (ax1: GP fitting, ax2: utility function)
        figure = plt.figure(fig_number)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=figure)
        ax1 = figure.add_subplot(gs[0])
        ax1.set_xticklabels([])  # turn off x labeling of upper plot
        ax1.set_title("Iteration %d" % fig_number)
        ax2 = figure.add_subplot(gs[1])

        # check if we need to set a logarithmic scale
        if self.mode == "logarithmic":
            ax1.set_xscale("log")
            ax2.set_xscale("log")

        # adjust borders to make it look better
        figure.subplots_adjust(bottom=.14, top=.95)

        # plot an additional function if given
        if additional_func:
            func, name = additional_func
            add_ys, _ = func(search_space)
            ax1.plot(search_space, add_ys, color="red", label=name)

        # plot the GP mean and std
        mu, std = gp
        mu = mu.reshape(-1)
        ax1.plot(search_space, mu,
                 color="blue", label="GP mean")
        ax1.fill_between(search_space,
                         mu - (std * 1), mu + (std * 1),
                         color="blue", alpha=0.3, label="GP std")

        # plot the dataset
        xs = np.array(dataset)[:, 0]
        ys = np.array(dataset)[:, 1]
        ax1.scatter(xs, ys, color="blue", label="Dataset")

        # plot the utility function
        ax2.plot(search_space, util, color="green", label="Utility function")
        ax2.fill_between(search_space,
                         np.zeros_like(util),
                         util.reshape(-1), alpha=0.3, color="green")

        figure.legend(loc="lower center", ncol=5 if additional_func else 4)

        if save:
            if not os.path.exists('./plots'):
                os.makedirs('./plots')
            fig_name = "./plots/BO_iteration_%d" % fig_number
            plt.savefig(fig_name)
        if show:
            plt.show()
