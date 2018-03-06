import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import descartes
import sys, os, datetime, collections
import open_cp.sources.chicago
import open_cp.geometry
import open_cp.plot
import open_cp.predictors
import open_cp.evaluation
import open_cp.kernels
import open_cp.kde
import pandas as pd
import scipy.stats


def load_geometry(datadir, side="North"):
    open_cp.sources.chicago.set_data_directory(datadir)
    return open_cp.sources.chicago.get_side(side)

def make_grid(geo):
    grid = open_cp.data.Grid(150, 150, 0, 0)
    return open_cp.geometry.mask_grid_by_intersection(geo, grid)

def load(datadir, side="North"):
    """Load and compute initial data.

    :return: `(geometry, grid)`
    """
    chicago = load_geometry(datadir, side)
    return chicago, make_grid(chicago)


class BaseModel():
    def __init__(self, grid):
        self._grid = grid
    
    def prob(self, x, y):
        want, pt = self._in_grid(x, y)
        if want:
            return self._probs[pt[1], pt[0]]
        return 0.0
        
    def max_prob(self):
        return np.max(np.ma.array(self._probs, mask=self._grid.mask))
        
    @property
    def grid(self):
        return self._grid
        
    def to_grid_coords(self, x, y):
        pt = np.asarray([x,y]) - [self.grid.xoffset, self.grid.yoffset]
        pt = np.floor_divide(pt, [self.grid.xsize, self.grid.ysize]).astype(np.int)
        return pt

    def in_grid(self, x, y):
        pt = self.to_grid_coords(x, y)
        if np.all((pt >= [0, 0]) & (pt < [self.grid.xextent, self.grid.yextent])):
            return self.grid.is_valid(*pt)
        return False
    
    def to_prediction(self):
        mat = np.array(self._probs)
        pred = open_cp.predictors.GridPredictionArray(self.grid.xsize, self.grid.ysize,
                                        mat, self.grid.xoffset, self.grid.yoffset)
        pred.mask_with(self.grid)
        return pred
    
    def map_to_grid(self, x, y):
        """Map a point in :math:`[0,1]^2` to the bounding box of the grid"""
        pt = np.asarray([x,y])
        pt = pt * [self._grid.xsize * self._grid.xextent, self._grid.ysize * self._grid.yextent]
        return pt + [self._grid.xoffset, self._grid.yoffset]
    
    def _in_grid(self, x, y):
        pt = self.to_grid_coords(x, y)
        if np.all((pt >= [0, 0]) & (pt < [self.grid.xextent, self.grid.yextent])):
            return self.grid.is_valid(*pt), pt
        return False, pt
    
    def _set_probs(self, probs):
        probs = self.grid.mask_matrix(probs)
        self._probs = probs / np.sum(probs)


class Model1(BaseModel):
    """Homogeneous poisson process."""
    def __init__(self, grid):
        super().__init__(grid)
        probs = np.zeros((grid.yextent, grid.xextent)) + 1
        self._set_probs(probs)
        
    def to_randomised_prediction(self):
        mat = np.array(self._probs)
        mat += np.random.random(mat.shape) * 1e-7
        pred = open_cp.predictors.GridPredictionArray(self.grid.xsize, self.grid.ysize,
                                        mat, self.grid.xoffset, self.grid.yoffset)
        pred.mask_with(self.grid)
        pred = pred.renormalise()
        return pred
    
    def __str__(self):
        return "Model1"


class Model2(BaseModel):
    """Inhomogeneous Poisson process, on `[0,1]^2` has intensity

      :math:`0.01 + \exp(-30(x-y)^2) + \exp(-100((x-0.8)^2+(y-0.2)^2))`

    then spread out to the grid.
    """
    def __init__(self, grid):
        super().__init__(grid)
        probs = np.empty((grid.yextent, grid.xextent))
        for x in range(grid.xextent):
            for y in range(grid.yextent):
                probs[y,x] = self.cts_prob((x+0.5) / grid.xextent, (y+0.5) / grid.yextent)
        self._set_probs(probs)
        
    def cts_prob(self, x, y):
        return 0.01 + np.exp(-(x-y)**2 * 30) + np.exp(-((x-0.8)**2 + (y-0.2)**2) * 100)
    
    def __str__(self):
        return "Model2"


def multiscale_brier(pred, tps):
    """Score the prediction using multiscale Brier."""
    maxsize = min(pred.xextent, pred.yextent)
    return {s : open_cp.evaluation.multiscale_brier_score(pred, tps,s) for s in range(1, maxsize+1)}

def multiscale_kl(pred, tps):
    """Score the prediction using multiscale Kullback-Leibler."""
    maxsize = min(pred.xextent, pred.yextent)
    return {s : open_cp.evaluation.multiscale_kl_score(pred, tps,s) for s in range(1, maxsize+1)}

def sample(model, size):
    """Return points from the model.

    :param model: The model to use to sample from.
    :param size: Number of points to return.

    :return: Array of shape `(size, 2)`.
    """
    out = []
    renorm = model.max_prob()
    while len(out) < size:
        pt = model.map_to_grid(*np.random.random(2))
        if model.in_grid(*pt):
            if model.prob(*pt) > np.random.random() * renorm:
                out.append(pt)
    return np.asarray(out)

def sample_to_timed_points(model, size):
    """As :func:`sample` but return in :class:`open_cp.data.TimedPoints`.
    """
    t = [datetime.datetime(2017,1,1)] * size
    pts = sample(model, size)
    assert pts.shape == (size, 2)
    return open_cp.data.TimedPoints.from_coords(t, *pts.T)

def make_data_preds(grid, num_trials=1000, base_intensity=10):
    """Make the data in one go.

    :param grid: The masked grid to base everything on.
    :param num_trials: The number of trials to run.  Will not yield empty point
      collections, so may return fewer than this many tuples.
    :param base_intensity: Each trial has `n` events where `n` is distributed
      as a Poisson with mean `base_intensity`.

    :return: A dictionary from `key` to a list of pairs `(pred, tps)`.  Here
      `key` is the pair `(source_model_name, pred_model_name)`, `pred` is the
      prediction as a :class:`GridPredictionArray` instance, and `tps` is a
      :class:`TimedPoints` instance.
    """
    predictions = collections.defaultdict(list)
    for trial in range(num_trials):
        for SourceModel in [Model1, Model2]:
            source_model = SourceModel(grid)
            while True:
                num_pts = np.random.poisson(base_intensity)
                if num_pts > 0:
                    break
            tps = sample_to_timed_points(source_model, num_pts)
            for PredModel in [Model1, Model2]:
                pred_model = PredModel(grid)
                pred = pred_model.to_prediction()
                #try:
                #    pred = pred_model.to_randomised_prediction()
                #except:
                #    pass
                pred = pred.break_ties()
                key = (str(source_model), str(pred_model))
                predictions[key].append((pred, tps))
        print("Done {}/{}".format(trial, num_trials), file=sys.__stdout__)
    return predictions

def process(all_data, func):
    """`func` should be a function object with signature `func(pred, tps)`, and
    should return some object.

    :param all_data: The output of :func:`make_data_preds`.
    
    :return: A dictionary from `key` to `object` where each key will be `(k1,
      k2, i)` with `k1,k2` as before, and `i` a counter.  `object` is the
      return value of func.
    """
    return { (key) + (i,) : func(pred, tps) for key in all_data
            for i, (pred, tps) in enumerate(all_data[key]) }

def constrain_to_number_events(all_data, minimum_event_count):
    """Remove trials with too few events.

    :param all_data: The output of :func:`make_data_preds`; or data conforming
      to this dictionary style.
    :param minimum_event_count: The minimum number of events we want in a
      trial.
      
    :return: The same format of data, but each "actual occurrance" will have at
      least `minimum_event_count` events.
    """
    return {k:[(g,tps) for g,tps in v if tps.number_data_points >= minimum_event_count]
               for k,v in all_data.items()}

def plot_models(data, func, fig):
    """Helper method to plot a 2 by 2 grid of results.

    :param data: A dictionary from `key` to object.  Each `key` is a tuple
      `(k1, k2, *)` where `k1` is the source name (which model generated the
      points) and `k2` is the prediction name (which model generated the
      prediction).
    :param func: A function object with signature  `func(result, ax, key)`
      where `result` is a list of the objects passed in `data` which have
      appropriate `k1` and `k2`; `ax` is the `matplotlib` `axis` object to
      plot to; `key` is `(k1,k2)`.  This function should process the object
      appropriately, and draw to the axis.
    :param fig: `None` to make a new figure, or the figure which should have 4
      axes.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,12))
    else:
        axes = np.asarray(fig.axes).reshape((2,2))
    labels = ["Model1", "Model2"]
    for axe, k1 in zip(axes, labels):
        for ax, k2 in zip(axe, labels):
            ax.set_title("{}/{}".format(k1,k2))
            result = [data[k] for k in data if k[:2] == (k1, k2)]
            func(result, ax, (k1, k2))
    return fig

def data_by_source_model(data):
    """Assuming `data` is as in the format returned from :func:`process`,
    yield data in the same format, first from source model 1, and then model 2,
    (with the key now being `(prediction_model, trial_number)`).
    """
    for key in ["Model1", "Model2"]:
        yield {k[1:] : v for k, v in data.items() if k[0] == key}
        
PairedItem = collections.namedtuple("PairedItem", "key one two")
        
def paired_data(data_for_one_source):
    """Assuming `data_for_one_source` is as returned by
    :func:`data_by_source_model`, return a list of :class:`PairedItem` objects,
    showing model1 compared to model2.
    """
    keys = {k[1:] for k in data_for_one_source if k[0] == "Model1"}
    keys.intersection_update({k[1:] for k in data_for_one_source if k[0] == "Model2"})
    keys = list(keys)
    keys.sort()
    return [PairedItem(k, data_for_one_source[("Model1",*k)], data_for_one_source[("Model2",*k)]) for k in keys]

def plot_paired_data(data, plot_func, fig=None, **kwargs):
    """Plot a row of plots, comparing model 1 with model 2.
    
    :param data: As before, return of :func:`process`.
    :param plot_func:` Should have signature `plot_func(ax, paired_data)`
      where `paired_data` is a list of :class:`PairedItem` objects.
    :param fig: `None` to auto-create, or a figure with 2 axes.
    """
    if fig is None:
        fig, axes = plt.subplots(ncols=2, figsize=(16,6))
    else:
        axes = np.asarray(fig.axes)
    models = ["Model1", "Model2"]
    for ax, row_data, source_name in zip(axes, data_by_source_model(data), models):
        d = paired_data(row_data)
        plot_func(ax, d, **kwargs)
        ax.set_title("Samples from {}".format(source_name))
    return fig, axes

def comparison_uni_paired(lower_bound, upper_bound):
    """Returns a `plot_func` which is suitable for passing to
    `plot_paired_data`.  Assumes that the "objects" returned from
    :func:`process` are just numbers, and plots the difference between "one"
    and "two" in the :class:`PairedItem` instance.  Plots an estimate of the
    cumulative density.
    
    :param lower_bound:
    :param upper_bound: `x` axis range.
    """
    def plot_func(ax, data, **kwargs):
        diff = np.asarray([d.one - d.two for d in data])
        cumulative_df_plot_func(ax, diff, lower_bound, upper_bound, **kwargs)
    return plot_func

def cumulative_df_plot_func(ax, diff, lower_bound, upper_bound, midpoint=0, **kwargs):
    """A "plot function" which finds the CDF of the input.

    :param diff: An array or list of values.
    """
    diff = np.asarray(diff)
    y = np.sum(diff <= midpoint) / len(diff)
    lc = matplotlib.collections.LineCollection([[[lower_bound,y],[midpoint,y]], [[midpoint,0],[midpoint,y]]],
        color="black", linewidth=1, linestyle="--")
    ax.add_collection(lc)

    x = np.linspace(lower_bound, upper_bound, 100)
    cumulative = []
    for cutoff in x:
        cumulative.append( np.sum(diff <= cutoff) )
    ax.plot(x, np.asarray(cumulative) / len(diff), color="black", **kwargs)

def scatter_uni_paired_plot_func(ax, data):
    """A `plot_func` which is suitable for passing to `plot_paired_data`.
    Assumes that the "objects" returned from :func:`process` are just numbers,
    and plots a scatter diagram."""
    x = [d.one for d in data]
    y = [d.two for d in data]
    ax.scatter(x, y, marker="x", color="black", linewidth=1)
    start = min(min(x), min(y))
    end = max(max(x), max(y))
    d = (end - start) * 0.1
    ax.plot([start-d, end+d], [start-d, end+d], color="black", linewidth=1, linestyle="--")

def scatter_uni_paired_plot_func_diffs(ax, data):
    """A `plot_func` which is suitable for passing to `plot_paired_data`.
    Assumes that the "objects" returned from :func:`process` are just numbers,
    and plots a scatter diagram.  Plots x against (y-x) instead of x against y.
    """
    x = [d.one for d in data]
    y = [d.two - d.one for d in data]
    ax.scatter(x, y, marker="x", color="black", linewidth=1)
    start = min(min(x), min(y))
    end = max(max(x), max(y))
    d = (end - start) * 0.1
    ax.plot([start-d, end+d], [0, 0], color="black", linewidth=1, linestyle="--")

def label_scatter_uni_paired(fig, axes):
    """Suitable for calling after using :func:`scatter_uni_paired_plot_func`.
    Sets labels on each axis."""
    for ax in axes:
        ax.set(xlabel="Model1", ylabel="Model2")

def joint_paired_data(data, func1, func2, fig=None):
    """Call :func:`plot_paired_data` twice, once with `func1`, and once with
    `func2` for the 2nd row of axes."""
    if fig is None:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,12))
    else:
        axes = np.asarray(fig.axes).reshape((2,2))
    AC = collections.namedtuple("AxesContainer", ["axes"])
    plot_paired_data(data, func1, AC(axes[0]))
    plot_paired_data(data, func2, AC(axes[1]))
    return fig

def three_up_plot(data, lower_bound, upper_bound):
    """Plots scatter plots for data from model1, model2, and then a CDF plot with
    turn curves.

    :param data: As from :func:`process`.
    :param lower_bound:
    :param upper_bound: For passing to :func:`comparison_uni_paired`

    :return: The figure.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))
    for one_model, ax in zip(data_by_source_model(data), axes):
        scatter_uni_paired_plot_func(ax, paired_data(one_model))
    for ax in axes[:2]:
        ax.set(xlabel="Model 1", ylabel="Model 2")
    axes[0].set_title("Samples from Model 1")
    axes[1].set_title("Samples from Model 2")
    func = comparison_uni_paired(lower_bound, upper_bound)
    for one_model, lab, ls in zip(data_by_source_model(data), ["Model 1", "Model 2"],
                                 ["-", "--"]):
        func(axes[2], paired_data(one_model), label=lab, linestyle=ls)
    axes[2].legend(loc="lower right")
    axes[2].set_title("Difference between models 1 and 2")
    return fig

def add_kernels(ax, data, xrange=(0, 1), yrange=1, **kwargs):
    """Add kde estimates as extra axes to the plot, which is expected to be a
    scatter plot, as from :func:`scatter_uni_paired_plot_func`.

    :param ax: The `Axes` object to add to.
    :param data: Paired data, as from :func:`paired_data`
    :param xrange: A pair `(xmin, xmax)` giving the range to compute the
      kernel over.
    :param yrange: The expected "height" of the kernel.

    :return: Pair `(ax1, ax2)` where `ax1` is the new `Axes` object for the
      Model1 data, and similarly for `ax2`.  
    """
    ker = scipy.stats.kde.gaussian_kde([x.one for x in data])
    x = np.linspace(*xrange, 100)
    ax2 = ax.twinx()
    ax2.plot(x, ker(x), **kwargs)
    ax2.set(ylim=[0, yrange])
    ax2.yaxis.set_visible(False)

    ker = scipy.stats.kde.gaussian_kde([x.two for x in data])
    x = np.linspace(*xrange, 100)
    ax3 = ax.twiny()
    ax3.plot(ker(x), x, **kwargs)
    ax3.set(xlim=[0, yrange])
    ax3.xaxis.set_visible(False)

    return ax2, ax3


#############################################################################
# More tentative, data vis stuff.
#############################################################################

def hitrate_inverse_to_hitrate(inv_dict, coverages):
    """Convert the "inverse hitrate dictionary" to a more traditional
    lookup, using `coverages`."""
    out = dict()
    for cov in coverages:
        choices = [k for k,v in inv_dict.items() if v <= cov]
        out[cov] = 0 if len(choices) == 0 else max(choices)
    return out

def plot_hit_rate(data, fig=None):
    """Draws a 2x2 grid; plots the mean and 25%, 75% percentiles of coverage
    against hit rate."""
    coverages = list(range(0,102,2))

    def to_by_coverage(result):
        by_cov = {cov : [] for cov in coverages}
        for row in result:
            out = hitrate_inverse_to_hitrate(row, coverages)
            for c in out:
                by_cov[c].append(out[c])
        return by_cov

    def plot_func(result, ax, key):
        frame = pd.DataFrame(to_by_coverage(result)).describe().T
        ax.plot(frame["mean"], label="mean", color="black", linestyle="-")
        ax.plot(frame["25%"], label="25% percentile", color="black", linestyle="--")
        ax.plot(frame["75%"], label="75% percentile", color="black", linestyle="-.")
        ax.set(xlabel="Coverage (%)", ylabel="Hit rate (%)")
        ax.set(xlim=[-5,105], ylim=[-5,105])
        ax.legend()
    
    return plot_models(data, plot_func, fig)
