import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import errno

from enum import Enum


BASE_DIR = 'conf_levels/'
SCENARIO_NAME = 'vec-fasttext-100'


def setup_input_files():
    correct_file_base = 'conf-correct-predictions-'
    error_file_base = 'conf-error-predictions-'
    file_paths = [
        (BASE_DIR + 'scenario0-' + SCENARIO_NAME + '/' + correct_file_base + '0-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario0-' + SCENARIO_NAME + '/' + error_file_base + '0-' + SCENARIO_NAME + '.json',
         '0'),
        (BASE_DIR + 'scenario1-' + SCENARIO_NAME + '/' + correct_file_base + '1-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario1-' + SCENARIO_NAME + '/' + error_file_base + '1-' + SCENARIO_NAME + '.json',
         '1'),
        (BASE_DIR + 'scenario2-' + SCENARIO_NAME + '/' + correct_file_base + '2-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario2-' + SCENARIO_NAME + '/' + error_file_base + '2-' + SCENARIO_NAME + '.json',
         '2'),
        (BASE_DIR + 'scenario31-' + SCENARIO_NAME + '/' + correct_file_base + '31-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario31-' + SCENARIO_NAME + '/' + error_file_base + '31-' + SCENARIO_NAME + '.json',
         '3.1'),
        (BASE_DIR + 'scenario32-' + SCENARIO_NAME + '/' + correct_file_base + '32-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario32-' + SCENARIO_NAME + '/' + error_file_base + '32-' + SCENARIO_NAME + '.json',
         '3.2'),
        (BASE_DIR + 'scenario33-' + SCENARIO_NAME + '/' + correct_file_base + '33-' + SCENARIO_NAME + '.json',
         BASE_DIR + 'scenario33-' + SCENARIO_NAME + '/' + error_file_base + '33-' + SCENARIO_NAME + '.json',
         '3.3'),
    ]
    return file_paths


class FileContents(Enum):
    correct = 'correct'
    errors = 'errors'

    def __str__(self):
        return self.value


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_histogram(confidence_levels, plot_filename, title=''):
    if type(confidence_levels) == list:
        bins = np.linspace(0, 100, 25)
        plt.hist(confidence_levels, bins, edgecolor='k')
    elif type(confidence_levels) == dict:
        bins = np.linspace(0, 100, 5)
        plt.hist(confidence_levels.values(), bins, edgecolor='k', label=confidence_levels.keys())
        plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    # plt.show()

    fname = '{}.png'.format(plot_filename)
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    plt.savefig(fname)
    plt.clf()


def plot_conf_all_intents(results, plot_filename, hist_title=''):
    prediction_conf_levels = []
    for ex in results:
        pred_intent = ex['predictedIntent']
        pred_conf = ex['confidenceList'][pred_intent]
        prediction_conf_levels.append(pred_conf * 100)
    plot_histogram(prediction_conf_levels, plot_filename, hist_title)


def average(lst):
    return sum(lst) / len(lst)


def plot_conf_matrix(results, plot_filename, title='', only_true_vs_pred=True):
    '''

    Args:
        results:
        plot_filename:
        hist_title:
        only_true_vs_pred: if this is true, the confidence matrix will have the confidence levels for
            those examples that had the true intent as x but were predicted as y (x - rows and y - cols
            in the matrix. If this is false, the confidence matrix will look at all reported confidence levels
            (e.g. a particular example will have one confidence level reported for each possible intent)

    Returns:
    '''

    conf_dict = dict()
    intents_set = set()
    for ex in results:
        intents_set.add(ex['trueIntent'])
        if ex['trueIntent'] not in conf_dict:
            conf_dict[ex['trueIntent']] = dict()
        intents_conf = ex['confidenceList']
        if only_true_vs_pred:
            pred_intent = ex['predictedIntent']
            intents_set.add(pred_intent)
            if pred_intent not in conf_dict[ex['trueIntent']]:
                conf_dict[ex['trueIntent']][pred_intent] = []
            conf = intents_conf[pred_intent]
            conf_dict[ex['trueIntent']][pred_intent].append(conf * 100)
        else:
            for intent, conf in intents_conf.items():
                if intent not in conf_dict[ex['trueIntent']]:
                    conf_dict[ex['trueIntent']][intent] = []
                conf_dict[ex['trueIntent']][intent].append(conf * 100)

    for true_intent, confs in conf_dict.items():
        for potential_intent, conf_list in confs.items():
            confs[potential_intent] = average(conf_list)

    intents = list(intents_set)
    intents.sort()
    confidences = [[0] * len(intents) for _ in intents]
    i = 0
    for t_intent in intents:
        j = 0
        for p_intent in intents:
            if (t_intent != p_intent) and t_intent in conf_dict and (p_intent in conf_dict[t_intent]):
                confidences[i][j] = conf_dict[t_intent][p_intent]
            j += 1
        i += 1

    np_confidences = np.array(confidences)

    fig, ax = plt.subplots(figsize=(20, 10))
    im, cbar = heatmap(np_confidences, intents, intents, ax=ax,
                       cmap="YlGn", cbarlabel="average confidence")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.title(title)

    fname = '{}.png'.format(plot_filename)
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.savefig(fname)
    plt.clf()
    # plt.show()


def plot_conf_levels(input_path, plot_filename, scenario_num, file_content, plot_matrix=False):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        results = json.load(f)
    title_prefix = 'Correct predictions sc ' + scenario_num
    if file_content is FileContents.errors:
        title_prefix = 'Error predictions sc ' + scenario_num
    plot_conf_all_intents(results, plot_filename.format('overall'), title_prefix + ' - overall')
    if plot_matrix:
        plot_conf_matrix(results, plot_filename.format('matrix'), title_prefix + ' - confidence matrix')


def main():
    files = setup_input_files()

    for corr, err, sc_nr in files:
        plots_base_dir = corr.split('/')[:-1]
        plots_base_dir = '/'.join(plots_base_dir) + '/plots/{}-'
        plot_conf_levels(corr, plots_base_dir + corr.split('/')[-1][:-5], sc_nr, FileContents.correct)
        plot_conf_levels(err, plots_base_dir + err.split('/')[-1][:-5], sc_nr, FileContents.errors, plot_matrix=True)


if __name__ == '__main__':
    main()

