import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import errno

from test import INTENT_CLASSES, INTENTS_ORDER

from enum import Enum


BASE_DIR = 'conf_levels/'
SCENARIO_NAME = 'vec-fasttext-100'
SLOT_ERRS_BASE_DIR = 'slot-errors/'
INTENT_ERRS_BASE_DIR = 'intent-error-categories/'

# ERR1 = Opposite intents
ERR1_INTENT_PAIRS = [
    ('aprindeLumina', 'stingeLumina'),
    ('cresteIntensitateLumina', 'scadeIntensitateLumina'),
    ('cresteTemperatura', 'scadeTemperatura'),
    ('pornesteTV', 'opresteTV'),
    ('puneMuzica', 'opresteMuzica'),
    ('cresteIntensitateMuzica', 'scadeIntensitateMuzica'),
]

# ERR2 = Same action, diff object
ERR2_INTENT_PAIRS = [
    ('aprindeLumina', 'pornesteTV'),
    ('stingeLumina', 'opresteTV'),
    ('opresteMuzica', 'opresteTV'),
    ('pornesteTV', 'puneMuzica'),
    # The next ones don't really appear too much, we can try to comment them out
    ('aprindeLumina', 'puneMuzica'),
    ('stingeLumina', 'opresteMuzica'),
    ('cresteIntensitateLumina', 'cresteIntensitateMuzica'),
    ('scadeIntensitateLumina', 'scadeIntensitateMuzica'),
    ('cresteIntensitateLumina', 'cresteTemperatura'),
    ('scadeIntensitateLumina', 'scadeTemperatura'),
    ('cresteTemperatura', 'cresteIntensitateMuzica'),
    ('scadeTemperatura', 'scadeIntensitateMuzica'),
]

# ERR3 = Same class of intents, same object of an action but the actions themselves are not quite opposites
ERR3_INTENT_PAIRS = [
    ('cresteTemperatura', 'seteazaTemperatura'),
    ('scadeTemperatura', 'seteazaTemperatura'),
    ('pornesteTV', 'schimbaCanalTV'),
]


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
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticklabels(row_labels, fontsize=13)

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
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()
    # plt.show()


def plot_conf_all_intents(results, plot_filename, hist_title=''):
    prediction_conf_levels = []
    for ex in results:
        pred_intent = ex['predictedIntent']
        pred_conf = ex['confidenceList'][pred_intent]
        prediction_conf_levels.append(pred_conf * 100)
    plot_histogram(prediction_conf_levels, plot_filename, hist_title)


def average(lst):
    return sum(lst) / len(lst)


def valid_intent(intent_name):
    return (intent_name[-2:] != 'Nr' and intent_name != 'nrTrue'
            and '-avg' not in intent_name and '-normalizedAvg' not in intent_name)


def get_conf_matrix(conf_dict, intents, value_key_prefix):
    confidences = [[0] * len(intents) for _ in intents]
    i = 0
    for t_intent in intents:
        j = 0
        for p_intent in intents:
            if (t_intent != p_intent) and t_intent in conf_dict and (p_intent in conf_dict[t_intent]):
                confidences[i][j] = conf_dict[t_intent][p_intent + value_key_prefix]
            j += 1
        i += 1

    np_confidences = np.array(confidences)
    return np_confidences


def do_conf_matrix_plot(confidences, intents, plot_filename, title='', cbarlabel="average confidence"):
    fig, ax = plt.subplots(figsize=(20, 10))
    im, cbar = heatmap(confidences, intents, intents, ax=ax,
                       cmap="YlGn", cbarlabel=cbarlabel)
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

    plt.savefig(fname, bbox_inches='tight')
    plt.clf()
    # plt.show()


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
            conf_dict[ex['trueIntent']]['nrTrue'] = 0
        intents_conf = ex['confidenceList']
        conf_dict[ex['trueIntent']]['nrTrue'] += 1
        if only_true_vs_pred:
            pred_intent = ex['predictedIntent']
            intents_set.add(pred_intent)
            if pred_intent not in conf_dict[ex['trueIntent']]:
                conf_dict[ex['trueIntent']][pred_intent] = []
                conf_dict[ex['trueIntent']][pred_intent + 'Nr'] = 0
                conf_dict[ex['trueIntent']][pred_intent + '-avg'] = 0
                conf_dict[ex['trueIntent']][pred_intent + '-normalizedAvg'] = 0
            conf = intents_conf[pred_intent]
            conf_dict[ex['trueIntent']][pred_intent + 'Nr'] += 1
            conf_dict[ex['trueIntent']][pred_intent].append(conf * 100)
        else:
            for intent, conf in intents_conf.items():
                if intent not in conf_dict[ex['trueIntent']]:
                    conf_dict[ex['trueIntent']][intent] = []
                conf_dict[ex['trueIntent']][intent].append(conf * 100)

    for true_intent, confs in conf_dict.items():
        for potential_intent, conf_list in confs.items():
            if valid_intent(potential_intent):
                confs[potential_intent + '-avg'] = average(conf_list)
                confs[potential_intent + '-normalizedAvg'] = average(conf_list) * confs[potential_intent + 'Nr'] / confs['nrTrue']

    intents = [x for x in INTENTS_ORDER if x in intents_set]
    confidences = get_conf_matrix(conf_dict, intents, '-avg')

    do_conf_matrix_plot(confidences, intents, plot_filename, title)

    normalized_confidences = get_conf_matrix(conf_dict, intents, '-normalizedAvg')

    do_conf_matrix_plot(normalized_confidences, intents, plot_filename + '-normalized', title + ' (normalized)')


def plot_conf_levels(input_path, plot_filename, scenario_num, file_content, plot_matrix=False):
    with open(input_path, errors='replace', encoding='utf-8') as f:
        results = json.load(f)
    title_prefix = 'Correct predictions sc ' + scenario_num
    if file_content is FileContents.errors:
        title_prefix = 'Error predictions sc ' + scenario_num
    plot_conf_all_intents(results, plot_filename.format('overall'), title_prefix + ' - overall')
    if plot_matrix:
        plot_conf_matrix(results, plot_filename.format('matrix'), title_prefix + ' - confidence matrix')


def get_err_type(true_intent, pred_intent):
    if (true_intent, pred_intent) in ERR1_INTENT_PAIRS or \
            (pred_intent, true_intent) in ERR1_INTENT_PAIRS:
        return 'ERR1'
    if (true_intent, pred_intent) in ERR2_INTENT_PAIRS or \
            (pred_intent, true_intent) in ERR2_INTENT_PAIRS:
        return 'ERR2'
    if (true_intent, pred_intent) in ERR3_INTENT_PAIRS or \
            (pred_intent, true_intent) in ERR3_INTENT_PAIRS:
        return 'ERR3'
    return 'other'


def classify_err_types(errs):
    errs_dict = dict()
    errs_dict['ERR1'] = []
    errs_dict['ERR2'] = []
    errs_dict['ERR3'] = []
    errs_dict['other'] = []

    for ex in errs:
        true_intent = ex['trueIntent']
        pred_intent = ex['predictedIntent']
        err_type = get_err_type(true_intent, pred_intent)
        errs_dict[err_type].append(ex)
    return errs_dict


def get_incorrect_slots(res):
    slot_errs = 0

    superclass_dict = dict()
    superclass_dict['lumina'] = 0
    superclass_dict['temperatura'] = 0
    superclass_dict['media'] = 0

    slot_err_str = ''
    for ex in res:
        if ex['trueSlots'] != ex['predSlots']:
            slot_err_str += '{}\n'.format(ex['testData'])
            slot_err_str += '{}\n'.format(ex['trueIntent'])
            slot_err_str += '{}\n'.format(ex['trueSlots'])
            slot_err_str += '{}\n\n'.format(ex['predSlots'])
            slot_errs += 1
            superclass_dict[INTENT_CLASSES[ex['trueIntent']]] += 1
    return slot_errs, slot_err_str, superclass_dict


def show_intent_errors(input_path_err, scenario_nr):
    with open(input_path_err, errors='replace', encoding='utf-8') as f:
        results_err = json.load(f)

    intent_str = 'SCENARIO: {}\n'.format(scenario_nr)

    total_intent_errs = 0
    err_types_dict = classify_err_types(results_err)
    err_types_count = dict()

    for err_type, errs in err_types_dict.items():
        intent_str += '------------{}------------\n'.format(err_type)
        for err in errs:
            intent_str += '{} {}->{}\n'.format(err['testData'], err['trueIntent'], err['predictedIntent'])
        err_types_count[err_type] = len(errs)
        total_intent_errs += err_types_count[err_type]
        intent_str += '{} TOTAL = {}\n\n'.format(err_type, err_types_count[err_type])

    intent_str += 'TOTAL INTENT ERRORS = {}\n\n\n'.format(total_intent_errs)

    for err_type, count in err_types_count.items():
        percentage = float(count) * 100 / total_intent_errs
        intent_str += '{} -- {:.2f}\n'.format(err_type, percentage)

    output_path = INTENT_ERRS_BASE_DIR + 'intent-errs-' + scenario_nr + '-' + SCENARIO_NAME + '.txt'
    with open(output_path, 'w') as f:
        f.write(intent_str)


def get_superclass_slot_errs_report(superclass_dict, superclass_dict_errs, total_slot_errs):
    slot_errs_str = ''

    light_errs_total = superclass_dict['lumina'] + superclass_dict_errs['lumina']
    temp_errs_total = superclass_dict['temperatura'] + superclass_dict_errs['temperatura']
    media_errs_total = superclass_dict['media'] + superclass_dict_errs['media']

    slot_errs_str += 'LIGHT class slot errors: {} ({:.2f})\n'.format(
        light_errs_total, float(light_errs_total) * 100 / total_slot_errs)
    slot_errs_str += 'TEMP class slot errors: {} ({:.2f})\n'.format(
        temp_errs_total, float(temp_errs_total) * 100 / total_slot_errs)
    slot_errs_str += 'MEDIA class slot errors: {} ({:.2f})\n'.format(
        media_errs_total, float(media_errs_total) * 100 / total_slot_errs)

    return slot_errs_str


def show_slot_errors(input_path_correct, input_path_err, scenario_nr):
    with open(input_path_correct, errors='replace', encoding='utf-8') as f:
        results_correct = json.load(f)

    slot_errs_str = 'SCENARIO: {}\n'.format(scenario_nr)

    total_slot_errs = 0
    slot_errs_str += '\nCORRECTLY PREDICTED INTENTS:\n'
    errs_nr, errs_str, superclass_dict = get_incorrect_slots(results_correct)
    total_slot_errs += errs_nr
    slot_errs_str += errs_str
    correct_intent_slot_errs = total_slot_errs
    slot_errs_str += '\nNR SLOT ERRORS (CORRECT INTENTS): {}\n'.format(correct_intent_slot_errs)

    with open(input_path_err, errors='replace', encoding='utf-8') as f:
        results_err = json.load(f)

    slot_errs_str += '\nINCORRECTLY PREDICTED INTENTS:\n'
    errs_nr, errs_str, superclass_dict_errs = get_incorrect_slots(results_err)
    total_slot_errs += errs_nr
    slot_errs_str += errs_str
    slot_errs_str += '\nNR SLOT ERRORS (INCORRECT INTENTS): {}\n'.format(total_slot_errs - correct_intent_slot_errs)

    slot_errs_str += '\nTotal slot errors: {}\n'.format(total_slot_errs)

    slot_errs_str += get_superclass_slot_errs_report(superclass_dict, superclass_dict_errs, total_slot_errs)

    output_path = SLOT_ERRS_BASE_DIR + 'slot-errs-' + scenario_nr + '-' + SCENARIO_NAME + '.txt'
    with open(output_path, 'w') as f:
        f.write(slot_errs_str)


def main():
    files = setup_input_files()

    for corr, err, sc_nr in files:
        plots_base_dir = corr.split('/')[:-1]
        plots_base_dir = '/'.join(plots_base_dir) + '/plots/{}-'
        print('------{}------'.format(corr))
        show_intent_errors(err, sc_nr)
        show_slot_errors(corr, err, sc_nr)
        # plot_conf_levels(corr, plots_base_dir + corr.split('/')[-1][:-5], sc_nr, FileContents.correct)
        # plot_conf_levels(err, plots_base_dir + err.split('/')[-1][:-5], sc_nr, FileContents.errors, plot_matrix=True)


if __name__ == '__main__':
    main()

