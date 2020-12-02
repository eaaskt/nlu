import json
import os


def generate_conf_reports(FLAGS, y_intent_labels_true,
                       y_intent_labels_pred,
                       y_slot_labels_true,
                       y_slot_labels_pred,
                       x_text_te,
                       intent_confidence_tuples):

    if FLAGS.scenario_num != '':
        confidence_dir = FLAGS.confidence_dir + 'scenario' + FLAGS.scenario_num + '/'
        if not os.path.exists(confidence_dir):
            os.makedirs(confidence_dir)
    else:
        confidence_dir = FLAGS.confidence_dir

    correct_data = []
    error_data = []

    for y_t, y_pr, y_st_t, y_st_pr, conf_list, txt \
            in zip(y_intent_labels_true, y_intent_labels_pred, y_slot_labels_true,
                   y_slot_labels_pred, intent_confidence_tuples, x_text_te):
        ex_dict = dict()
        ex_dict["trueIntent"] = y_t
        ex_dict["predictedIntent"] = y_pr
        ex_dict["testData"] = ' '.join(txt)
        ex_dict["trueSlots"] = str(y_st_t)
        ex_dict["predSlots"] = str(y_st_pr)
        ex_dict["confidenceList"] = conf_list_to_dict(conf_list)
        if y_t == y_pr:
            correct_data.append(ex_dict)
        else:
            error_data.append(ex_dict)

    with open(os.path.join(confidence_dir, 'conf-correct-predictions-' + FLAGS.scenario_num + '.json'),
              'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(confidence_dir, 'conf-error-predictions-' + FLAGS.scenario_num + '.json'),
              'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=4)


def conf_list_to_dict(conf_list):
    conf_dict = dict()
    for intent, conf in conf_list:
        conf_dict[intent] = conf
    return conf_dict
