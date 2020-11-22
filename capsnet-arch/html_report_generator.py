import os

from dominate import document
from dominate.tags import *

def generateHtmlReport(FLAGS, y_intent_labels_true,
                       y_intent_labels_pred,
                       y_slot_labels_true,
                       y_slot_labels_pred,
                       x_text_te,
                       total_attention,
                       intent_confidence_tuples):
    if FLAGS.scenario_num != '':
        results_dir = FLAGS.results_dir + 'scenario' + FLAGS.scenario_num + '/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    else:
        results_dir = FLAGS.results_dir

    errorDataToPrint = []
    correctDataToPrint = []
    i, ccount, ecount = 0, 0, 0
    for t, pr, s, spr, confList  in zip(y_intent_labels_true, y_intent_labels_pred, y_slot_labels_true, y_slot_labels_pred,
                              intent_confidence_tuples):
        errorDataToPrint.append({})
        correctDataToPrint.append({})
        if t == pr:
            correctDataToPrint[ccount]["trueIntent"] = t
            correctDataToPrint[ccount]["confidenceList"] = confList
            correctDataToPrint[ccount]["predictedIntent"] = pr
            correctDataToPrint[ccount]["testData"] = ' '.join(x_text_te[i])
            correctDataToPrint[ccount]["trueSlots"] = str(s)
            correctDataToPrint[ccount]["predSlots"] = str(spr)
            correctDataToPrint[ccount]["attentionPerHeads"] = total_attention[i]
            ccount += 1
        else:
            errorDataToPrint[ecount]["trueIntent"] = t
            errorDataToPrint[ecount]["confidenceList"] = confList
            errorDataToPrint[ecount]["predictedIntent"] = pr
            errorDataToPrint[ecount]["testData"] = ' '.join(x_text_te[i])
            errorDataToPrint[ecount]["trueSlots"] = str(s)
            errorDataToPrint[ecount]["predSlots"] = str(spr)
            errorDataToPrint[ecount]["attentionPerHeads"] = total_attention[i]
            ecount += 1
        i += 1

    generateDocument(correctDataToPrint, ccount, results_dir, FLAGS.r, "correct-predictions-" + FLAGS.scenario_num)
    generateDocument(errorDataToPrint, ecount, results_dir, FLAGS.r, "error-predictions-" + FLAGS.scenario_num)



def generateDocument(dataToPrint, count, dir, attentionHeads, name):
    doc = document(title='Results - scenario - ' + name)
    with doc.head:
        meta(charset='utf-8')
        link(rel="stylesheet", href="../../result-styles.css")
    with doc:
        for j in range(count - 1):
            with div():
                attr(cls="example")
                p('Example nr: ' + str(j))
                p('true intent: ' + dataToPrint[j]["trueIntent"])
                p('pred intent: ' + dataToPrint[j]["predictedIntent"])
                p('test data:   ' + dataToPrint[j]["testData"])
                p('true slots:  ' + dataToPrint[j]["trueSlots"])
                p('pred slots:  ' + dataToPrint[j]["predSlots"])
                p('Intent confidence levels:')
                with div():
                    attr(cls="intents-head")
                    with p():
                        for intent, confidence in dataToPrint[j]["confidenceList"]:
                            span(intent, cls="intent")
                    with p():
                        for _, confidence in dataToPrint[j]["confidenceList"]:
                            span("%10.3f" % confidence, cls="intent", style="background-color: rgba(0,0,255, " + str(confidence) + ")")
                p('attention heads:')
                for x in range(attentionHeads):
                    with div():
                        attr(cls="attention-head")
                        with p():
                            for word, f in zip(dataToPrint[j]["testData"].split(' '),
                                               dataToPrint[j]["attentionPerHeads"][x]):
                                span(word, style="background-color: rgba(0,0,255, " + str(f) + ")", cls="word")
                        with p():
                            for f in dataToPrint[j]["attentionPerHeads"][x]:
                                span("%10.3f" % f, cls="word", style="background-color: rgba(0,0,255, " + str(f) + ")")

    with open(os.path.join(dir, 'results-' + name + '.html'), 'w', encoding='utf-8') as f:
        f.write(doc.render())