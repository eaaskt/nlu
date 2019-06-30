import math
import random


ERRORS_3_1 = 'errors/scenario31/errors.txt'
ERRORS_3_2 = 'errors/scenario31/errors.txt'
ERRORS_3_3 = 'errors/scenario31/errors.txt'
ERROR_FILES = [ERRORS_3_1, ERRORS_3_2, ERRORS_3_3]


def sample(data, percentage=0.1):
    nr_total_errors = len(data)
    nr_sample = math.ceil(percentage * nr_total_errors)
    sample = random.sample(data, k=nr_sample)
    return sample


def print_intent_errors(err):
    for e in err:
        print('Text: ' + str(e[0]))
        print('TRUE: ' + str(e[1]))
        print()


def print_slot_errors(err):
    for e in err:
        print('Text: ' + str(e[0]))
        print('TRUE: ' + str(e[1]))
        print('PRED: ' + str(e[2]))
        print()


if __name__ == '__main__':
    intent_errors = []
    slot_errors = []
    for f1 in ERROR_FILES:
        read_intents = True
        read_slots = False
        with open(f1, 'r') as f:
            line = f.readline()
            while line:
                if line != '\n':
                    if line == 'INTENT ERRORS\n':
                        read_intents = True
                        read_slots = False
                    if line == 'SLOT ERRORS\n':
                        read_intents = False
                        read_slots = True
                    if len(line.split(' ')) == 1:
                        # True intent name
                        intent = line
                    elif len(line.split(' ')) > 2:
                        # Error example
                        if read_intents:
                            # print(intent)
                            # print(line)
                            intent_errors.append((intent, line))
                        if read_slots and line[0] != '[':
                            # print(line)
                            slots_true = f.readline()
                            # print(line)
                            slots_pred = f.readline()
                            # print(line)
                            slot_errors.append((line, slots_true, slots_pred))
                line = f.readline()

    print(intent_errors)
    print(slot_errors)

    print('Total # of errors (intent):' + str(len(intent_errors)))
    print('Total # of errors (slot):' + str(len(slot_errors)))
    intent_sample = sample(intent_errors)
    slot_sample = sample(slot_errors)

    print('INTENT ERRORS')
    print_intent_errors(intent_sample)
    print()
    print('SLOT ERRORS')
    print_slot_errors(slot_sample)
