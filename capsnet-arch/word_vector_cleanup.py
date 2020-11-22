

def readData():
    baseDir = '../data-capsnets/scenario'
    baseDirDia = '../data-capsnets/diacritics/scenario'
    test = '/test.txt'
    train = '/train.txt'
    filePaths = [
        baseDir + '0' + test, baseDir + '0' + train,
        baseDir + '1' + test, baseDir + '1' + train,
        baseDir + '2' + test, baseDir + '2' + train,
        baseDir + '3.1' + test, baseDir + '3.1' + train,
        baseDir + '3.2' + test, baseDir + '3.2' + train,
        baseDir + '3.3' + test, baseDir + '3.3' + train,
        baseDirDia + '0' + test, baseDirDia + '1' + train,
        baseDirDia + '1' + test, baseDirDia + '1' + train,
        baseDirDia + '2' + test, baseDirDia + '2' + train,
        baseDirDia + '31' + test, baseDirDia + '31' + train,
        baseDirDia + '32' + test, baseDirDia + '32' + train,
        baseDirDia + '33' + test, baseDirDia + '33' + train,
    ]

    wordsSet = set()

    for filePath in filePaths:
        for line in open(filePath, encoding='utf-8'):
            arr = line.strip().split('\t')
            text = arr[2].split(' ')
            for word in text:
                wordsSet.add(word)
                if word[0].isupper():
                    wordsSet.add(word.lower())

    word_vector_path = "../../romanian_word_vecs/cc.ro.50.vec"

    word_vector_lines = {}
    firstLine = True
    for line in open(word_vector_path, encoding='utf-8'):
        if firstLine:
            firstLine = False
            continue
        lineSplit = line.split(" ", 1)
        word_vector_lines[lineSplit[0]] = lineSplit[1]

    final_vectors = []
    i = 0
    for word in wordsSet:
        if word in word_vector_lines.keys():
            final_vectors.append(word + ' ' + word_vector_lines[word])
        else:
            i += 1
            print("Word not in model: " + str(i) + ' ' + word)

    with open("../../romanian_word_vecs/cleaned-vectors-diacritice-cc-50.vec", "w", encoding="utf-8") as output:
        output.write(str(len(final_vectors)) + ' 50\n')
        for line in final_vectors:
            output.write(line)

    print("done")

readData()