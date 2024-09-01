def read_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        label = []
        for line in file:
            line = line.strip()
            if not line:  # End of a sentence
                if sentence:  # Avoid empty sentences
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                word, tag = line.split('\t')
                sentence.append(word)
                label.append(tag)
        if sentence:  # Add the last sentence if file doesn't end with a newline
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels
