class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, label_list, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_len = max_len
        self.label_map = {label: i for i, label in enumerate(label_list)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        # Tokenize the sentence
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # Convert tokenized words to label ids
        labels_ids = []
        for offset, label in zip(encoding.offset_mapping, labels):
            if offset[0] == 0 and offset[1] != 0:  # Only assign label to the first subword token
                labels_ids.append(self.label_map[label])
            else:
                labels_ids.append(-100)  # Ignore loss for other subword tokens

        # Remove offset mapping, we don't need it anymore
        encoding.pop("offset_mapping")

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels_ids)

        return item
