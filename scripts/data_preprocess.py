import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, labels_to_id):
        self.tokenizer = tokenizer
        self.labels_to_id = labels_to_id
        self.texts = []
        self.labels = []

        # Read data
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens, label_ids = [], []
            for line in f:
                line = line.strip()
                if line == "":
                    # end of an example; process and reset for the next example
                    if tokens:
                        self.texts.append(tokens)
                        self.labels.append(label_ids)
                        tokens, label_ids = [], []
                else:
                    token, label = line.split('\t')
                    tokens.append(token)
                    label_ids.append(self.labels_to_id[label])
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenization
        encoding = self.tokenizer(
            self.texts[idx],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )

        # Convert label IDs to tensor and handle padding
        labels = torch.LongTensor(self.labels[idx])
        labels_padded = torch.ones(128, dtype=torch.long) * -100  # Padding index for labels
        labels_padded[:len(labels)] = labels

        # Set up the dictionary to return
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = labels_padded
        return item

def create_dataset(file_path, tokenizer, label_list):
    # Map labels to IDs
    labels_to_id = {label: idx for idx, label in enumerate(label_list)}

    # Create the dataset
    dataset = NERDataset(file_path, tokenizer, labels_to_id)
    return dataset

def load_processed_data():
    train_loader = torch.load("data/processed/train_loader.pt")
    test_loader = torch.load("data/processed/test_loader.pt")
    return train_loader, test_loader

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    label_list = ['O', 'B-musicgenre', 'I-musicgenre', ...]  # Specify your label list here
    train_loader, test_loader = load_processed_data()