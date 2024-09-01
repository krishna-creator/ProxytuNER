import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizer
from data_preprocess import create_dataset

class SmallExpertModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(SmallExpertModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits

def train_small_model(file_path, tokenizer, label_list, epochs=50, learning_rate=1e-3):
    dataset = create_dataset(file_path, tokenizer, label_list)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    embedding_dim = 768
    hidden_dim = 256
    output_dim = len(label_list)
    vocab_size = tokenizer.vocab_size
    expert_model = SmallExpertModel(embedding_dim, hidden_dim, output_dim, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(expert_model.parameters(), lr=learning_rate)

    expert_model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = expert_model(input_ids)
            logits = outputs.view(-1, output_dim)
            labels = labels.view(-1)
            loss = criterion(logits.to(device), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    anti_expert_model = SmallExpertModel(embedding_dim, hidden_dim, output_dim, vocab_size).to(device)
    return expert_model, anti_expert_model

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_file_path = "./ner_data/music/test.txt"
    label_list = ['O', 'B-musicgenre', 'I-musicgenre', ...]  # Specify your label list here
    expert_model, anti_expert_model = train_small_model(train_file_path, tokenizer, label_list)
