import torch
from transformers import BertForTokenClassification
from train import SmallExpertModel
from data_preprocess import create_dataset
from data_preprocess import DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from transformers import BertForTokenClassification
from sklearn.metrics import f1_score

def adjust_bert_predictions(bert_logits, expert_logits, anti_expert_logits):
    expertise_difference = expert_logits - anti_expert_logits
    adjusted_logits = bert_logits + expertise_difference
    return adjusted_logits

def make_adjusted_predictions_dataset(file_path, tokenizer, bert_model, expert_model, anti_expert_model, label_list):
    dataset = create_dataset(file_path, tokenizer, label_list)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    bert_model.eval()
    expert_model.eval()
    anti_expert_model.eval()

    all_adjusted_logits = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            bert_outputs = bert_model(input_ids).logits
            expert_outputs = expert_model(input_ids)
            anti_expert_outputs = anti_expert_model(input_ids)
            adjusted_logits = adjust_bert_predictions(bert_outputs, expert_outputs, anti_expert_outputs)
            all_adjusted_logits.append(adjusted_logits)

    all_adjusted_logits = torch.cat(all_adjusted_logits, dim=0)
    return all_adjusted_logits

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list)).to(device)
    expert_model, anti_expert_model = train_small_model(train_file_path, tokenizer, label_list)
    adjusted_logits = make_adjusted_predictions_dataset(test_file_path, tokenizer, bert_model, expert_model, anti_expert_model, label_list)
