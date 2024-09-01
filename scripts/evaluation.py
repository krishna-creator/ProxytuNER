import torch
from sklearn.metrics import accuracy_score, f1_score

def get_accuracy(adjusted_logits, file_path, tokenizer, label_list):
    probabilities = torch.softmax(adjusted_logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    true_labels = []
    dataset = create_dataset(file_path, tokenizer, label_list)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for batch in data_loader:
        batch_true_labels = batch['labels'].to(device)
        true_labels.append(batch_true_labels)
    true_labels = torch.cat(true_labels, dim=0)

    flattened_predictions = predictions.view(-1).cpu().numpy()
    flattened_true_labels = true_labels.view(-1).cpu().numpy()
    mask = (flattened_true_labels != -100)
    flattened_true_labels = flattened_true_labels[mask]
    flattened_predictions = flattened_predictions[mask]

    accuracy = accuracy_score(flattened_true_labels, flattened_predictions)
    f1 = f1_score(flattened_true_labels, flattened_predictions, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    return accuracy, f1

if __name__ == "__main__":
    # Assuming adjusted_logits is computed already
    accuracy, f1 = get_accuracy(adjusted_logits, test_file_path, tokenizer, label_list)
