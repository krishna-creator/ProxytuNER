# Proxy TuNER: Advancing Cross-Domain Named Entity Recognition through Proxy Tuning

You can read the full research paper [here](https://figshare.com/articles/journal_contribution/Proxy_TuNER_Advancing_Cross-Domain_NamedEntity_Recognition_through_Proxy_Tuning/26822227/1).

Proxy TuNER is an innovative project focused on improving the versatility and accuracy of Named Entity Recognition (NER) systems across different domains by implementing advanced proxy tuning methods.

## Table of Contents

- [Named Entity Recognition (NER) using BERT](#named-entity-recognition-ner-using-bert)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#inference)
  - [Contribution](#contribution)
  - [License](#licence)

## Project Overview

Traditional NER models often struggle with domain-specific language nuances, but Proxy TuNER addresses this challenge by fine-tuning BERT models using proxy tuning. This technique allows the model to adapt to new domains without modifying its original pre-trained weights, preserving its generalization abilities while enhancing domain-specific performance using the Hugging Face `transformers` library and PyTorch.

## Dataset

The dataset should be in a tab-separated format (TSV), where each line contains a word and its corresponding entity tag, separated by a tab (`\t`). An empty line indicates the end of a sentence. Here's an example:

```bash
    Barack B-person Obama I-person was O born O in O Hawaii B-location
```

Ensure that your dataset is divided into training and testing files, typically named `train.txt` and `test.txt`, respectively, and located in the `ner_data/music/` directory or a directory you specify.

## Installation

To get started, clone this repository and install the required packages:

```bash
  git clone https://github.com/yourusername/ner-bert.git
  cd ner-bert
  pip install -r requirements.txt
```
Make sure you have Python 3.7 or above installed.

## Usage
Prepare your dataset in the required format and place it in a directory. Update the file paths in the script accordingly.
## Training

Run the training script to train the BERT model on your dataset:
```bash
python train.py --data_dir ner_data/music --model_name bert-base-cased --output_dir models/ --num_train_epochs 3 --batch_size 16
```
Replace the following arguments as needed:

--data_dir: Directory where your training data is located.
--model_name: Name of the BERT model to use. Default is bert-base-cased.
--output_dir: Directory where the model checkpoints and other outputs will be saved.
--num_train_epochs: Number of epochs for training. Default is 3.
--batch_size: Training batch size. Default is 16.

## Evaluation

After training, evaluate the model's performance on the test dataset:

```bash
python evaluate.py --data_dir ner_data/music --model_name models/ --batch_size 16
```

Replace the following arguments as needed:

--data_dir: Directory where your test data is located.
--model_name: Path to the trained model directory.
--batch_size: Evaluation batch size. Default is 16.
The script will output evaluation metrics such as accuracy, precision, recall, and F1-score.

## Inference

To use the trained model for inference on new text, you can run the inference script:

```bash
python inference.py --model_name models/ --input_text "Barack Obama was born in Hawaii."
```

Replace the following arguments as needed:

--model_name: Path to the trained model directory.
--input_text: Text input for which you want to predict named entities.
The script will output the input text with the identified named entities and their respective tags.

## Contribution

Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature-name).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature/your-feature-name).
Open a Pull Request.

## Licence 

This project is licensed under the Apache License - see the [LICENSE](http://www.apache.org/licenses/) file for details.



