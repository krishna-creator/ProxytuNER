# Proxy TuNER: Advancing Cross-Domain Named Entity Recognition through Proxy Tuning

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
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

Traditional NER models often struggle with domain-specific language nuances, but Proxy TuNER addresses this challenge by fine-tuning BERT models using proxy tuning. This technique allows the model to adapt to new domains without modifying its original pre-trained weights, preserving its generalization abilities while enhancing domain-specific performance using the Hugging Face `transformers` library and PyTorch.

## Dataset

The dataset should be in a tab-separated format (TSV), where each line contains a word and its corresponding entity tag, separated by a tab (`\t`). An empty line indicates the end of a sentence. Here's an example:

```bash
    Barack B-person Obama I-person was O born O in O Hawaii B-location

```
