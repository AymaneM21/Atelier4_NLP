# Lab 4: NLP Language Models Using PyTorch

## Objective:
The main goal is to get familiar with NLP language models using the PyTorch library.

## Part 1: Classification and Regression

### Data Collection:
- Use Scrapy/BeautifulSoup to scrape Arabic text data from websites.
- Assign a relevance score (0-10) to each text.

### Data Preprocessing:
- Tokenization, stemming, lemmatization, stop word removal, and discretization.

### Model Training:
- Train models using RNN, Bidirectional RNN, GRU, and LSTM.
- Tune hyperparameters for optimal performance.

### Evaluation:
- Evaluate models using standard metrics (MSE, MAE).

### Results:
- RNN: MSE: 7.42, MAE: 2.20
- GRU: MSE: 7.69, MAE: 2.04
- LSTM: MSE: 6.78, MAE: 1.81
- Bidirectional RNN: MSE: 5.98, MAE: 1.55

## Part 2: Transformer (Text Generation)

### Model Setup:
- Install PyTorch transformers and load the GPT-2 pre-trained model.
- Fine-tune GPT-2 on a custom dataset.

### Text Generation:
- Generate a new paragraph based on a given sentence.

## Part 3: BERT

### Model Setup:
- Use the pre-trained BERT base-uncased model.

### Data Preparation:
- Prepare data and adapt the BERT embedding layer.

### Fine-Tuning and Training:
- Fine-tune and train the model with optimal hyperparameters.

### Evaluation:
- Evaluate using accuracy, F1 score, loss, BLEU score, and BERT-specific metrics.

### Results:
- Accuracy: 99.53%
- F1 Score: 0.997
- Loss: 0.021

## Conclusion
The lab provided a comprehensive understanding of NLP model training and evaluation using PyTorch. Each part focused on different aspects: classification and regression with traditional RNNs, text generation with transformers, and fine-tuning pre-trained BERT models. The hands-on approach enhanced familiarity with various NLP techniques and tools.

## Tools Used:
- Kaggle
- GitHub/GitLab
- Spacy, NLTK, PyTorch , Tensorflow, transformers, qalsadi, sklearn, numpy, pandas, Beautifulsoup
  
---

**Université Abdelmalek Essaadi** 
- Faculté des Sciences et Techniques de Tanger
- Département Génie Informatique
- Master : AISD
- NLP
- Pr . ELAACHAK LOTFI

