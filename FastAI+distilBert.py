from fastai.text.all import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch.nn as nn

# Load data and clean
def clean_data(df):
    df['Phrase'] = df['Phrase'].fillna('').astype(str)  # Convert to string and fill NaN
    return df

df = clean_data(pd.read_csv('train.tsv', sep='\t')).sample(n=50000, random_state=42)
df_test = clean_data(pd.read_csv('test.tsv', sep='\t'))

# Prepare data
dls = TextDataLoaders.from_df(df, 
                              text_col='Phrase', 
                              label_col='Sentiment', 
                              valid_pct=0.1,
                              batch_size=32,
                              max_len=128,
                              text_vocab=None,  # Let fastai create the vocabulary
                              num_workers=0)  # Set to 0 if you encounter multiprocessing issues

# Load pre-trained model and tokenizer
pretrained_model_name = 'distilbert-base-uncased'
distil_bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

# Create custom model
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer = transformer_model
        
    def forward(self, input):
        attention_mask = (input != tokenizer.pad_token_id).long()
        return self.transformer(input_ids=input, attention_mask=attention_mask)[0]

# Prepare learner
model = CustomTransformerModel(distil_bert)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.to_fp16()

# Train the model
learn.fit_one_cycle(3, 2e-5)

# Make predictions on test set
def get_preds(learn, test_df):
    dl = learn.dls.test_dl(test_df['Phrase'])
    preds, _ = learn.get_preds(dl=dl)
    return preds

preds = get_preds(learn, df_test)

# Prepare submission
submission = pd.DataFrame({'PhraseId': df_test['PhraseId'], 'Sentiment': preds.argmax(dim=1)})
submission.to_csv('submission.csv', index=False)
