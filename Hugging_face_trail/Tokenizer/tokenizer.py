from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I will be working on a new FinBERT model project")

print(res)

#This is what's happening in the background the classifier automatically takes the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)
res = classifier("I will be working on a new FinBERT model project")
print(res)

'''
As we can see in the output which is same for both as the reference

output for the above code from line 4 to 8
[{'label': 'POSITIVE', 'score': 0.9600949883460999}]

output for the above code from line 11 to 14
[{'label': 'POSITIVE', 'score': 0.9600949883460999}]
'''