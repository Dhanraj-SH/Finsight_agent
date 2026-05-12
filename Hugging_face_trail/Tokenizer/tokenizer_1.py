
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("I will be working on a new FinBERT model project")
print(res)

# Whats happening with the tokenizer???

'''
A tokenizer basically puts text into mathematical representation that a model understands.
And inorder to use it we can call the `tokenizer` directly
'''
sequence = "Using Transformer network is simple"
res = tokenizer(sequence)
print(res)


# Or we can do it seperately
tokens = tokenizer.tokenize(sequence) # This would give us token specs
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens) # This would give us id's
print(ids)

decode_string = tokenizer.decode(ids)
print(decode_string)