import transformers
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("Finally i got a chance to learn abt the hugging face")

print(res)