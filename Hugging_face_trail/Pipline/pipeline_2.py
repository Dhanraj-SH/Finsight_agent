from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehesion",
    candidate_labels = ["education", "politics", "bussiness", "programming"],
)

print(res)