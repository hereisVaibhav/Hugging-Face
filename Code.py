import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
categories = ['positive', 'negative']
text = "I really enjoyed this movie. The acting was great and the plot kept me engaged throughout."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
predicted_category = categories[predictions.argmax()]
print(predicted_category)

# The output should be positive
