# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

input_text = "List out all the different primitive actions that you think could be collected via this robot manipulating on household objects or parts of furniture( not heavier than 3kg, and the maximum width of the object not exceeding 80mm):"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

outputs = model.generate(input_ids, max_length=1000)
print(tokenizer.decode(outputs[0]))
