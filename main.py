# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-7B-Instruct-v1")
model = AutoModelForCausalLM.from_pretrained("Equall/Saul-7B-Instruct-v1")

# Define the prompt
prompt = """
A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word.

Q: The mark "Ivory" for a product made of elephant tusks. What is the type of mark?
A: generic

Q: The mark "Tasty" for bread. What is the type of mark?
A: descriptive

Q: The mark "Caress" for body soap. What is the type of mark?
A: suggestive

Q: The mark "Virgin" for wireless communications. What is the type of mark?
A: arbitrary

Q: The mark "Aswelly" for a taxi service. What is the type of mark?
A: fanciful

Q: {{text}} What is the type of mark?
A:
"""

# Prepare the input for the model
input_ids = tokenizer(prompt.format(text = "The mark 'Aswelly' for a taxi service."), return_tensors="pt").input_ids

# Generate a response
output = model.generate(input_ids, max_length=150, num_return_sequences=1)

# Decode the generated output
result = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the result
print(result)
