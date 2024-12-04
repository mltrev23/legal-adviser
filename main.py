from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

# Determine the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-7B-Instruct-v1")
model = AutoModelForCausalLM.from_pretrained("Equall/Saul-7B-Instruct-v1").to(device)

# Ensure `pad_token_id` is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the dataset
dataset = datasets.load_dataset('nguha/legalbench', 'abercrombie')
print(dataset['train'].to_pandas())  # Display the training dataset as a DataFrame

# Define the prompt template
prompt_template = """
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

Q: {text} What is the type of mark?
A:
"""

# Replace this with your query
query_text = "The mark 'Aswelly' for a taxi service."

# Prepare the input for the model
prompt = prompt_template.format(text=query_text)
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Generate a response
output = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=1500,  # Adjust based on expected output length
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=1
)

# Decode and print the result
result = tokenizer.decode(output[0], skip_special_tokens=True)

response_start = len(prompt.strip())  # Locate where the response begins
response = result[response_start:].strip()  # Extract only the model's response

print(response)
