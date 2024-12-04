from llama_cpp import Llama

# # Ensure that the code uses only GPU by setting appropriate parameters
# llm = Llama(
#     model_path="./Saul-Base-GGUF.Q4_K_M.gguf",  # Path to the model file
#     n_ctx=32768,  # Max sequence length
#     n_gpu_layers=-1
# )

# system_message = "Translate into Germany!"
# prompt = "Hello, How are you doing today?"

# # Simple inference example with GPU only
# output = llm(
#     f"""<|im_start|>system
# {system_message}<|im_end|>
# <|im_start|>user
# {prompt}<|im_end|>
# <|im_start|>assistant""",  # Input prompt
#     max_tokens=512,  # Max tokens to generate
#     stop=["</s>"],  # Stop token (ensure this matches your model's token)
#     echo=True  # Echo the prompt in the response
# )

# print(output)

# Chat Completion API (using GPU exclusively)
llm = Llama(
    model_path="./saul-base_q2_k.gguf", 
    chat_format="llama-2",  # Use correct chat format based on the model
    n_gpu_layers=-1
)

output = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a story writing assistant."},
        {"role": "user", "content": "Write a story about llamas."}
    ]
)

print(output)
