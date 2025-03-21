import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLM model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()

# Input prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Inference with KV cache
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, use_cache=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))