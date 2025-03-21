import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Inference without KV cache
start = time.time()
with torch.no_grad():
    model.generate(**inputs, max_length=50, use_cache=False)
end = time.time()
print(f"Inference without KV cache: {end - start:.4f} sec")

# Inference with KV cache
start = time.time()
with torch.no_grad():
    model.generate(**inputs, max_length=50, use_cache=True)
end = time.time()
print(f"Inference with KV cache: {end - start:.4f} sec")