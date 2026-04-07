import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def run_handshake(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", threshold=0.12):
    print(f"Initializing Handshake with {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

    prompt = "The capital of France is Paris. If x=2, then 5+x is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-5), dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
    print(f"\n[EIP-0.12] Handshake Audit Results:")
    for i, token in enumerate(tokens):
        h = entropy[0, i].item()
        status = "✅ EXIT" if h < threshold else "🧠 REASON"
        print(f"{token:<15} | H: {h:.4f} | {status}")

if __name__ == "__main__":
    run_handshake()
