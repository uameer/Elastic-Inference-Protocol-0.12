import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_layer_entropy(hidden_state, model):
    """Project hidden state through LM head and compute entropy."""
    normed = model.model.norm(hidden_state)
    logits = model.lm_head(normed)
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log2(probs + 1e-5), dim=-1)


def run_eip_audit(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="The capital of France is Paris. If x=2, then 5+x is",
    probe_layers=[4, 8, 12, 16, 20, 22],
    threshold=0.12
):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # --- Table ---
    print(f"\n[EIP] Layer-wise Entropy Audit | Threshold: {threshold} bits")
    print(f"Prompt: {prompt}")
    print(f"Model layers: {len(hidden_states) - 1}\n")

    header = f"{'Token':<15}" + "".join(f"| L{l:<5}" for l in probe_layers) + "| Decision"
    print(header)
    print("-" * len(header))

    for i, token in enumerate(tokens):
        row = f"{token.replace('▁', '_'):<15}"
        settled = False
        settle_layer = None

        for l in probe_layers:
            h = hidden_states[l][:, i:i+1, :]
            e = compute_layer_entropy(h, model)[0, 0].item()
            row += f"| {e:<6.3f}"
            if e < threshold and not settled:
                settled = True
                settle_layer = l

        decision = f"✅ EXIT @ L{settle_layer}" if settled else "🧠 FULL PASS"
        row += f"| {decision}"
        print(row)

    # --- Plot ---
    print("\nGenerating entropy trajectory plot...")
    layers = list(range(1, len(hidden_states)))
    plt.figure(figsize=(14, 6))

    for i, token in enumerate(tokens):
        traj = []
        for l in layers:
            h = hidden_states[l][:, i:i+1, :]
            e = compute_layer_entropy(h, model)[0, 0].item()
            traj.append(e)

        final_e = traj[-1]
        color = plt.cm.RdYlBu_r(min(final_e / 12.0, 1.0))
        plt.plot(layers, traj, color=color, alpha=0.7, linewidth=1.5)
        plt.annotate(token.replace('▁', '_'), (layers[-1], traj[-1]), fontsize=7)

    plt.axhline(y=0.12, color='green', linestyle='--', linewidth=2, label='0.12 Gate (EIP)')
    plt.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, label='2.0 Soft Gate')
    plt.xlabel("Layer")
    plt.ylabel("Entropy (bits)")
    plt.title(f"Token Entropy Trajectories — {model_id.split('/')[1]}\n'{prompt}'")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eip_entropy_audit.png", dpi=150)
    print("[EIP] Chart saved to eip_entropy_audit.png")
    plt.show()


if __name__ == "__main__":
    run_eip_audit()
