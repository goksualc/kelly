# 🔐 Secure AI Chatbot - Privacy-Preserving MPC System

A complete implementation of a privacy-preserving AI chatbot using Secure Multiparty Computation (MPC) based on the FLUID protocol from the 2018 Nature Biotechnology paper "Secure genome-wide association analysis using multiparty computation" by Cho, Wu, and Berger.

## 🚀 Quick Start - Use the Chatbot Now!

**Want to chat right away?** Choose one:

1. **Command Line (CLI)**: 
   ```bash
   python3 chat.py
   ```
   Then type your messages and press Enter!

2. **Web Interface**: 
   ```bash
   python3 web_chat.py
   ```
   Then open `http://localhost:8080` in your browser!

That's it! Your messages are protected using secret sharing. 🔒

## Overview

This system enables AI inference (specifically transformer-based language models) while keeping:
- ✅ **User inputs private** - No single party sees the query
- ✅ **Model weights private** - Split across multiple parties
- ✅ **Intermediate computations private** - All activations remain secret-shared

## Architecture

The system consists of 4 parties:

1. **CP0 (Preprocessing Server)**: Generates Beaver triples offline (no input dependency)
2. **CP1 & CP2 (Computing Parties)**: Hold shares of model weights and perform secure computation
3. **Gateway**: Accepts user inputs, splits into shares, reconstructs final outputs

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 🚀 Use the Chatbot (Easiest Way)

**Option 1: Command-Line Interface (CLI)**
```bash
python3 chat.py
```
This starts an interactive chat session where you can type messages and get responses. Your messages are protected using secret sharing!

**Option 2: Web Interface**
```bash
python3 web_chat.py
```
Then open your browser and go to: `http://localhost:8080`

You'll see a beautiful web interface where you can chat with the secure AI chatbot.

### Run Examples

```bash
# Run example usage
python3 example_usage.py

# Run tests
pytest tests/
```

### Start Full System Components (Advanced)

For the complete distributed MPC system:

```bash
# Terminal 1: Start preprocessing server (CP0)
python3 main.py --mode cp0

# Terminal 2: Start computing party 1
python3 main.py --mode cp1 --port 8001

# Terminal 3: Start computing party 2
python3 main.py --mode cp2 --port 8002

# Terminal 4: Start gateway
python3 main.py --mode gateway --port 8000
```

## Features

### Cryptographic Primitives
- ✅ Additive secret sharing over finite field Zq
- ✅ Fixed-point arithmetic with encoding/decoding
- ✅ Beaver multiplication triples
- ✅ Secure matrix multiplication (single round, linear communication)

### Secure Neural Network Operations
- ✅ Secure linear layers
- ✅ Secure activation functions (ReLU, GELU, Softmax)
- ✅ Secure multi-head self-attention
- ✅ Secure layer normalization
- ✅ Secure feedforward networks

### Complete Transformer
- ✅ Multi-layer transformer architecture
- ✅ Secure embedding layers
- ✅ Secure token sampling (argmax, top-k)

## Security Guarantees

### Information-Theoretic Privacy
- Each share is uniformly random → reveals 0 bits about secret
- View of CP1/CP2 consists only of:
  - Their own shares (uniform random)
  - Blinded values x-r where r is uniform random

### Threat Model
- Semi-honest adversaries (honest-but-curious)
- Can corrupt CP1 OR CP2 (not both)
- Can corrupt CP0 in offline phase (but no online collusion)

### What is Protected
- ✅ User input: Information-theoretic
- ✅ Model weights: Information-theoretic
- ✅ Intermediate activations: Information-theoretic
- ✅ Attention patterns: Information-theoretic

## Performance

### Communication Complexity (per token)
- Input sharing: O(vocab_size) or O(hidden_dim)
- Per layer: O(seq_len · hidden_dim)
- Total: O(num_layers · seq_len · hidden_dim)

### Round Complexity
- Per layer: 1 round (Beaver partition reconstruction)
- Total: num_layers rounds

### Estimated Performance (GPT-2 Small, 512 tokens)
- Per token generation: ~5 seconds
- 50 token response: ~4 minutes
- Communication per token: ~275 MB

## Project Structure

```
secure_chatbot/
├── crypto/              # Cryptographic primitives
│   ├── secret_sharing.py
│   ├── fixed_point.py
│   ├── beaver_triples.py
│   └── secure_primitives.py
├── nn/                  # Secure neural network operations
│   ├── linear.py
│   ├── attention.py
│   ├── activations.py
│   ├── layernorm.py
│   └── transformer.py
├── parties/             # MPC parties
│   ├── preprocessing_server.py
│   ├── computing_party.py
│   └── gateway.py
├── protocols/           # Secure computation protocols
│   ├── matmul.py
│   └── sampling.py
└── utils/               # Utilities
    ├── prg.py
    ├── communication.py
    └── serialization.py
```

## Example Usage

### Quick Chat (CLI)

```bash
python3 chat.py
```

Then type your messages:
```
You: Hello, how are you?
Bot: Hello! I'm a secure AI chatbot. Your messages are protected with secret sharing. How can I help you?

You: Tell me about privacy
Bot: Yes! I use Secure Multiparty Computation (MPC) based on the FLUID protocol. Your inputs are secret-shared across multiple parties, so no single party can see your message. The model weights are also secret-shared for protection.
```

### Web Interface

```bash
python3 web_chat.py
# Open http://localhost:8080 in your browser
```

### Programmatic Usage

```python
from secure_chatbot.parties.gateway import Gateway

gateway = Gateway()
response = gateway.chat("Hello, how are you?", tokenizer)
print(response)
```

### Medical Chatbot Example

```python
# In a full implementation:
user = "I'm 55, diabetic, HbA1c 8.5. Constant fatigue. Help?"
response = chatbot.chat(user)
# Neither CP1 nor CP2 sees the medical query!
```

### Financial Advisory Example

```python
user = "Income: $500K, Savings: $2M. 10-year college fund?"
response = chatbot.chat(user)
# Financial details remain private!
```

## Documentation

- [DESIGN.md](DESIGN.md) - Detailed protocol description and security analysis
- [API.md](API.md) - Complete API reference
- [TUTORIAL.md](TUTORIAL.md) - Step-by-step tutorial

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_crypto.py
pytest tests/test_nn.py
```

## License

MIT License

## Citation

If you use this code, please cite the FLUID paper:

```
Cho, H., Wu, D. J., & Berger, B. (2018). 
Secure genome-wide association analysis using multiparty computation. 
Nature Biotechnology, 36(6), 547-551.
```

## Contributing

Contributions welcome! Please open an issue or pull request.

## Acknowledgments

Based on the FLUID protocol from:
- Cho, H., Wu, D. J., & Berger, B. (2018). Secure genome-wide association analysis using multiparty computation. Nature Biotechnology, 36(6), 547-551.
# kelly
