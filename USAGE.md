# How to Use the Secure Chatbot

This guide shows you how to actually use the chatbot to have conversations.

## 🎯 Quick Start

### Option 1: Command-Line Interface (Recommended for First Time)

1. **Install dependencies** (if you haven't already):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the chatbot**:
   ```bash
   python3 chat.py
   ```

3. **Start chatting!**:
   ```
   You: Hello!
   Bot: Hello! I'm a secure AI chatbot. Your messages are protected...
   
   You: Tell me about privacy
   Bot: Yes! I use Secure Multiparty Computation (MPC)...
   ```

4. **Exit**: Type `quit` or `exit`, or press `Ctrl+C`

### Option 2: Web Interface (Beautiful UI)

1. **Start the web server**:
   ```bash
   python3 web_chat.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:8080
   ```

3. **Start chatting** in the beautiful web interface!

4. **Stop the server**: Press `Ctrl+C` in the terminal

## 💬 Example Conversations

### Basic Chat
```
You: Hello, how are you?
Bot: Hello! I'm a secure AI chatbot. Your messages are protected with secret sharing. How can I help you?

You: What can you do?
Bot: I'm here to help! I can answer questions while protecting your privacy. Try asking me about privacy, security, or anything else. Remember, your messages are encrypted and secret-shared!
```

### Privacy Questions
```
You: How does privacy work?
Bot: Yes! I use Secure Multiparty Computation (MPC) based on the FLUID protocol. Your inputs are secret-shared across multiple parties, so no single party can see your message. The model weights are also secret-shared for protection.
```

### General Questions
```
You: What is MPC?
Bot: I understand you're asking about 'What is MPC'. In a full secure implementation, I would process this using secret-shared computation across multiple parties, ensuring your privacy. Your message is protected!
```

## 🔒 What Happens Behind the Scenes

When you send a message:

1. **Tokenization**: Your message is converted to tokens
2. **Secret Sharing**: Tokens are split into random shares
3. **Secure Computation**: Shares are processed without revealing the original
4. **Response Generation**: A secure response is generated
5. **Privacy Preserved**: No single party sees your original message!

## 🎨 Features

- ✅ **Privacy-Preserving**: Your messages are secret-shared
- ✅ **Interactive**: Real-time conversation
- ✅ **Secure**: Uses cryptographic techniques
- ✅ **Easy to Use**: Simple command-line or web interface

## 🛠️ Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Port already in use (web interface)
Change the port in `web_chat.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Python version
Make sure you're using Python 3.7+:
```bash
python3 --version
```

## 📚 Next Steps

- Read [README.md](README.md) for full documentation
- Read [TUTORIAL.md](TUTORIAL.md) to understand how it works
- Read [DESIGN.md](DESIGN.md) for technical details
- Check [API.md](API.md) for programming interface

## 🎓 Learn More

The chatbot demonstrates:
- Secret sharing
- Secure multiparty computation
- Privacy-preserving AI
- Cryptographic protocols

Enjoy your secure conversations! 🔐
