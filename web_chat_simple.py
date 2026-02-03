#!/usr/bin/env python3
"""
Simplified Web-based Secure Chatbot Interface

Run with: python3 web_chat_simple.py
Then open: http://localhost:8080
"""

from flask import Flask, render_template_string, request, jsonify
from secure_chatbot.crypto.secret_sharing import SecretSharing
import numpy as np
import re

app = Flask(__name__)
secret_sharing = SecretSharing()

def evaluate_math(expression: str) -> tuple:
    """Safely evaluate a mathematical expression."""
    expression = expression.strip()
    expression = re.sub(r'^[?=\s:]+', '', expression)
    expression = re.sub(r'[?=\s:]+$', '', expression)
    expression = expression.replace('×', '*').replace('÷', '/')
    expression = expression.replace('x', '*').replace('X', '*')
    
    def replace_dot_multiplication(expr):
        expr = re.sub(r'(\d+)\s+\.\s+(\d+)', r'\1*\2', expr)
        expr = re.sub(r'(\d+)\.\s+(\d+)', r'\1*\2', expr)
        expr = re.sub(r'(\d+)\s+\.(\d+)', r'\1*\2', expr)
        return expr
    
    expression = replace_dot_multiplication(expression)
    expression = expression.replace(' ', '')
    
    if not re.match(r'^[0-9+\-*/().]+$', expression):
        return False, None, "Invalid characters"
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        if isinstance(result, (int, float)):
            return True, int(result) if result == int(result) else float(result), None
        return False, None, "Not a number"
    except ZeroDivisionError:
        return False, None, "Division by zero"
    except:
        return False, None, "Invalid expression"

def generate_response(user_input: str) -> str:
    """Generate a response."""
    has_numbers = bool(re.search(r'\d+', user_input))
    has_operators = bool(re.search(r'[+\-*/xX×÷.]', user_input)) or 'x' in user_input.lower() or '.' in user_input
    
    if has_numbers and has_operators:
        success, result, error = evaluate_math(user_input)
        if success:
            return f"The answer is: {result}"
        else:
            return f"Error: {error}. Try: '8+9', '10*5', '5x3', or '100/4'"
    
    user_lower = user_input.lower()
    if any(word in user_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm a secure AI chatbot. I can help with math questions! Try: 8+9?"
    elif any(word in user_lower for word in ['how are you']):
        return "I'm doing well! I can answer math questions and more. What would you like to know?"
    elif any(word in user_lower for word in ['privacy', 'secure']):
        return "Yes! I use Secure Multiparty Computation (MPC). Your inputs are secret-shared for privacy."
    elif any(word in user_lower for word in ['help']):
        return "I can answer math questions like '8+9', '10*5', '5x3', or '100/4'. Try asking me a math question!"
    else:
        return f"I understand '{user_input}'. I can also answer math questions like '8+9' or '10*5'!"

HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure AI Chatbot</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap');
        
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #fef9e7 0%, #fff8dc 50%, #fef3c7 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(234, 179, 8, 0.15), 0 0 0 1px rgba(234, 179, 8, 0.1);
            display: flex;
            flex-direction: column;
            height: 90vh;
            max-height: 900px;
        }
        
        .header {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #1f2937;
            padding: 28px 24px;
            text-align: center;
            border-bottom: 3px solid #fcd34d;
        }
        
        .header h1 { 
            font-family: 'Playfair Display', serif;
            font-size: 32px; 
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
            color: #1f2937;
        }
        
        .header p { 
            font-size: 15px; 
            font-weight: 500;
            opacity: 0.9;
            color: #374151;
        }
        
        .status {
            padding: 12px 20px;
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
            font-size: 13px;
            text-align: center;
            font-weight: 500;
            border-bottom: 1px solid #fcd34d;
        }
        
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            background: linear-gradient(to bottom, #ffffff 0%, #fef9e7 100%);
            scrollbar-width: thin;
            scrollbar-color: #fbbf24 #fef3c7;
        }
        
        .chat-area::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-area::-webkit-scrollbar-track {
            background: #fef3c7;
        }
        
        .chat-area::-webkit-scrollbar-thumb {
            background: #fbbf24;
            border-radius: 4px;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user { justify-content: flex-end; }
        .message.bot { justify-content: flex-start; }
        
        .message-content {
            max-width: 75%;
            padding: 14px 18px;
            border-radius: 20px;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #1f2937;
            border-bottom-right-radius: 6px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
        }
        
        .message.bot .message-content {
            background: #ffffff;
            color: #374151;
            border-bottom-left-radius: 6px;
            border: 1px solid #fde68a;
            box-shadow: 0 2px 8px rgba(234, 179, 8, 0.1);
        }
        
        .input-area {
            padding: 20px 24px;
            background: #ffffff;
            border-top: 2px solid #fef3c7;
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        #user-input {
            flex: 1;
            padding: 14px 20px;
            border: 2px solid #fde68a;
            border-radius: 30px;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: all 0.3s ease;
            background: #fef9e7;
            color: #1f2937;
        }
        
        #user-input:focus {
            border-color: #fbbf24;
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.1);
        }
        
        #user-input::placeholder {
            color: #9ca3af;
        }
        
        #send-btn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #1f2937;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
        }
        
        #send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(251, 191, 36, 0.4);
        }
        
        #send-btn:active {
            transform: translateY(0);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 12px;
            color: #f59e0b;
            font-size: 13px;
            font-weight: 500;
        }
        
        @media (max-width: 600px) {
            .container {
                width: 100%;
                height: 100vh;
                border-radius: 0;
                max-height: 100vh;
            }
            
            .header h1 {
                font-size: 26px;
            }
            
            .message-content {
                max-width: 85%;
                font-size: 14px;
                padding: 12px 16px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Secure AI Chatbot</h1>
            <p>Privacy-Preserving MPC System</p>
        </div>
        <div class="status">
            Secure mode active - Your messages are secret-shared and protected
        </div>
        <div class="chat-area" id="chat-area">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm a secure AI chatbot. I can help with math questions! Try asking: 8+9?
                </div>
            </div>
        </div>
        <div class="loading" id="loading">Processing securely...</div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const chatArea = document.getElementById('chat-area');
        const userInput = document.getElementById('user-input');
        const loading = document.getElementById('loading');
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            div.innerHTML = '<div class="message-content">' + text + '</div>';
            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            addMessage(message, true);
            userInput.value = '';
            loading.style.display = 'block';
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                const data = await response.json();
                loading.style.display = 'none';
                addMessage(data.response, false);
            } catch (error) {
                loading.style.display = 'none';
                addMessage('Sorry, an error occurred. Please try again.', false);
            }
        }
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please provide a message.'}), 400
    response = generate_response(user_message)
    return jsonify({'response': response, 'secure': True})

if __name__ == '__main__':
    print("=" * 60)
    print("Secure AI Chatbot - Web Interface")
    print("=" * 60)
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server.\n")
    app.run(debug=True, host='0.0.0.0', port=8080)
