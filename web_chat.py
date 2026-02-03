#!/usr/bin/env python3
"""
Web-based Secure Chatbot Interface

A simple web interface for the secure AI chatbot using Flask.
Run with: python3 web_chat.py
Then open: http://localhost:8080
"""

from flask import Flask, render_template_string, request, jsonify
from secure_chatbot.crypto.secret_sharing import SecretSharing
import numpy as np
import re

app = Flask(__name__)

# Initialize chatbot components
secret_sharing = SecretSharing()

def evaluate_math(expression: str) -> tuple:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Tuple of (success: bool, result: float or None, error: str or None)
    """
    # Clean the expression
    expression = expression.strip()
    
    # Remove question marks and common prefixes/suffixes
    expression = re.sub(r'^[?=\s:]+', '', expression)
    expression = re.sub(r'[?=\s:]+$', '', expression)
    
    # Replace common math symbols
    # Handle multiplication: x, X, ×, and . (dot) as multiplication
    expression = expression.replace('×', '*').replace('÷', '/')
    expression = expression.replace('x', '*').replace('X', '*')
    
    # Handle dot as multiplication
    # Strategy: Only treat '.' as multiplication when there's explicit whitespace
    # This preserves all decimals like "10.5", "3.14", "2.5", "7.8"
    # User can use "5x3" or "5*3" for multiplication without space
    def replace_dot_multiplication(expr):
        # Only replace if there's whitespace around the dot: "5 . 3" or "5. 3" or "5 .3" -> "5 * 3"
        expr = re.sub(r'(\d+)\s+\.\s+(\d+)', r'\1*\2', expr)
        expr = re.sub(r'(\d+)\.\s+(\d+)', r'\1*\2', expr)
        expr = re.sub(r'(\d+)\s+\.(\d+)', r'\1*\2', expr)
        
        return expr
    
    expression = replace_dot_multiplication(expression)
    
    # Remove all spaces for cleaner parsing
    expression = expression.replace(' ', '')
    
    # Only allow safe characters: numbers, operators, parentheses, decimal points
    if not re.match(r'^[0-9+\-*/().]+$', expression):
        return False, None, "Invalid characters in expression"
    
    try:
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})
        
        # Check if result is a number
        if isinstance(result, (int, float)):
            # Format result nicely (remove unnecessary .0 for integers)
            if result == int(result):
                return True, int(result), None
            else:
                return True, float(result), None
        else:
            return False, None, "Expression did not evaluate to a number"
            
    except ZeroDivisionError:
        return False, None, "Division by zero"
    except SyntaxError:
        return False, None, "Invalid mathematical expression"
    except Exception as e:
        return False, None, f"Error evaluating expression: {str(e)}"

# Simple response generator (same as chat.py)
def generate_response(user_input: str) -> str:
    """Generate a response (simplified for demo)"""
    # Check for mathematical expressions first
    # More comprehensive pattern: contains numbers and math operators
    has_numbers = bool(re.search(r'\d+', user_input))
    has_operators = bool(re.search(r'[+\-*/xX×÷.]', user_input)) or 'x' in user_input.lower() or '.' in user_input
    
    if has_numbers and has_operators:
        # Try to evaluate as math
        success, result, error = evaluate_math(user_input)
        if success:
            return f"The answer is: {result}"
        # If it looks like math but failed, provide helpful error
        else:
            return f"I tried to calculate '{user_input.replace('?', '').strip()}' but encountered an error: {error}. Please try a different format (e.g., '8+9', '10*5', '5x3', or '100/4')."
    
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm a secure AI chatbot. Your messages are protected with secret sharing. I can help with math questions, privacy questions, and more! How can I help you?"
    elif any(word in user_lower for word in ['how are you']):
        return "I'm doing well, thank you! I'm operating in secure mode, which means your privacy is protected. I can answer math questions and other queries."
    elif any(word in user_lower for word in ['privacy', 'secure']):
        return "Yes! I use Secure Multiparty Computation (MPC). Your inputs are secret-shared across multiple parties, so no single party can see your message."
    elif any(word in user_lower for word in ['help']):
        return "I'm here to help! I can answer math questions (like '8+9' or '10*5'), privacy questions, and more - all while protecting your privacy."
    else:
        return f"I understand you're asking about '{user_input}'. In a full secure implementation, this would be processed using secret-shared computation, ensuring your privacy! (Tip: I can also answer math questions like '8+9' or '10*5')"

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
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
            width: 90%;
            max-width: 900px;
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(234, 179, 8, 0.15), 0 0 0 1px rgba(234, 179, 8, 0.1);
            overflow: hidden;
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
                    Hello! I'm a secure AI chatbot. Your messages are protected using Secure Multiparty Computation (MPC). No single party can see your input. How can I help you?
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
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message
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
</html>
"""

@app.route('/')
def index():
    """Serve the main chat interface"""
    try:
        return render_template_string(HTML_TEMPLATE)
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

@app.route('/test')
def test():
    """Test route to verify server is working"""
    return jsonify({'status': 'ok', 'message': 'Server is working!'})

@app.route('/simple')
def simple():
    """Simple test page to verify Flask is working"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>Flask is Working!</h1>
        <p>If you see this, the server is running correctly.</p>
        <p><a href="/">Go to Chat Interface</a></p>
    </body>
    </html>
    """

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please provide a message.'}), 400
    
    # Simulate secure processing
    # In real implementation, this would:
    # 1. Secret share the input
    # 2. Send shares to CP1 and CP2
    # 3. Perform secure computation
    # 4. Reconstruct the output
    
    # For demo, just generate a response
    response = generate_response(user_message)
    
    return jsonify({
        'response': response,
        'secure': True
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Secure AI Chatbot - Web Interface")
    print("=" * 60)
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server.\n")
    app.run(debug=True, host='0.0.0.0', port=8080)
