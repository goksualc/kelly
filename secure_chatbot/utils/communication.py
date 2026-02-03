"""
Communication Protocols for MPC Parties

Handles network communication between:
- Gateway <-> CP1
- Gateway <-> CP2
- CP1 <-> CP2
"""

import socket
import pickle
import json
from typing import Any, Dict, Optional
import threading
import queue


class MPCChannel:
    """Bidirectional communication channel between two parties"""
    
    def __init__(self, host: str, port: int, is_server: bool = False):
        """
        Initialize communication channel.
        
        Args:
            host: Host address
            port: Port number
            is_server: Whether this party is the server
        """
        self.host = host
        self.port = port
        self.is_server = is_server
        self.socket = None
        self.conn = None
        self.receive_queue = queue.Queue()
        self.receive_thread = None
        self.running = False
    
    def connect(self):
        """Establish connection"""
        if self.is_server:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.conn, addr = self.socket.accept()
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.conn = self.socket
        
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
    
    def _receive_loop(self):
        """Background thread for receiving messages"""
        while self.running:
            try:
                # Receive message length
                length_bytes = self.conn.recv(4)
                if not length_bytes:
                    break
                length = int.from_bytes(length_bytes, 'big')
                
                # Receive message
                data = b''
                while len(data) < length:
                    chunk = self.conn.recv(length - len(data))
                    if not chunk:
                        break
                    data += chunk
                
                # Deserialize
                message = pickle.loads(data)
                self.receive_queue.put(message)
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break
    
    def send(self, message: Any):
        """
        Send message.
        
        Args:
            message: Message to send (will be pickled)
        """
        data = pickle.dumps(message)
        length = len(data).to_bytes(4, 'big')
        self.conn.sendall(length + data)
    
    def receive(self, timeout: Optional[float] = None) -> Any:
        """
        Receive message.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Received message
        """
        try:
            return self.receive_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Close connection"""
        self.running = False
        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()
        if self.receive_thread:
            self.receive_thread.join()


class PartyCommunicator:
    """Manages communication for an MPC party"""
    
    def __init__(self, party_id: str):
        """
        Initialize party communicator.
        
        Args:
            party_id: Party identifier ('cp1', 'cp2', 'gateway', 'cp0')
        """
        self.party_id = party_id
        self.channels = {}
    
    def add_channel(self, name: str, channel: MPCChannel):
        """Add a communication channel"""
        self.channels[name] = channel
    
    def connect_all(self):
        """Connect all channels"""
        for channel in self.channels.values():
            channel.connect()
    
    def send_to(self, target: str, message: Any):
        """Send message to target party"""
        if target in self.channels:
            self.channels[target].send(message)
        else:
            raise ValueError(f"No channel to {target}")
    
    def receive_from(self, source: str, timeout: Optional[float] = None) -> Any:
        """Receive message from source party"""
        if source in self.channels:
            return self.channels[source].receive(timeout)
        else:
            raise ValueError(f"No channel from {source}")
    
    def close_all(self):
        """Close all channels"""
        for channel in self.channels.values():
            channel.close()
