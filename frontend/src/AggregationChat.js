import React, { useState, useEffect, useRef } from 'react';
import { PlusCircle, RefreshCw } from 'lucide-react';

// Message component for chat bubbles
const ChatMessage = ({ message, isUser }) => {
  const messageClass = isUser 
    ? "bg-blue-100 text-blue-800 ml-auto" 
    : "bg-gray-100 text-gray-800 mr-auto";
  
  return (
    <div className={`rounded-lg p-3 my-2 max-w-3/4 ${messageClass}`}>
      <p className="whitespace-pre-wrap">{message}</p>
    </div>
  );
};

// Loading indicator for chat
const ChatLoadingIndicator = () => (
  <div className="flex items-center space-x-2 p-2 text-gray-500">
    <div className="animate-pulse h-2 w-2 rounded-full bg-blue-500"></div>
    <div className="animate-pulse h-2 w-2 rounded-full bg-blue-500" style={{ animationDelay: '0.2s' }}></div>
    <div className="animate-pulse h-2 w-2 rounded-full bg-blue-500" style={{ animationDelay: '0.4s' }}></div>
  </div>
);

// Main chat component
const AggregationChat = ({ 
  sessionId, 
  aggregation, 
  onNewAggregation,
  initialMessages = []
}) => {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);

  // Scroll to bottom of chat when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load chat history for this aggregation on component mount
  useEffect(() => {
    if (sessionId && aggregation && aggregation.id) {
      fetchChatHistory();
    }
  }, [sessionId, aggregation?.id]);

  const fetchChatHistory = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/chat-history/${sessionId}/${aggregation.id}`);
      const data = await response.json();
      
      if (data.chat_history && data.chat_history.length > 0) {
        const formattedMessages = data.chat_history.flatMap(entry => [
          { text: entry.user, isUser: true },
          { text: entry.assistant, isUser: false }
        ]);
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const generateNewAggregation = async () => {
    if (!input.trim()) return;
    
    const userMessage = input.trim();
    setInput('');
    setLoading(true);
    setError(null);
    
    // Add user message to chat
    setMessages(prev => [...prev, { 
      text: userMessage, 
      isUser: true 
    }]);
    
    try {
      const response = await fetch('http://localhost:8000/api/generate-from-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          feedback: userMessage,
          aggregation_id: aggregation.id // Adding this to provide context
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate new aggregation');
      }
      
      // Add success message to chat
      setMessages(prev => [...prev, { 
        text: `Successfully created a new aggregation based on your feedback: "${data.new_aggregation.title}"
        
This new aggregation specifically focuses on addressing your request: ${userMessage}
        
You can find the new analysis in the aggregations list.`, 
        isUser: false 
      }]);
      
      // Notify parent component about the new aggregation
      if (onNewAggregation) {
        onNewAggregation(data.new_aggregation);
      }
      
    } catch (err) {
      setError(`Error: ${err.message}`);
      console.error('Error generating new aggregation:', err);
      
      // Add error message to chat
      setMessages(prev => [...prev, { 
        text: `Error: Failed to generate new aggregation. ${err.message}`, 
        isUser: false 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      generateNewAggregation();
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow overflow-hidden">
      {/* Chat header */}
      <div className="bg-blue-50 p-3 border-b border-blue-100">
        <h3 className="text-lg font-medium text-blue-800">
          Feedback & New Aggregations
        </h3>
        <p className="text-sm text-blue-600">
          Provide feedback to generate new targeted aggregations based on your specific needs
        </p>
      </div>
      
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2" style={{ maxHeight: '400px' }}>
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 my-8">
            <p>No messages yet. Start by providing feedback on what additional analysis you'd like to see.</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <ChatMessage 
              key={idx} 
              message={msg.text} 
              isUser={msg.isUser} 
            />
          ))
        )}
        
        {loading && <ChatLoadingIndicator />}
        {error && (
          <div className="text-red-500 text-sm p-2 bg-red-50 rounded">
            {error}
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      
      {/* Chat input */}
      <div className="border-t border-gray-200 p-3">
        <div className="flex items-center">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What additional analysis would you like to see?"
            className="flex-1 p-2 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={2}
            disabled={loading}
          />
          <div className="ml-2 flex-shrink-0">
            <button
              onClick={generateNewAggregation}
              disabled={!input.trim() || loading}
              className="p-3 bg-green-600 text-white rounded-full hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed"
              title="Generate new analysis from your feedback"
            >
              <PlusCircle size={20} />
            </button>
          </div>
        </div>
        
        {/* Action description */}
        <div className="text-xs text-gray-500 mt-2">
          <span>Press Enter or use the button to generate a new analysis based on your feedback</span>
        </div>
        
        {/* Loading indicator for new aggregation generation */}
        {loading && (
          <div className="flex items-center justify-center text-sm text-green-600 mt-2">
            <RefreshCw size={16} className="animate-spin mr-2" />
            Generating new aggregation... (This may take a minute)
          </div>
        )}
      </div>
    </div>
  );
};

export default AggregationChat;