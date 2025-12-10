import React, { useState, useRef } from "react";
import "./TransparentAIAgentUI.css";

const API_BASE_URL = "http://localhost:8000";

export default function TransparentAIAgentUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const fileInputRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    
    // Add human message
    const userMsg = { sender: "human", text: userMessage };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Add loading message
    const loadingMsg = { sender: "ai", text: "Thinking...", isLoading: true };
    setMessages((prev) => [...prev, loadingMsg]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Chat failed" }));
        throw new Error(errorData.detail || errorData.error || `Chat failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Remove loading message and add AI response
      setMessages((prev) => {
        const withoutLoading = prev.filter(msg => !msg.isLoading);
        return [...withoutLoading, {
          sender: "ai",
          text: data.answer || "No response received",
          thinkingSteps: data.thinking_steps || [],
          retrievedChunks: data.retrieved_chunks_count || 0,
        }];
      });
    } catch (error) {
      // Remove loading message and add error message
      setMessages((prev) => {
        const withoutLoading = prev.filter(msg => !msg.isLoading);
        return [...withoutLoading, {
          sender: "ai",
          text: `❌ Error: ${error.message}`,
        }];
      });
    } finally {
      setLoading(false);
    }
  };

  const clearMemory = async () => {
    if (clearing || uploading || loading) return;
    setClearing(true);
    try {
      const res = await fetch(`${API_BASE_URL}/memory/clear`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Clear failed" }));
        throw new Error(err.detail || err.error || "Clear failed");
      }
      setMessages([{ sender: "ai", text: "Memory cleared. Please upload a document to start fresh." }]);
      setFileUploaded(false);
      setInput("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (e) {
      alert(`Clear memory failed: ${e.message}`);
    } finally {
      setClearing(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Mark file as uploaded to show chat interface
      setFileUploaded(true);

      // Add success message to chat
      setMessages([
        {
          sender: "ai",
          text: `✅ Document "${data.filename}" has been uploaded and indexed! (${data.characters} characters extracted)\n\nHello! How can I assist you today?`,
        },
      ]);

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      // Show error but don't enable chat yet
      alert(`Upload failed: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="agent-layout">
      {/* LEFT CHAT AREA */}
      <div className="chat-area">
        <div className="top-bar"></div>

        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          style={{ display: "none" }}
          accept=".pdf,.txt,.docx,.doc,.md"
        />

        {!fileUploaded ? (
          /* Initial state: Show only the center plus icon for upload */
          <div className="upload-state">
            <div 
              className={`center-plus ${uploading ? "uploading" : ""}`}
              onClick={uploading ? undefined : triggerFileInput}
              style={{ cursor: uploading ? "not-allowed" : "pointer" }}
            >
              {uploading ? "⏳" : "+"}
            </div>
            {uploading && (
              <div className="uploading-text">Uploading document...</div>
            )}
          </div>
        ) : (
          /* After upload: Show chat interface */
          <>
            <div className="messages-container">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.sender} ${msg.isLoading ? 'isLoading' : ''}`}>
                  {msg.text}
                </div>
              ))}
            </div>

            {/* INPUT */}
            <div className="input-section">
              <input
                type="text"
                placeholder={loading ? "Thinking..." : "Type your message..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !loading && sendMessage()}
                disabled={loading}
              />
              <button 
                className="round-btn" 
                onClick={sendMessage}
                disabled={loading}
                style={{ opacity: loading ? 0.6 : 1, cursor: loading ? "not-allowed" : "pointer" }}
              >
                {loading ? "..." : ""}
              </button>
            </div>

            <div className="thread-list">
              Create a message before starting a new thread
            </div>

            <div className="controls-row">
              <button 
                className="secondary-btn"
                onClick={triggerFileInput}
                disabled={uploading || loading}
              >
                {uploading ? "Uploading..." : "Upload new doc"}
              </button>
              <button 
                className="secondary-btn"
                onClick={clearMemory}
                disabled={clearing || loading || uploading}
              >
                {clearing ? "Clearing..." : "Clear previous memory"}
              </button>
            </div>
          </>
        )}
      </div>

      {/* RIGHT SIDEBAR */}
      <div className="side-panel">
        <div className="side-box">Knowledge Graph Visualization</div>
        <div className="side-box">Real-time Agent Thinking / Mapping</div>
        <div className="side-box">
          Relevance Score + Retrieved Sources
        </div>
      </div>
    </div>
  );
}
