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
    const [fileHistory, setFileHistory] = useState([]);
    const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
    const [currentThinkingSteps, setCurrentThinkingSteps] = useState([]);
    const [indexing, setIndexing] = useState(false);
    const [indexingProgress, setIndexingProgress] = useState({ status: "", processed: 0, total: 0 });
    const [currentDocId, setCurrentDocId] = useState(null);
    const fileInputRef = useRef(null);
    const graphCanvasRef = useRef(null);

    // Poll indexing status
    React.useEffect(() => {
        if (!currentDocId || !indexing) return;

        const pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/indexing-status/${currentDocId}`);
                if (res.ok) {
                    const data = await res.json();
                    setIndexingProgress({
                        status: data.status,
                        processed: data.processed_chunks || 0,
                        total: data.total_chunks || 0
                    });

                    if (data.status === "completed") {
                        setIndexing(false);
                        setMessages(prev => [
                            ...prev.filter(m => !m.isIndexing),
                            {
                                sender: "ai",
                                text: `✅ Document indexed successfully! You can now start querying.`
                            }
                        ]);
                        clearInterval(pollInterval);
                    } else if (data.status === "failed") {
                        setIndexing(false);
                        setMessages(prev => [
                            ...prev.filter(m => !m.isIndexing),
                            {
                                sender: "ai",
                                text: `❌ Indexing failed: ${data.error || "Unknown error"}`
                            }
                        ]);
                        clearInterval(pollInterval);
                    }
                }
            } catch (e) {
                console.error("Failed to poll indexing status:", e);
            }
        }, 2000); // Poll every 2 seconds

        return () => clearInterval(pollInterval);
    }, [currentDocId, indexing]);

    const sendMessage = async () => {
        if (!input.trim() || loading || indexing) return;

        const userMessage = input.trim();

        // Add human message
        const userMsg = { sender: "human", text: userMessage };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        // Add loading message
        const loadingMsg = { sender: "ai", text: "Thinking...", isLoading: true };
        setMessages((prev) => [...prev, loadingMsg]);

        // Clear previous thinking steps
        setCurrentThinkingSteps([]);

        try {
            // Use EventSource for streaming
            const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`Stream failed: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let finalAnswer = "";
            let finalGraphData = null;
            let finalRetrievalDetails = [];
            let finalRetrievedCount = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        if (data === "[DONE]") continue;

                        try {
                            const event = JSON.parse(data);

                            if (event.type === "thinking") {
                                // Add thinking step in real-time
                                setCurrentThinkingSteps((prev) => [
                                    ...prev,
                                    {
                                        iteration: event.iteration,
                                        thought: event.thought,
                                        retrieved: event.retrieved,
                                    },
                                ]);
                            } else if (event.type === "answer") {
                                finalAnswer = event.answer;
                                finalGraphData = event.graph_data;
                                finalRetrievalDetails = event.retrieval_details || [];
                                finalRetrievedCount = event.retrieved_chunks_count || 0;
                            } else if (event.type === "error") {
                                throw new Error(event.message);
                            }
                        } catch (parseError) {
                            console.error("Failed to parse event:", parseError);
                        }
                    }
                }
            }

            // Update graph data if present
            if (finalGraphData) {
                setGraphData(finalGraphData);
            }

            // Remove loading message and add final answer
            setMessages((prev) => {
                const withoutLoading = prev.filter((msg) => !msg.isLoading);
                return [
                    ...withoutLoading,
                    {
                        sender: "ai",
                        text: finalAnswer || "No response received",
                        thinkingSteps: currentThinkingSteps,
                        retrievalDetails: finalRetrievalDetails,
                        retrievedChunks: finalRetrievedCount,
                    },
                ];
            });
        } catch (error) {
            // Remove loading message and add error message
            setMessages((prev) => {
                const withoutLoading = prev.filter((msg) => !msg.isLoading);
                return [
                    ...withoutLoading,
                    {
                        sender: "ai",
                        text: `❌ Error: ${error.message}`,
                    },
                ];
            });
        } finally {
            setLoading(false);
        }
    };

    const clearMemory = async () => {
        if (clearing || uploading || loading) return;
        setClearing(true);
        try {
            const res = await fetch(`${API_BASE_URL}/api/memory/clear`, { method: "POST" });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: "Clear failed" }));
                throw new Error(err.detail || err.error || "Clear failed");
            }
            setMessages([{ sender: "ai", text: "Memory cleared. Please upload a document to start fresh." }]);
            setFileUploaded(false);
            setFileHistory([]);
            setGraphData({ nodes: [], edges: [] });
            setCurrentThinkingSteps([]);
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

    const fetchFileHistory = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/files`);
            if (res.ok) {
                const data = await res.json();
                setFileHistory(data.files || []);
            }
        } catch (e) {
            console.error("Failed to fetch file history:", e);
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

            // Start tracking indexing status
            setCurrentDocId(data.doc_id);
            setIndexing(true);
            setIndexingProgress({ status: "indexing", processed: 0, total: data.chunks });

            // Add upload success + indexing message
            setMessages([
                {
                    sender: "ai",
                    text: `✅ Document "${data.filename}" uploaded! (${data.characters} characters, ${data.chunks} chunks)\n\n🔄 Indexing in progress... Please wait before querying.`,
                    isIndexing: true
                },
            ]);

            // Refresh file history
            await fetchFileHistory();

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

    // Render knowledge graph on canvas with beautiful visuals
    React.useEffect(() => {
        if (!graphCanvasRef.current || !graphData.nodes.length) return;

        const canvas = graphCanvasRef.current;
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.offsetWidth * 2; // 2x for retina
        const height = canvas.height = canvas.offsetHeight * 2;
        canvas.style.width = canvas.offsetWidth + 'px';
        canvas.style.height = canvas.offsetHeight + 'px';

        // Clear canvas with gradient background
        const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
        bgGradient.addColorStop(0, '#0f172a');
        bgGradient.addColorStop(1, '#1e293b');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        // Enhanced hierarchical layout based on node types
        const centerX = width / 2;
        const centerY = height / 2;

        // Separate nodes by type
        const docNodes = graphData.nodes.filter(n => n.type === 'document');
        const chunkNodes = graphData.nodes.filter(n => n.type === 'chunk');
        const entityNodes = graphData.nodes.filter(n => n.type === 'entity');

        const nodes = graphData.nodes.map((node, i) => {
            let x, y, radius, color;

            // Position based on node type
            if (node.type === 'document') {
                // Documents in center
                const docIdx = docNodes.indexOf(node);
                const docAngle = (docIdx * 2 * Math.PI / Math.max(docNodes.length, 1));
                x = centerX + Math.cos(docAngle) * 100;
                y = centerY + Math.sin(docAngle) * 100;
                radius = node.size || 20;
                color = node.color || '#4A90E2';
            } else if (node.type === 'chunk') {
                // Chunks in middle ring
                const chunkIdx = chunkNodes.indexOf(node);
                const chunkAngle = (chunkIdx * 2 * Math.PI / Math.max(chunkNodes.length, 1));
                const chunkRadius = Math.min(width, height) * 0.25;
                x = centerX + Math.cos(chunkAngle) * chunkRadius;
                y = centerY + Math.sin(chunkAngle) * chunkRadius;
                radius = node.size || 12;
                color = node.color || '#50C878';
            } else if (node.type === 'entity') {
                // Entities in outer ring
                const entityIdx = entityNodes.indexOf(node);
                const entityAngle = (entityIdx * 2 * Math.PI / Math.max(entityNodes.length, 1));
                const entityRadius = Math.min(width, height) * 0.4;
                x = centerX + Math.cos(entityAngle) * entityRadius;
                y = centerY + Math.sin(entityAngle) * entityRadius;
                radius = node.size || 8;
                color = node.color || '#FFB347';
            } else {
                // Fallback circular layout
                const angle = (i * 2 * Math.PI / graphData.nodes.length);
                x = centerX + Math.cos(angle) * Math.min(width, height) * 0.35;
                y = centerY + Math.sin(angle) * Math.min(width, height) * 0.35;
                radius = 10;
                color = '#888';
            }

            return { ...node, x, y, radius, color };
        });

        // Draw edges with different styles based on relationship type
        graphData.edges.forEach((edge) => {
            const source = nodes.find(n => n.id === edge.source);
            const target = nodes.find(n => n.id === edge.target);
            if (source && target) {
                // Edge styling based on label
                let strokeColor, lineWidth, lineDash;

                if (edge.label === 'contains') {
                    strokeColor = 'rgba(74, 144, 226, 0.6)';
                    lineWidth = 2;
                    lineDash = [];
                } else if (edge.label === 'mentions') {
                    strokeColor = 'rgba(255, 179, 71, 0.5)';
                    lineWidth = 1.5;
                    lineDash = [5, 5];
                } else if (edge.label === 'next') {
                    strokeColor = 'rgba(80, 200, 120, 0.4)';
                    lineWidth = 1;
                    lineDash = [3, 3];
                } else {
                    strokeColor = 'rgba(150, 150, 150, 0.4)';
                    lineWidth = 1;
                    lineDash = [];
                }

                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
                ctx.setLineDash(lineDash);
                ctx.globalAlpha = 0.7;

                // Draw straight or curved line
                ctx.beginPath();
                ctx.moveTo(source.x, source.y);

                if (edge.label === 'contains' || edge.label === 'mentions') {
                    // Curved for hierarchical relationships
                    const midX = (source.x + target.x) / 2;
                    const midY = (source.y + target.y) / 2;
                    const offset = 20;
                    ctx.quadraticCurveTo(midX + offset, midY - offset, target.x, target.y);
                } else {
                    // Straight for sequential
                    ctx.lineTo(target.x, target.y);
                }

                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            }
        });

        // Draw nodes with type-specific styling
        nodes.forEach((node) => {
            // Outer glow
            ctx.shadowBlur = 15;
            ctx.shadowColor = node.color;

            // Gradient fill
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius
            );
            gradient.addColorStop(0, node.color);
            gradient.addColorStop(0.7, node.color);
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.3)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
            ctx.fill();

            // Border
            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = node.type === 'document' ? 3 : 2;
            ctx.stroke();

            // Inner highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.beginPath();
            ctx.arc(node.x - node.radius / 3, node.y - node.radius / 3, node.radius / 4, 0, 2 * Math.PI);
            ctx.fill();

            // Node label with shadow
            ctx.shadowBlur = 3;
            ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
            ctx.fillStyle = '#ffffff';
            ctx.font = node.type === 'document' ? 'bold 18px Inter, sans-serif' : 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const label = node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label;
            ctx.fillText(label, node.x, node.y + node.radius + 20);
            ctx.shadowBlur = 0;
        });
    }, [graphData]);

    return (
        <div className="agent-layout">
            {/* LEFT FILE HISTORY PANEL */}
            {fileUploaded && (
                <div className="file-history-panel">
                    <div className="panel-title">Uploaded Files</div>
                    <div className="file-list">
                        {fileHistory.length > 0 ? (
                            fileHistory.map((file) => (
                                <div key={file.id} className="file-item">
                                    <div className="file-icon">📄</div>
                                    <div className="file-info">
                                        <div className="file-name">{file.filename}</div>
                                        <div className="file-meta">
                                            {(file.size / 1000).toFixed(1)}KB • {file.processed_chunks || file.chunks} chunks
                                        </div>
                                        <div className="file-date">
                                            {new Date(file.uploaded_at).toLocaleTimeString()}
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="no-files">No files uploaded yet</div>
                        )}
                    </div>
                </div>
            )}

            {/* CENTER CHAT AREA */}
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
                                    <div className="message-text">{msg.text}</div>

                                    {/* Show retrieved sources if available */}
                                    {msg.retrievalDetails && msg.retrievalDetails.length > 0 && (
                                        <div className="sources-section">
                                            <div className="sources-header">
                                                📚 Retrieved from {msg.retrievedChunks || msg.retrievalDetails.length} source{(msg.retrievedChunks || msg.retrievalDetails.length) > 1 ? 's' : ''}
                                            </div>
                                            <div className="sources-list">
                                                {msg.retrievalDetails.map((source, idx) => (
                                                    <div key={idx} className="source-card">
                                                        <div className="source-header">
                                                            <span className="source-doc">📄 {source.document_name || 'Document'}</span>
                                                            <span className="source-score" title="Relevance Score">
                                                                {(source.relevance_score * 100).toFixed(0)}%
                                                            </span>
                                                        </div>
                                                        <div className="source-snippet">{source.snippet}</div>
                                                        <div className="source-meta">
                                                            <span className="source-type">{source.source_type}</span>
                                                            <span className="source-id">Chunk #{source.source_id}</span>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        {/* INPUT */}
                        <div className="input-section">
                            <input
                                type="text"
                                placeholder={indexing ? "⏳ Indexing document... Please wait" : loading ? "Thinking..." : "Type your message..."}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && !loading && !indexing && sendMessage()}
                                disabled={loading || indexing}
                            />
                            <button
                                className="round-btn"
                                onClick={sendMessage}
                                disabled={loading || indexing}
                                style={{ opacity: (loading || indexing) ? 0.6 : 1, cursor: (loading || indexing) ? "not-allowed" : "pointer" }}
                            >
                                {loading ? "..." : indexing ? "⏳" : "→"}
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
                {/* Knowledge Graph */}
                <div className="side-box">
                    <div className="side-box-title">Knowledge Graph Visualization</div>
                    {graphData.nodes.length > 0 ? (
                        <div className="graph-container">
                            <canvas ref={graphCanvasRef} className="graph-canvas"></canvas>
                            <div className="graph-stats">
                                {graphData.nodes.length} entities • {graphData.edges.length} relations
                            </div>
                        </div>
                    ) : (
                        <div className="no-data">No graph data yet</div>
                    )}
                </div>

                {/* Thinking Steps */}
                <div className="side-box">
                    <div className="side-box-title">Real-time Agent Thinking</div>
                    {currentThinkingSteps.length > 0 ? (
                        <div className="thinking-container">
                            {currentThinkingSteps.map((step, idx) => (
                                <div key={idx} className="thinking-step" data-iteration={step.iteration}>
                                    <div className="thinking-step-text">{step.thought}</div>
                                    <div className="thinking-step-meta">
                                        {step.retrieved > 0 && (
                                            <span className="thinking-step-retrieved">
                                                {step.retrieved} chunks retrieved
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="no-data">No thinking data yet</div>
                    )}
                </div>

                {/* Retrieval Details */}
                <div className="side-box">
                    <div className="side-box-title">Relevance Score + Retrieved Sources</div>
                    {messages.length > 0 && messages[messages.length - 1].sender === "ai" && messages[messages.length - 1].retrievalDetails?.length > 0 ? (
                        <div className="retrieval-details">
                            {messages[messages.length - 1].retrievalDetails.map((detail, idx) => (
                                <div key={idx} className="retrieval-item">
                                    <div className="retrieval-header">
                                        <span className="source-id">Source {detail.source_id}</span>
                                        <span className="relevance-score">{(detail.relevance_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="source-type">{detail.source_type}</div>
                                    <div className="snippet">{detail.snippet}</div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="no-data">No retrieval data yet</div>
                    )}
                </div>
            </div>
        </div >




    );
}
