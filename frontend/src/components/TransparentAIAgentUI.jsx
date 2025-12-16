import React, { useState, useRef } from "react";
import "./TransparentAIAgentUI.css";
import { API_BASE_URL } from "./constants";
import FileHistoryPanel from "./FileHistoryPanel";
import ChatArea from "./ChatArea";
import SidePanel from "./SidePanel";

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

    return (
        <div className="agent-layout">
            {/* LEFT FILE HISTORY PANEL */}
            {fileUploaded && (
                <FileHistoryPanel fileHistory={fileHistory} />
            )}

            {/* CENTER CHAT AREA */}
            <ChatArea
                fileUploaded={fileUploaded}
                uploading={uploading}
                triggerFileInput={triggerFileInput}
                messages={messages}
                input={input}
                setInput={setInput}
                sendMessage={sendMessage}
                loading={loading}
                indexing={indexing}
                clearing={clearing}
                clearMemory={clearMemory}
                fileInputRef={fileInputRef}
                handleFileUpload={handleFileUpload}
            />

            {/* RIGHT SIDEBAR */}
            <SidePanel
                graphData={graphData}
                currentThinkingSteps={currentThinkingSteps}
                messages={messages}
            />
        </div>
    );
}
