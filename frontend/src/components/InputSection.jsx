import React from "react";

export default function InputSection({ input, setInput, sendMessage, loading, indexing }) {
    return (
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
    );
}
