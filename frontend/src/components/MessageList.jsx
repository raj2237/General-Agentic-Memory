import React from "react";
import SourceCard from "./SourceCard";

export default function MessageList({ messages }) {
    return (
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
                                    <SourceCard key={idx} source={source} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}
