import React from "react";

export default function RetrievalDetails({ messages }) {
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
    const hasRetrievalDetails = lastMessage?.sender === "ai" && lastMessage?.retrievalDetails?.length > 0;

    return (
        <div className="side-box">
            <div className="side-box-title">Relevance Score + Retrieved Sources</div>
            {hasRetrievalDetails ? (
                <div className="retrieval-details">
                    {lastMessage.retrievalDetails.map((detail, idx) => (
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
    );
}
