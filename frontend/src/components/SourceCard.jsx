import React from "react";

export default function SourceCard({ source }) {
    return (
        <div className="source-card">
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
    );
}
