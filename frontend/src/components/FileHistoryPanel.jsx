import React from "react";

export default function FileHistoryPanel({ fileHistory }) {
    return (
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
    );
}
