import React from "react";

export default function UploadState({ uploading, triggerFileInput }) {
    return (
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
    );
}
