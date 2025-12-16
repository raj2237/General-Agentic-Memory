import React from "react";
import MessageList from "./MessageList";
import InputSection from "./InputSection";
import UploadState from "./UploadState";

export default function ChatArea({
    fileUploaded,
    uploading,
    triggerFileInput,
    messages,
    input,
    setInput,
    sendMessage,
    loading,
    indexing,
    clearing,
    clearMemory,
    fileInputRef,
    handleFileUpload
}) {
    return (
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
                <UploadState uploading={uploading} triggerFileInput={triggerFileInput} />
            ) : (
                /* After upload: Show chat interface */
                <>
                    <MessageList messages={messages} />

                    {/* INPUT */}
                    <InputSection
                        input={input}
                        setInput={setInput}
                        sendMessage={sendMessage}
                        loading={loading}
                        indexing={indexing}
                    />

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
    );
}
