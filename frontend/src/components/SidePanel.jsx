import React from "react";
import KnowledgeGraph from "./KnowledgeGraph";
import ThinkingSteps from "./ThinkingSteps";
import RetrievalDetails from "./RetrievalDetails";

export default function SidePanel({ graphData, currentThinkingSteps, messages }) {
    return (
        <div className="side-panel">
            <KnowledgeGraph graphData={graphData} />
            <ThinkingSteps currentThinkingSteps={currentThinkingSteps} />
            <RetrievalDetails messages={messages} />
        </div>
    );
}
