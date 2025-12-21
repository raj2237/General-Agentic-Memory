import React from "react";
import KnowledgeGraph from "./KnowledgeGraph";
import ThinkingSteps from "./ThinkingSteps";
import RetrievalDetails from "./RetrievalDetails";
import Popup from "reactjs-popup";
import "reactjs-popup/dist/index.css";

export default function SidePanel({ graphData, currentThinkingSteps, messages }) {
    return (
        <div className="side-panel">
            {/**GRAPH DATA POPUP */}
            <Popup
                trigger={
                    <div className="clickable-content">
                        <KnowledgeGraph graphData={graphData} />
                    </div>
                }
                modal
                nested
            >
                {
                    (close) => (
                        <div className="popup-box">
                            <div className="popup-header">
                                <h2>Graph Data</h2>
                                <button onClick={close}>X</button>
                            </div>

                            <KnowledgeGraph graphData={graphData} />

                        </div>
                    )
                }
            </Popup>

            {/** Thinking Steps Popup */}
            <Popup
                trigger={
                    <div className="clickable-content">
                        <ThinkingSteps currentThinkingSteps={currentThinkingSteps} />
                    </div>
                }
                modal
                nested
            >
                {
                    (close) => (
                        <div className="popup-box">
                            <div className="popup-header">
                                <h2>thinking steps</h2>
                                <button onClick={close}>X</button>
                            </div>

                            <ThinkingSteps currentThinkingSteps={currentThinkingSteps} />

                        </div>
                    )
                }
            </Popup>

            {/** Retrieval Details Popup */}
            <Popup
                trigger={
                    <div className="clickable-content">
                        <RetrievalDetails messages={messages} />
                    </div>
                }
                modal
                nested
            >
                {
                    (close) => (
                        <div className="popup-box">
                            <div className="popup-header">
                                <h2>thinking steps</h2>
                                <button onClick={close}>X</button>
                            </div>

                            <RetrievalDetails messages={messages} />

                        </div>
                    )
                }
            </Popup>

            {/* <KnowledgeGraph graphData={graphData} /> 
            <ThinkingSteps currentThinkingSteps={currentThinkingSteps} />
            <RetrievalDetails messages={messages} /> */}
        </div>
    );
}
