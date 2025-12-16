import React from "react";

export default function ThinkingSteps({ currentThinkingSteps }) {
    return (
        <div className="side-box">
            <div className="side-box-title">Real-time Agent Thinking</div>
            {currentThinkingSteps.length > 0 ? (
                <div className="thinking-container">
                    {currentThinkingSteps.map((step, idx) => (
                        <div key={idx} className="thinking-step" data-iteration={step.iteration}>
                            <div className="thinking-step-text">{step.thought}</div>
                            <div className="thinking-step-meta">
                                {step.retrieved > 0 && (
                                    <span className="thinking-step-retrieved">
                                        {step.retrieved} chunks retrieved
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="no-data">No thinking data yet</div>
            )}
        </div>
    );
}
