import React, { useRef, useEffect } from "react";

export default function KnowledgeGraph({ graphData }) {
    const graphCanvasRef = useRef(null);

    // Render knowledge graph on canvas with beautiful visuals
    useEffect(() => {
        if (!graphCanvasRef.current || !graphData.nodes.length) return;

        const canvas = graphCanvasRef.current;
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.offsetWidth * 2; // 2x for retina
        const height = canvas.height = canvas.offsetHeight * 2;
        canvas.style.width = canvas.offsetWidth + 'px';
        canvas.style.height = canvas.offsetHeight + 'px';

        // Clear canvas with gradient background
        const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
        bgGradient.addColorStop(0, '#0f172a');
        bgGradient.addColorStop(1, '#1e293b');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        // Enhanced hierarchical layout based on node types
        const centerX = width / 2;
        const centerY = height / 2;

        // Separate nodes by type
        const docNodes = graphData.nodes.filter(n => n.type === 'document');
        const chunkNodes = graphData.nodes.filter(n => n.type === 'chunk');
        const entityNodes = graphData.nodes.filter(n => n.type === 'entity');

        const nodes = graphData.nodes.map((node, i) => {
            let x, y, radius, color;

            // Position based on node type
            if (node.type === 'document') {
                // Documents in center
                const docIdx = docNodes.indexOf(node);
                const docAngle = (docIdx * 2 * Math.PI / Math.max(docNodes.length, 1));
                x = centerX + Math.cos(docAngle) * 100;
                y = centerY + Math.sin(docAngle) * 100;
                radius = node.size || 20;
                color = node.color || '#4A90E2';
            } else if (node.type === 'chunk') {
                // Chunks in middle ring
                const chunkIdx = chunkNodes.indexOf(node);
                const chunkAngle = (chunkIdx * 2 * Math.PI / Math.max(chunkNodes.length, 1));
                const chunkRadius = Math.min(width, height) * 0.25;
                x = centerX + Math.cos(chunkAngle) * chunkRadius;
                y = centerY + Math.sin(chunkAngle) * chunkRadius;
                radius = node.size || 12;
                color = node.color || '#50C878';
            } else if (node.type === 'entity') {
                // Entities in outer ring
                const entityIdx = entityNodes.indexOf(node);
                const entityAngle = (entityIdx * 2 * Math.PI / Math.max(entityNodes.length, 1));
                const entityRadius = Math.min(width, height) * 0.4;
                x = centerX + Math.cos(entityAngle) * entityRadius;
                y = centerY + Math.sin(entityAngle) * entityRadius;
                radius = node.size || 8;
                color = node.color || '#FFB347';
            } else {
                // Fallback circular layout
                const angle = (i * 2 * Math.PI / graphData.nodes.length);
                x = centerX + Math.cos(angle) * Math.min(width, height) * 0.35;
                y = centerY + Math.sin(angle) * Math.min(width, height) * 0.35;
                radius = 10;
                color = '#888';
            }

            return { ...node, x, y, radius, color };
        });

        // Draw edges with different styles based on relationship type
        graphData.edges.forEach((edge) => {
            const source = nodes.find(n => n.id === edge.source);
            const target = nodes.find(n => n.id === edge.target);
            if (source && target) {
                // Edge styling based on label
                let strokeColor, lineWidth, lineDash;

                if (edge.label === 'contains') {
                    strokeColor = 'rgba(74, 144, 226, 0.6)';
                    lineWidth = 2;
                    lineDash = [];
                } else if (edge.label === 'mentions') {
                    strokeColor = 'rgba(255, 179, 71, 0.5)';
                    lineWidth = 1.5;
                    lineDash = [5, 5];
                } else if (edge.label === 'next') {
                    strokeColor = 'rgba(80, 200, 120, 0.4)';
                    lineWidth = 1;
                    lineDash = [3, 3];
                } else {
                    strokeColor = 'rgba(150, 150, 150, 0.4)';
                    lineWidth = 1;
                    lineDash = [];
                }

                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
                ctx.setLineDash(lineDash);
                ctx.globalAlpha = 0.7;

                // Draw straight or curved line
                ctx.beginPath();
                ctx.moveTo(source.x, source.y);

                if (edge.label === 'contains' || edge.label === 'mentions') {
                    // Curved for hierarchical relationships
                    const midX = (source.x + target.x) / 2;
                    const midY = (source.y + target.y) / 2;
                    const offset = 20;
                    ctx.quadraticCurveTo(midX + offset, midY - offset, target.x, target.y);
                } else {
                    // Straight for sequential
                    ctx.lineTo(target.x, target.y);
                }

                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            }
        });

        // Draw nodes with type-specific styling
        nodes.forEach((node) => {
            // Outer glow
            ctx.shadowBlur = 15;
            ctx.shadowColor = node.color;

            // Gradient fill
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius
            );
            gradient.addColorStop(0, node.color);
            gradient.addColorStop(0.7, node.color);
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.3)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
            ctx.fill();

            // Border
            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = node.type === 'document' ? 3 : 2;
            ctx.stroke();

            // Inner highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.beginPath();
            ctx.arc(node.x - node.radius / 3, node.y - node.radius / 3, node.radius / 4, 0, 2 * Math.PI);
            ctx.fill();

            // Node label with shadow
            ctx.shadowBlur = 3;
            ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
            ctx.fillStyle = '#ffffff';
            ctx.font = node.type === 'document' ? 'bold 18px Inter, sans-serif' : 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const label = node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label;
            ctx.fillText(label, node.x, node.y + node.radius + 20);
            ctx.shadowBlur = 0;
        });
    }, [graphData]);

    return (
        <div className="side-box">
            <div className="side-box-title">Knowledge Graph Visualization</div>
            {graphData.nodes.length > 0 ? (
                <div className="graph-container">
                    <canvas ref={graphCanvasRef} className="graph-canvas"></canvas>
                    <div className="graph-stats">
                        {graphData.nodes.length} entities • {graphData.edges.length} relations
                    </div>
                </div>
            ) : (
                <div className="no-data">No graph data yet</div>
            )}
        </div>
    );
}
