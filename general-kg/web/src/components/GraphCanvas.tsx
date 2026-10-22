import { useEffect, useRef } from "react";
import cytoscape, { type Core } from "cytoscape";
import { Maximize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import type { GraphEdge, GraphNode } from "../types";
import { colorForType } from "./graphColors";

type Selection = { kind: "node" | "edge"; id: string } | null;

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selection: Selection;
  onSelect: (selection: Selection) => void;
};

export default function GraphCanvas({ nodes, edges, selection, onSelect }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [
        ...nodes.map((node) => ({
          data: { ...node, color: colorForType(node.type) },
        })),
        ...edges.map((edge) => ({ data: { ...edge } })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            color: "#26343c",
            "font-size": 11,
            "font-weight": 600,
            "text-wrap": "wrap",
            "text-max-width": "110px",
            "text-valign": "bottom",
            "text-margin-y": 7,
            width: 38,
            height: 38,
            "border-width": 2,
            "border-color": "#ffffff",
            "overlay-opacity": 0,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 4,
            "border-color": "#173f3b",
            width: 46,
            height: 46,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#aab8bf",
            "target-arrow-color": "#8f9ea6",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "",
            color: "#56666e",
            "font-size": 8,
            "text-rotation": "autorotate",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.88,
            "text-background-padding": "2px",
            "overlay-opacity": 0,
          },
        },
        {
          selector: "edge:selected",
          style: {
            width: 3,
            "line-color": "#087e78",
            "target-arrow-color": "#087e78",
            color: "#174f4b",
            "font-weight": 700,
            label: "data(label)",
          },
        },
      ],
      layout: {
        name: "cose",
        animate: false,
        padding: 56,
        nodeRepulsion: 16000,
        idealEdgeLength: 170,
        nodeOverlap: 24,
        gravity: 0.14,
        randomize: true,
      },
      minZoom: 0.35,
      maxZoom: 2.4,
      wheelSensitivity: 0.18,
      boxSelectionEnabled: false,
    });

    cy.on("tap", "node", (event) => onSelect({ kind: "node", id: event.target.id() }));
    cy.on("tap", "edge", (event) => onSelect({ kind: "edge", id: event.target.id() }));
    cy.on("tap", (event) => {
      if (event.target === cy) onSelect(null);
    });
    cyRef.current = cy;
    const resizeObserver = new ResizeObserver(() => {
      cy.resize();
      cy.fit(undefined, 38);
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, [nodes, edges, onSelect]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect();
    if (selection) cy.getElementById(selection.id).select();
  }, [selection]);

  const zoomBy = (factor: number) => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
  };

  const fit = () => cyRef.current?.fit(undefined, 38);
  const relayout = () => cyRef.current?.layout({ name: "cose", animate: true, padding: 64, nodeRepulsion: 16000, idealEdgeLength: 170, nodeOverlap: 24, gravity: 0.14 }).run();

  return (
    <div className="graph-canvas" ref={containerRef} aria-label="交互式知识图谱画布">
      <div className="canvas-tools">
        <button className="icon-button" type="button" title="放大" aria-label="放大" onClick={() => zoomBy(1.18)}><ZoomIn size={16} /></button>
        <button className="icon-button" type="button" title="缩小" aria-label="缩小" onClick={() => zoomBy(0.84)}><ZoomOut size={16} /></button>
        <span className="tool-divider" />
        <button className="icon-button" type="button" title="适配画布" aria-label="适配画布" onClick={fit}><Maximize2 size={15} /></button>
        <button className="icon-button" type="button" title="重新布局" aria-label="重新布局" onClick={relayout}><RotateCcw size={15} /></button>
      </div>
    </div>
  );
}
