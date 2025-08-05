from typing import List, Dict, Optional
from .tracing import TraceNode

class MermaidDiagramGenerator:
    
    def __init__(self):
        self.node_counter = 0
        self.node_ids: Dict[str, str] = {}
        
    def generate(self, trace: TraceNode) -> str:
        self.node_counter = 0
        self.node_ids = {}
        
        lines = [
            "```mermaid",
            "graph TD",
            "    %% Tölvera LLM Pipeline Trace",
            f"    %% Input: {trace.input_data.get('description', 'N/A')}",
            f"    %% Duration: {self._format_duration(trace.duration_ms)}",
            ""
        ]
        
        self._add_node(trace, lines)
        self._add_connections(trace, lines)
        
        lines.extend([
            "",
            "    %% Styling",
            "    classDef synthesis fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;",
            "    classDef decomposition fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;",
            "    classDef llm_call fill:#fff3e0,stroke:#f57c00,stroke-width:2px;",
            "    classDef routing fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;",
            "    classDef parsing fill:#fce4ec,stroke:#c2185b,stroke-width:2px;",
            "    classDef drawing fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;",
            "    classDef state_analysis fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;",
            "    classDef temporal_update fill:#fffde7,stroke:#f9a825,stroke-width:2px;",
            "    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:3px;",
            "    classDef success fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;",
            "    classDef demo fill:#f5f5f5,stroke:#666666,stroke-width:2px;",
            "    classDef default fill:#f5f5f5,stroke:#666666,stroke-width:2px;",
            "",
            "    %% Apply styles",
        ])
        
        known_types = {
            'synthesis', 'decomposition', 'llm_call', 'routing', 
            'parsing', 'drawing', 'state_analysis', 'temporal_update',
            'demo', 'error', 'success'
        }
        
        for node_id, node_data in self._collect_all_nodes(trace).items():
            node = node_data['node']
            if node.status == "error":
                lines.append(f"    class {node_id} error")
            elif node.status == "success" and node.type == "synthesis":
                lines.append(f"    class {node_id} success")
            elif node.type in known_types:
                lines.append(f"    class {node_id} {node.type}")
            else:
                lines.append(f"    class {node_id} default")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_raw(self, trace: TraceNode) -> str:
        full_diagram = self.generate(trace)
        lines = full_diagram.split('\n')
        if lines[0] == '```mermaid' and lines[-1] == '```':
            return '\n'.join(lines[1:-1])
        return full_diagram
    
    def generate_sequence_diagram(self, trace: TraceNode) -> str:
        lines = [
            "```mermaid",
            "sequenceDiagram",
            "    participant User",
            "    participant Agent",
            "    participant Decomposer",
            "    participant StateAnalyzer as State Analyzer", 
            "    participant Synthesizer",
            "    participant LLM",
            ""
        ]
        
        description = trace.input_data.get('description', 'Behavior request')
        lines.append(f"    User->>+Agent: {self._escape_mermaid_text(description)}")
        
        self._add_chronological_sequence(trace, lines)
        
        lines.append("    Agent-->>-User: Behavior ready")
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_sequence_raw(self, trace: TraceNode) -> str:
        full_diagram = self.generate_sequence_diagram(trace)
        lines = full_diagram.split('\n')
        if lines[0] == '```mermaid' and lines[-1] == '```':
            return '\n'.join(lines[1:-1])
        return full_diagram
    
    def _add_node(self, node: TraceNode, lines: List[str], depth: int = 0):
        node_id = self._get_node_id(node)
        label = self._get_node_label(node)
        shape = self._get_node_shape(node)
        
        indent = "    " * (depth + 1)
        lines.append(f"{indent}{node_id}{shape[0]}{label}{shape[1]}")
        
        for child in node.children:
            self._add_node(child, lines, depth)
    
    def _add_connections(self, node: TraceNode, lines: List[str], parent_id: Optional[str] = None):
        node_id = self._get_node_id(node)
        
        if parent_id:
            if node.type == "llm_call":
                arrow = "-->"
                label = self._get_connection_label(node)
                if label:
                    lines.append(f"    {parent_id} {arrow}|{label}| {node_id}")
                else:
                    lines.append(f"    {parent_id} {arrow} {node_id}")
            else:
                lines.append(f"    {parent_id} --> {node_id}")
        
        for child in node.children:
            self._add_connections(child, lines, node_id)
    
    def _get_node_id(self, node: TraceNode) -> str:
        if node.id not in self.node_ids:
            self.node_ids[node.id] = f"node{self.node_counter}"
            self.node_counter += 1
        return self.node_ids[node.id]
    
    def _escape_mermaid_text(self, text: str) -> str:
        text = str(text)
        text = text.replace('"', "'")
        text = text.replace("\n", " ")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        return text
    
    def _get_node_label(self, node: TraceNode) -> str:
        label = self._escape_mermaid_text(node.name)
        
        if node.type == "decomposition" and node.output_data:
            components = node.output_data.get("components", [])
            is_simple = node.output_data.get("is_simple", False)
            if is_simple:
                label += f"<br/>Simple → 1 expert"
            else:
                label += f"<br/>Complex → {len(components)} experts"
            
        elif node.type == "synthesis" and node.output_data:
            expert_count = 0
            for child in node.children:
                if child.type == "llm_call" and child.output_data.get("expert_name"):
                    expert_count += 1
            if expert_count == 0:
                expert_count = node.output_data.get("expert_count", node.output_data.get("experts_added", 0))
            label += f"<br/>{expert_count} experts"
            
        elif node.type == "llm_call" and node.llm_call:
            if 'decompose' in node.name.lower() or (node.parent_id and 'decompose' in node.parent_id):
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    is_simple = parsed.get("is_simple", False)
                    components = parsed.get("components", [])
                    if is_simple:
                        label += "<br/>Decompose → Simple"
                    else:
                        label += f"<br/>Decompose → {len(components)} experts"
                else:
                    label += "<br/>Decomposition"
            else:
                if node.input_data.get("description"):
                    input_desc = self._escape_mermaid_text(node.input_data["description"])
                    if len(input_desc) > 30:
                        input_desc = input_desc[:27] + "..."
                    label += f"<br/>{input_desc}"
                
                if node.output_data.get("expert_name"):
                    expert_name = self._escape_mermaid_text(node.output_data['expert_name'])
                    label += f"<br/>→ {expert_name}"
            
                
        elif node.type == "routing" and node.metadata:
            decision = node.metadata.get("routing_decision", "")
            confidence = node.metadata.get("routing_confidence", 0)
            if decision:
                label += f"<br/>→ {decision} ({confidence:.0%})"
                
        elif node.type == "state_analysis" and node.output_data:
            states_count = (
                node.output_data.get("global_states", 0) +
                node.output_data.get("particle_states", 0) +
                node.output_data.get("species_states", 0)
            )
            temporal_count = node.output_data.get("temporal_updates", 0)
            if states_count > 0:
                label += f"<br/>{states_count} states"
            if temporal_count > 0:
                label += f"<br/>{temporal_count} temporal updates"
        
        if node.status == "error" and node.error:
            error_msg = node.error[:30] + "..." if len(node.error) > 30 else node.error
            label += f"<br/>❌ {error_msg}"
        
        return label
    
    def _get_node_shape(self, node: TraceNode) -> tuple:
        shapes = {
            "synthesis": ("[", "]"),  # Rectangle
            "decomposition": ("([", "])"),  # Stadium
            "llm_call": ("{{", "}}"),  # Hexagon
            "routing": ("{", "}"),  # Rhombus
            "parsing": ("[[", "]]"),  # Subroutine
            "drawing": ("[(", ")]"),  # Cylindrical
            "state_analysis": ("([", "])"),  # Stadium shape for state_analysis
            "temporal_update": ("((", "))"),  # Double circle
        }
        return shapes.get(node.type, ("(", ")"))
    
    def _get_connection_label(self, node: TraceNode) -> Optional[str]:
        if node.type == "llm_call" and node.llm_call:
            if node.llm_call.total_tokens:
                return f"{node.llm_call.total_tokens}t"
        return None
    
    def _format_duration(self, ms: Optional[float]) -> str:
        if ms is None:
            return "N/A"
        
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms/1000:.1f}s"
        else:
            return f"{ms/60000:.1f}m"
    
    def _collect_all_nodes(self, root: TraceNode) -> Dict[str, Dict]:
        nodes = {}
        
        def collect(node: TraceNode):
            node_id = self._get_node_id(node)
            nodes[node_id] = {"node": node}
            for child in node.children:
                collect(child)
        
        collect(root)
        return nodes
    
    def _add_chronological_sequence(self, node: TraceNode, lines: List[str]):
        def collect_llm_calls_chronologically(node: TraceNode, llm_calls: List[TraceNode]):
            if node.type == "llm_call" and node.llm_call:
                llm_calls.append(node)
            for child in node.children:
                collect_llm_calls_chronologically(child, llm_calls)
        
        llm_calls = []
        collect_llm_calls_chronologically(node, llm_calls)
        
        for llm_node in llm_calls:
            
            if 'decompose' in llm_node.name.lower():
                lines.append("    Agent->>+Decomposer: Analyze behavior complexity")
                lines.append("    Decomposer->>+LLM: Decompose request")
                
                if llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        is_simple = parsed.get("is_simple", False)
                        components = parsed.get("components", [])
                        if is_simple:
                            lines.append("    LLM-->>-Decomposer: Simple behavior")
                            lines.append("    Decomposer-->>-Agent: Use single expert")
                        else:
                            lines.append(f"    LLM-->>-Decomposer: {len(components)} components")
                            lines.append(f"    Decomposer-->>-Agent: {len(components)} experts needed")
                
            elif 'state_analysis' in llm_node.name.lower():
                lines.append("    Agent->>+StateAnalyzer: Analyze state requirements")
                lines.append("    StateAnalyzer->>+LLM: Check custom states")
                
                if llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        needs_states = parsed.get("needs_states", False)
                        if needs_states:
                            global_states = parsed.get("global_states", [])
                            particle_states = parsed.get("particle_states", [])
                            species_states = parsed.get("species_states", [])
                            total_states = len(global_states) + len(particle_states) + len(species_states)
                            lines.append(f"    LLM-->>-StateAnalyzer: {total_states} states needed")
                            lines.append("    StateAnalyzer-->>-Agent: States configured")
                        else:
                            lines.append("    LLM-->>-StateAnalyzer: No custom states needed")
                            lines.append("    StateAnalyzer-->>-Agent: Using default states")
                
            else:
                if llm_node.input_data.get("description"):
                    desc = llm_node.input_data["description"]
                    if len(desc) > 40:
                        desc = desc[:37] + "..."
                    
                    lines.append("    Agent->>+Synthesizer: Generate expert")
                    lines.append(f"    Synthesizer->>+LLM: {self._escape_mermaid_text(desc)}")
                    
                    if llm_node.output_data.get("expert_name"):
                        expert_name = llm_node.output_data["expert_name"]
                        lines.append(f"    LLM-->>-Synthesizer: {self._escape_mermaid_text(expert_name)}")
                        lines.append("    Synthesizer-->>-Agent: Expert compiled")
                    else:
                        lines.append("    LLM-->>-Synthesizer: Expert code")
                        lines.append("    Synthesizer-->>-Agent: Expert ready")


def generate_mermaid_diagram(trace: TraceNode, diagram_type: str = "flow") -> str:
    generator = MermaidDiagramGenerator()
    
    if diagram_type == "sequence":
        return generator.generate_sequence_diagram(trace)
    else:
        return generator.generate(trace)