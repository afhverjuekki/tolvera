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
            "    classDef color_resolution fill:#fff3e0,stroke:#ff5722,stroke-width:2px;",
            "    classDef temporal_update fill:#fffde7,stroke:#f9a825,stroke-width:2px;",
            "    classDef analysis fill:#f0e6ff,stroke:#9c27b0,stroke-width:2px;",
            "    classDef implementation fill:#ede7f6,stroke:#7c4dff,stroke-width:2px;",
            "    classDef refinement fill:#ede7f6,stroke:#7c4dff,stroke-width:2px;",
            "    classDef error_correction fill:#ffebee,stroke:#f44336,stroke-width:3px;",
            "    classDef sketch_repair fill:#fff3e0,stroke:#ff9800,stroke-width:2px;",
            "    %% Expert type styles",
            "    classDef force_expert fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;",
            "    classDef interaction_expert fill:#e0f2f1,stroke:#00695c,stroke-width:2px;",
            "    classDef temporal_update_expert fill:#fff8e1,stroke:#ef6c00,stroke-width:2px;",
            "    classDef state_update_expert fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;",
            "    classDef utility_expert fill:#fce4ec,stroke:#ad1457,stroke-width:2px;",
            "    classDef visual_expert fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;",
            "    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:3px;",
            "    classDef success fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;",
            "    classDef demo fill:#f5f5f5,stroke:#666666,stroke-width:2px;",
            "    classDef default fill:#f5f5f5,stroke:#666666,stroke-width:2px;",
            "",
            "    %% Apply styles",
        ])
        
        known_types = {
            'synthesis', 'decomposition', 'llm_call', 'routing', 
            'parsing', 'drawing', 'state_analysis', 'color_resolution', 'temporal_update',
            'analysis', 'implementation', 'refinement', 'error_correction', 'behavior_modification', 'sketch_repair',
            'demo', 'error', 'success'
        }
        
        expert_types = {
            'force', 'interaction', 'temporal_update', 'state_update', 'utility', 'visual'
        }
        
        for node_id, node_data in self._collect_all_nodes(trace).items():
            node = node_data['node']
            if node.status == "error":
                lines.append(f"    class {node_id} error")
            elif node.status == "success" and node.type == "synthesis":
                lines.append(f"    class {node_id} success")
            elif node.type == "synthesis":
                # Check for expert type in synthesis nodes
                expert_type = node.metadata.get("expert_type") or node.input_data.get("expert_type")
                if expert_type and expert_type in expert_types:
                    lines.append(f"    class {node_id} {expert_type}_expert")
                else:
                    lines.append(f"    class {node_id} synthesis")
            elif node.type == "analysis":
                lines.append(f"    class {node_id} analysis")
            elif node.type == "implementation":
                lines.append(f"    class {node_id} implementation")
            elif node.type == "refinement":
                # Check for error correction refinement
                if "error_correction" in node.name:
                    lines.append(f"    class {node_id} error_correction")
                else:
                    lines.append(f"    class {node_id} refinement")
            elif node.type == "sketch_repair":
                lines.append(f"    class {node_id} sketch_repair")
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
            "    participant Refiner",
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
            component_count = len(components)
            label += f"<br/>{component_count} expert{'s' if component_count != 1 else ''}"
            
            # Show expert types breakdown
            expert_types = {}
            for comp in components:
                expert_type = comp.get("expert_type", "unknown")
                expert_types[expert_type] = expert_types.get(expert_type, 0) + 1
            
            if expert_types:
                type_summary = []
                for etype, count in expert_types.items():
                    type_summary.append(f"{count}×{etype}")
                if len(type_summary) <= 3:  # Show details if not too many
                    label += f"<br/>{', '.join(type_summary)}"
            
        elif node.type == "synthesis" and node.output_data:
            expert_count = 0
            for child in node.children:
                if child.type == "llm_call" and child.output_data.get("expert_name"):
                    expert_count += 1
            if expert_count == 0:
                expert_count = node.output_data.get("expert_count", node.output_data.get("experts_added", 0))
            label += f"<br/>{expert_count} experts"
            
            # Show expert type if available
            expert_type = node.metadata.get("expert_type") or node.input_data.get("expert_type")
            if expert_type:
                label += f"<br/>{expert_type}"
            
        elif node.type == "analysis":
            # Two-stage refinement: Stage 1 analysis
            if node.output_data:
                plan_length = len(node.output_data.get('implementation_plan', ''))
                errors_found = node.output_data.get('errors_found', '')
                label += f"<br/>Stage 1: Analysis"
                if plan_length > 0:
                    label += f"<br/>{plan_length} char plan"
                if errors_found:
                    label += f"<br/>Errors detected"
            else:
                label += "<br/>Stage 1: Analysis"
                
        elif node.type == "implementation":
            # Two-stage refinement: Stage 2 implementation
            if node.output_data:
                changes = node.output_data.get('changes_summary', '')
                code_length = len(node.output_data.get('refined_code', ''))
                label += f"<br/>Stage 2: Implementation"
                if changes:
                    changes_preview = changes[:30] + "..." if len(changes) > 30 else changes
                    label += f"<br/>{self._escape_mermaid_text(changes_preview)}"
                if code_length > 0:
                    label += f"<br/>{code_length} chars"
            else:
                label += "<br/>Stage 2: Implementation"
                
        elif node.type == "llm_call" and node.llm_call:
            if 'analyze_sketch' in node.name.lower():
                # Stage 1 LLM call
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    plan_length = len(parsed.get('implementation_plan', ''))
                    errors_found = parsed.get('errors_found', '')
                    label += f"<br/>Analysis LLM Call"
                    if plan_length > 0:
                        label += f"<br/>→ {plan_length} char plan"
                    if errors_found:
                        label += f"<br/>→ Errors found"
                else:
                    label += "<br/>Analysis LLM Call"
            elif 'implement_refinement' in node.name.lower():
                # Stage 2 LLM call
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    changes = parsed.get('changes_summary', '')
                    if changes:
                        changes_preview = changes[:25] + "..." if len(changes) > 25 else changes
                        label += f"<br/>Implementation LLM Call"
                        label += f"<br/>→ {self._escape_mermaid_text(changes_preview)}"
                    else:
                        label += f"<br/>Implementation LLM Call"
                else:
                    label += "<br/>Implementation LLM Call"
            elif 'decompose' in node.name.lower() or (node.parent_id and 'decompose' in node.parent_id):
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    components = parsed.get("components", [])
                    component_count = len(components)
                    label += f"<br/>Decompose → {component_count} expert{'s' if component_count != 1 else ''}"
                else:
                    label += "<br/>Decomposition"
            elif 'color_resolution' in node.name.lower():
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    color_name = parsed.get("color_name", "unknown")
                    rgba_list = parsed.get("rgba_list", [])
                    if rgba_list and len(rgba_list) >= 3:
                        r, g, b = rgba_list[:3]
                        label += f"<br/>'{color_name}' → rgb({r:.2f},{g:.2f},{b:.2f})"
                    else:
                        label += f"<br/>Resolve '{color_name}'"
                else:
                    color_name = node.metadata.get("color_name", "unknown")
                    label += f"<br/>Resolve '{color_name}'"
            elif 'state_analysis' in node.name.lower():
                parsed = node.llm_call.parsed_response
                if parsed and isinstance(parsed, dict):
                    global_states = parsed.get("global_states", [])
                    particle_states = parsed.get("particle_states", [])
                    species_states = parsed.get("species_states", [])
                    
                    global_count = len(global_states) if isinstance(global_states, list) else global_states if isinstance(global_states, int) else 0
                    particle_count = len(particle_states) if isinstance(particle_states, list) else particle_states if isinstance(particle_states, int) else 0
                    species_count = len(species_states) if isinstance(species_states, list) else species_states if isinstance(species_states, int) else 0
                    
                    total_states = global_count + particle_count + species_count
                    label += f"<br/>Found {total_states} states"
                    
                    # Show breakdown if states found
                    if total_states > 0:
                        parts = []
                        if global_count > 0:
                            parts.append(f"{global_count}G")
                        if particle_count > 0:
                            parts.append(f"{particle_count}P")
                        if species_count > 0:
                            parts.append(f"{species_count}S")
                        label += f"<br/>{'+'.join(parts)}"
                else:
                    label += "<br/>Analyze states"
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
                label += f"<br/>→ {decision} {confidence:.0%}"
                
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
                
        elif node.type == "refinement" and node.output_data:
            changes = node.output_data.get("changes_made", "")
            if changes and len(changes) > 30:
                changes = changes[:27] + "..."
            if changes:
                label += f"<br/>→ {self._escape_mermaid_text(changes)}"
            if node.metadata.get("has_error"):
                label += f"<br/>⚡ Error fix"
                
        elif node.type == "sketch_repair" and node.output_data:
            if node.output_data.get("repair_success"):
                changes = node.output_data.get("changes_made", "")
                if changes and len(changes) > 30:
                    changes = changes[:27] + "..."
                if changes:
                    label += f"<br/>✅ {self._escape_mermaid_text(changes)}"
                lines_changed = node.output_data.get("code_lines_changed", 0)
                if lines_changed:
                    label += f"<br/>{lines_changed} lines modified"
            else:
                error = node.output_data.get("error", "")
                if error and len(error) > 30:
                    error = error[:27] + "..."
                if error:
                    label += f"<br/>❌ {self._escape_mermaid_text(error)}"
        
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
            "analysis": ("((", "))"),  # Double circle for analysis
            "implementation": ("[/", "\\]"),  # Parallelogram for implementation
            "refinement": ("[/", "\\]"),  # Parallelogram
            "sketch_repair": ("{{", "}}"),  # Hexagon
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
    
    def _find_node_by_id(self, root: TraceNode, target_id: str) -> Optional[TraceNode]:
        """Find a node by its ID in the trace tree."""
        if root.id == target_id:
            return root
        for child in root.children:
            found = self._find_node_by_id(child, target_id)
            if found:
                return found
        return None
    
    def _add_chronological_sequence(self, node: TraceNode, lines: List[str]):
        def collect_all_events_chronologically(node: TraceNode, events: List[TraceNode]):
            # Collect LLM calls and sketch repair events
            if node.type == "llm_call" and node.llm_call:
                events.append(node)
            elif node.type == "sketch_repair":
                events.append(node)
            for child in node.children:
                collect_all_events_chronologically(child, events)
        
        events = []
        collect_all_events_chronologically(node, events)
        
        for event_node in events:
            
            # Handle sketch repair events separately
            if event_node.type == "sketch_repair":
                lines.append("    User->>+Agent: Repair sketch errors")
                if event_node.output_data:
                    if event_node.output_data.get("repair_success"):
                        changes = event_node.output_data.get("changes_made", "")
                        if changes:
                            if len(changes) > 40:
                                changes = changes[:37] + "..."
                            lines.append(f"    Agent-->>-User: {self._escape_mermaid_text(changes)}")
                        else:
                            lines.append("    Agent-->>-User: Repair completed")
                    else:
                        error = event_node.output_data.get("error", "")
                        if error:
                            if len(error) > 40:
                                error = error[:37] + "..."
                            lines.append(f"    Agent-->>-User: Repair failed: {self._escape_mermaid_text(error)}")
                        else:
                            lines.append("    Agent-->>-User: Repair failed")
                continue
            
            # Handle LLM calls
            llm_node = event_node
            
            if 'analyze_sketch' in llm_node.name.lower():
                # Stage 1: Analysis
                lines.append("    User->>+Refiner: Request two-stage refinement")
                lines.append("    Refiner->>+LLM: Stage 1: Analyze sketch")
                
                if llm_node.llm_call and llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        plan_length = len(parsed.get('implementation_plan', ''))
                        errors_found = parsed.get('errors_found', '')
                        if plan_length > 0:
                            lines.append(f"    LLM-->>-Refiner: Analysis complete ({plan_length} chars)")
                        else:
                            lines.append("    LLM-->>-Refiner: Analysis complete")
                        if errors_found:
                            lines.append("    Note over Refiner: Errors detected in sketch")
                    else:
                        lines.append("    LLM-->>-Refiner: Analysis complete")
                else:
                    lines.append("    LLM-->>-Refiner: Analysis complete")
                    
            elif 'implement_refinement' in llm_node.name.lower():
                # Stage 2: Implementation
                lines.append("    Refiner->>+LLM: Stage 2: Implement plan")
                
                if llm_node.llm_call and llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        changes = parsed.get('changes_summary', '')
                        if changes:
                            changes_preview = changes[:40] + "..." if len(changes) > 40 else changes
                            lines.append(f"    LLM-->>-Refiner: {self._escape_mermaid_text(changes_preview)}")
                        else:
                            lines.append("    LLM-->>-Refiner: Implementation complete")
                        lines.append("    Refiner-->>-User: Refined sketch ready")
                    else:
                        lines.append("    LLM-->>-Refiner: Implementation complete")
                        lines.append("    Refiner-->>-User: Refined sketch ready")
                else:
                    lines.append("    LLM-->>-Refiner: Implementation complete")
                    lines.append("    Refiner-->>-User: Refined sketch ready")
                    
            elif 'decompose' in llm_node.name.lower():
                lines.append("    Agent->>+Decomposer: Analyze behavior complexity")
                lines.append("    Decomposer->>+LLM: Decompose request")
                
                if llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        components = parsed.get("components", [])
                        component_count = len(components)
                        if component_count == 1:
                            lines.append("    LLM-->>-Decomposer: Single expert")
                            lines.append("    Decomposer-->>-Agent: Use 1 expert")
                        else:
                            lines.append(f"    LLM-->>-Decomposer: {component_count} components")
                            lines.append(f"    Decomposer-->>-Agent: {component_count} experts needed")
                
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
            
            elif 'refinement' in llm_node.name.lower():
                # Handle refinement nodes
                lines.append("    User->>+Refiner: Request refinement")
                
                # Check if this is an error correction or behavior modification
                if llm_node.parent_id:
                    parent_node = self._find_node_by_id(node, llm_node.parent_id)
                    if parent_node and parent_node.metadata.get("has_error"):
                        lines.append("    Refiner->>+LLM: Fix error")
                    else:
                        lines.append("    Refiner->>+LLM: Modify behavior")
                else:
                    lines.append("    Refiner->>+LLM: Refine sketch")
                
                if llm_node.llm_call and llm_node.llm_call.parsed_response:
                    parsed = llm_node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        changes = parsed.get("changes_made", "")
                        if changes:
                            if len(changes) > 40:
                                changes = changes[:37] + "..."
                            lines.append(f"    LLM-->>-Refiner: {self._escape_mermaid_text(changes)}")
                        else:
                            lines.append("    LLM-->>-Refiner: Refinement complete")
                        lines.append("    Refiner-->>-User: Sketch updated")
                
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