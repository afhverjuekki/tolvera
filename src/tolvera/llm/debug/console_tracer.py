import sys
from typing import Optional, Dict, Set
from .tracing import TraceNode, TraceCollector, get_collector

class ConsoleTracer:
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    ICONS = {
        "synthesis": "🎯",
        "decomposition": "🧩",
        "llm_call": "🤖",
        "routing": "🔍",
        "parsing": "📝",
        "drawing": "🎨",
        "state_analysis": "🧬",
        "temporal_update": "⏰",
        "error": "❌",
        "success": "✅",
        "running": "⏳",
    }
    
    def __init__(self, collector: Optional[TraceCollector] = None, colored: bool = True):
        self.collector = collector or get_collector()
        self.colored = colored and sys.stdout.isatty()
        self.indent_size = 2
        self.show_timing = True
        self.show_tokens = True
        self.expanded_nodes: Set[str] = set()  # Nodes to show details for
        
        # Hook into collector callbacks
        self.collector.on_node_start = self._on_node_start
        self.collector.on_node_complete = self._on_node_complete
    
    def _color(self, text: str, color: str) -> str:
        if not self.colored:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _get_icon(self, node: TraceNode) -> str:
        if node.status == "error":
            return self.ICONS.get("error", "")
        elif node.status == "success":
            return self.ICONS.get("success", "")
        elif node.status == "running":
            return self.ICONS.get("running", "")
        return self.ICONS.get(node.type, "📍")
    
    def _format_duration(self, ms: Optional[float]) -> str:
        if ms is None:
            return ""
        
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms/1000:.1f}s"
        else:
            return f"{ms/60000:.1f}m"
    
    def _get_indent(self, depth: int) -> str:
        if depth == 0:
            return ""
        prefix = "│  " * (depth - 1)
        return prefix + "├─ "
    
    def _on_node_start(self, node: TraceNode):
        depth = self._get_node_depth(node)
        indent = self._get_indent(depth)
        
        icon = self._get_icon(node)
        name = self._color(node.name, "bold")
        if node.type == "decomposition":
            name = self._color(f"Decomposing: {node.name}", "cyan")
        elif node.type == "synthesis":
            name = self._color(f"Synthesizing: {node.name}", "blue")
        elif node.type == "llm_call":
            model = node.metadata.get("model") or node.input_data.get("model", "unknown")
            if 'decompose' in node.name.lower():
                name = self._color(f"Decomposition LLM Call ({model})", "cyan")
            elif 'state_analysis' in node.name.lower():
                name = self._color(f"State Analysis LLM Call ({model})", "green")
            else:
                name = self._color(f"LLM Call ({model})", "magenta")
        elif node.type == "routing":
            name = self._color(f"Routing: {node.name}", "yellow")
        elif node.type == "state_analysis":
            name = self._color(f"Analyzing States: {node.name}", "green")
        elif node.type == "temporal_update":
            name = self._color(f"Temporal Update: {node.name}", "cyan")
        
        print(f"{indent}{icon} {name}")
        
        if node.id in self.expanded_nodes and node.input_data:
            self._print_details(node, depth + 1)
    
    def _on_node_complete(self, node: TraceNode):
        depth = self._get_node_depth(node)
        indent = self._get_indent(depth) + "  "
        
        if node.status == "error":
            status = self._color(f"✗ Failed: {node.error}", "red")
            print(f"{indent}{status}")
        elif self.show_timing and node.duration_ms:
            duration = self._format_duration(node.duration_ms)
            timing = self._color(f"[{duration}]", "dim")
            
            extra = ""
            if node.type == "llm_call" and node.llm_call:
                if self.show_tokens and node.llm_call.total_tokens:
                    extra = f" ({node.llm_call.total_tokens} tokens)"
                elif 'state_analysis' in node.name.lower() and node.llm_call.parsed_response:
                    parsed = node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        global_count = len(parsed.get('global_states', []))
                        particle_count = len(parsed.get('particle_states', []))
                        species_count = len(parsed.get('species_states', []))
                        temporal_count = len(parsed.get('temporal_updates', []))
                        total_states = global_count + particle_count + species_count
                        if total_states > 0:
                            extra = f" → {total_states} states, {temporal_count} temporal"
                        else:
                            extra = " → no states needed"
            elif node.type == "decomposition" and "components" in node.output_data:
                count = len(node.output_data.get("components", []))
                extra = f" → {count} components"
            
            print(f"{indent}{timing}{extra}")
        
        if node.type == "decomposition" and node.output_data:
            self._show_decomposition_output(node, depth + 1)
        elif node.type == "routing" and node.metadata.get("routing_decision"):
            self._show_routing_output(node, depth + 1)
        elif node.type == "state_analysis" and node.output_data:
            self._show_state_analysis_output(node, depth + 1)
    
    def _show_decomposition_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        output = node.output_data
        
        if "interpretation" in output:
            interp = self._color(f'"{output["interpretation"]}"', "dim")
            print(f"{indent}Interpretation: {interp}")
        
        # Show component count instead of complexity classification
        components = output.get('components', [])
        component_count = len(components)
        complexity_color = "green" if component_count == 1 else "yellow"
        complexity_text = f"{component_count} expert{'s' if component_count != 1 else ''}"
        print(f"{indent}Components: {self._color(complexity_text, complexity_color)}")
        
        if "components" in output:
            print(f"{indent}Components:")
            for comp in output["components"]:
                expert_name = comp.get("expert_name", "unknown")
                expert_type = comp.get("expert_type", "")
                priority = comp.get("priority", 1.0)
                desc = comp.get("description", "")
                implementation = comp.get("implementation", "")
                
                name_str = self._color(expert_name, "bold")
                type_str = self._color(f"[{expert_type}]", "dim")
                weight_str = self._color(f"({priority})", "dim")
                
                print(f"{indent}  • {name_str} {type_str} {weight_str}")
                print(f"{indent}    {desc}")
                if implementation:
                    impl_preview = implementation.split('\n')[0][:60]
                    if len(implementation) > 60:
                        impl_preview += "..."
                    print(f"{indent}    → {self._color(impl_preview, 'dim')}")
        
        if "context" in output and output["context"]:
            context = output["context"]
            constraints = context.get("constraints", [])
            relationships = context.get("relationships", [])
            
            if constraints or relationships:
                print(f"{indent}Context Threading:")
                for constraint in constraints:
                    print(f"{indent}  - {self._color(constraint, 'yellow')}")
                for rel in relationships:
                    print(f"{indent}  - {self._color(rel, 'yellow')}")
    
    def _show_routing_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        decision = node.metadata.get("routing_decision", "")
        reason = node.metadata.get("routing_reason", "")
        confidence = node.metadata.get("routing_confidence", 0)
        
        decision_str = self._color(decision, "yellow")
        confidence_str = self._color(f"({confidence:.0%})", "dim")
        print(f"{indent}→ {decision_str} {confidence_str}")
        
        if reason:
            print(f"{indent}  Reason: {self._color(reason, 'dim')}")
    
    def _show_state_analysis_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        output = node.output_data
        
        if node.llm_call and node.llm_call.parsed_response:
            parsed = node.llm_call.parsed_response
            print(f"{indent}{self._color('LLM Response:', 'bold')}")
            
            if isinstance(parsed, dict):
                needs_states = parsed.get('needs_states', False)
                print(f"{indent}  Needs states: {self._color(str(needs_states), 'yellow')}")
                
                for category in ['global_states', 'particle_states', 'species_states']:
                    states = parsed.get(category, [])
                    if states:
                        category_name = category.replace('_states', '').capitalize()
                        print(f"{indent}  {self._color(f'{category_name} states ({len(states)}):', 'cyan')}")
                        
                        for state in states:
                            if isinstance(state, dict):
                                name = state.get('name', 'unnamed')
                                state_type = state.get('type', 'unknown')
                                min_val = state.get('min', 'N/A')
                                max_val = state.get('max', 'N/A')
                                desc = state.get('description', '')
                                initial = state.get('initial', 'auto')
                                
                                name_str = self._color(name, 'bold')
                                type_str = self._color(f'({state_type})', 'dim')
                                range_str = self._color(f'[{min_val}-{max_val}]', 'dim')
                                initial_str = self._color(f'init:{initial}', 'dim')
                                
                                print(f"{indent}    • {name_str} {type_str} {range_str} {initial_str}")
                                if desc:
                                    print(f"{indent}      {self._color(desc, 'dim')}")
                            else:
                                print(f"{indent}    • {self._color(str(state), 'yellow')}")
                
                temporal_updates = parsed.get('temporal_updates', [])
                if temporal_updates:
                    print(f"{indent}  {self._color(f'Temporal updates ({len(temporal_updates)}):', 'cyan')}")
                    for update in temporal_updates:
                        if isinstance(update, dict):
                            state_name = update.get('state_name', 'unknown')
                            expression = update.get('update_expression', 'unknown')
                            desc = update.get('description', '')
                            
                            update_str = self._color(state_name, 'yellow')
                            expr_str = self._color(expression, 'dim')
                            print(f"{indent}    • {update_str} = {expr_str}")
                            if desc:
                                print(f"{indent}      {self._color(desc, 'dim')}")
                        else:
                            print(f"{indent}    • {self._color(str(update), 'yellow')}")
            else:
                print(f"{indent}  {self._color(str(parsed), 'yellow')}")
        elif output.get("needs_states"):
            states_summary = []
            if output.get("global_states", 0) > 0:
                states_summary.append(f"Global: {output['global_states']}")
            if output.get("particle_states", 0) > 0:
                states_summary.append(f"Particle: {output['particle_states']}")
            if output.get("species_states", 0) > 0:
                states_summary.append(f"Species: {output['species_states']}")
            if output.get("temporal_updates", 0) > 0:
                states_summary.append(f"Temporal: {output['temporal_updates']}")
            
            if states_summary:
                print(f"{indent}{self._color('States needed (summary):', 'yellow')} {', '.join(states_summary)}")
                print(f"{indent}{self._color('Note: Full state details not available in trace', 'dim')}")
            
            temporal_details = output.get("temporal_update_details", [])
            if temporal_details:
                print(f"{indent}Temporal updates:")
                for update in temporal_details:
                    update_str = self._color(f"{update['state']}", "yellow")
                    expr_str = self._color(f"{update['expression']}", "dim")
                    print(f"{indent}  • {update_str} = {expr_str}")
                    if update.get("description"):
                        print(f"{indent}    {self._color(update['description'], 'dim')}")
        else:
            print(f"{indent}{self._color('No custom states needed', 'green')}")
    
    def _print_details(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        
        if node.llm_call and self.collector.capture_llm_content:
            llm = node.llm_call
            print(f"{indent}Model: {llm.model}")
            
            if llm.system_prompt:
                print(f"{indent}System prompt: ({len(llm.system_prompt)} chars)")
            
            if llm.user_prompt:
                print(f"{indent}User prompt: ({len(llm.user_prompt)} chars)")
                first_line = llm.user_prompt.split('\n')[0][:80]
                print(f"{indent}  {self._color(first_line + '...', 'dim')}")
    
    def _get_node_depth(self, node: TraceNode) -> int:
        depth = 0
        current = node
        
        while current.parent_id:
            depth += 1
            parent = self._find_parent(current)
            if not parent:
                break
            current = parent
        
        return depth
    
    def _find_parent(self, node: TraceNode) -> Optional[TraceNode]:
        if not node.parent_id:
            return None
        
        current_trace = self.collector.get_current_trace()
        if not current_trace:
            return None
        
        return self._find_node_by_id(current_trace, node.parent_id)
    
    def _find_node_by_id(self, root: TraceNode, node_id: str) -> Optional[TraceNode]:
        if root.id == node_id:
            return root
        
        for child in root.children:
            found = self._find_node_by_id(child, node_id)
            if found:
                return found
        
        return None
    
    def print_summary(self, trace: TraceNode):
        print("\n" + "=" * 60)
        print(self._color("Trace Summary", "bold"))
        print("=" * 60)
        
        print(f"Input: {trace.input_data.get('description', 'N/A')}")
        print(f"Total duration: {self._format_duration(trace.duration_ms)}")
        print(f"Status: {trace.status}")
        
        counts = self._count_nodes_by_type(trace)
        print("\nNodes processed:")
        for node_type, count in counts.items():
            icon = self.ICONS.get(node_type, "")
            print(f"  {icon} {node_type}: {count}")
        
        errors = self._collect_errors(trace)
        if errors:
            print(f"\n{self._color('Errors:', 'red')}")
            for error in errors:
                print(f"  • {error}")
    
    def _count_nodes_by_type(self, root: TraceNode) -> Dict[str, int]:
        counts = {}
        
        def count(node):
            counts[node.type] = counts.get(node.type, 0) + 1
            for child in node.children:
                count(child)
        
        count(root)
        return counts
    
    def _collect_errors(self, root: TraceNode) -> list:
        errors = []
        
        def collect(node):
            if node.status == "error" and node.error:
                errors.append(f"{node.name}: {node.error}")
            for child in node.children:
                collect(child)
        
        collect(root)
        return errors


def enable_console_tracing(colored: bool = True):
    console_tracer = ConsoleTracer(colored=colored)
    return console_tracer