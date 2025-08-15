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
        "color_resolution": "🌈",
        "temporal_update": "⏰",
        "error": "❌",
        "success": "✅",
        "running": "⏳",
        # Refinement icons
        "analysis": "🔍",
        "implementation": "⚙️",
        "refinement": "🔧",
        "error_correction": "⚡",
        "behavior_modification": "✨",
        "sketch_repair": "🩹",
        # Expert type icons
        "force": "⚡",
        "interaction": "🔗", 
        "state_update": "🔄",
        "utility": "🛠️",
        "visual": "👁️",
        "temporal": "⏰",
        "configuration": "⚙️",
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
            # Check if we can extract expert type from the node metadata
            expert_type = node.metadata.get("expert_type") or node.input_data.get("expert_type")
            if expert_type:
                expert_icon = self.ICONS.get(expert_type, "🎯")
                type_color = {
                    "force": "blue",
                    "interaction": "cyan", 
                    "temporal_update": "yellow",
                    "state_update": "green",
                    "utility": "magenta",
                    "visual": "red"
                }.get(expert_type, "blue")
                name = self._color(f"Synthesizing {expert_type}: {node.name}", type_color)
                icon = expert_icon
            else:
                name = self._color(f"Synthesizing: {node.name}", "blue")
        elif node.type == "llm_call":
            model = node.metadata.get("model") or node.input_data.get("model", "unknown")
            if 'decompose' in node.name.lower():
                name = self._color(f"Decomposition LLM Call ({model})", "cyan")
            elif 'state_analysis' in node.name.lower():
                # Check if this is a component-level state analysis
                parent = self._find_parent(node)
                if parent and parent.name and "component:" in parent.name.lower():
                    component_name = parent.name.split("component:")[-1].strip()
                    name = self._color(f"Component State Analysis ({model}) for: {component_name}", "green")
                else:
                    name = self._color(f"State Analysis LLM Call ({model})", "green")
            elif 'color_resolution' in node.name.lower():
                color_name = node.metadata.get("color_name", "unknown")
                name = self._color(f"Color Resolution LLM Call ({model}) → '{color_name}'", "yellow")
            elif 'analyze_sketch' in node.name.lower():
                name = self._color(f"Stage 1 Analysis LLM Call ({model})", "magenta")
            elif 'implement_refinement' in node.name.lower():
                name = self._color(f"Stage 2 Implementation LLM Call ({model})", "blue")
            else:
                name = self._color(f"LLM Call ({model})", "magenta")
        elif node.type == "routing":
            name = self._color(f"Routing: {node.name}", "yellow")
        elif node.type == "state_analysis":
            name = self._color(f"Analyzing States: {node.name}", "green")
        elif node.type == "color_resolution":
            color_name = node.metadata.get("color_name", node.input_data.get("color_name", "unknown"))
            name = self._color(f"Resolving Color: '{color_name}'", "yellow")
        elif node.type == "temporal_update":
            name = self._color(f"Temporal Update: {node.name}", "cyan")
        elif node.type == "analysis":
            icon = self.ICONS.get("analysis", "🔍")
            name = self._color(f"Stage 1 Analysis: {node.name}", "magenta")
        elif node.type == "implementation":
            icon = self.ICONS.get("implementation", "⚙️")
            name = self._color(f"Stage 2 Implementation: {node.name}", "blue")
        elif node.type == "refinement":
            # Check refinement subtype
            if "error_correction" in node.name:
                icon = self.ICONS.get("error_correction", "⚡")
                name = self._color(f"Error Correction: {node.metadata.get('refinement_request', 'Fixing errors')}", "red")
            elif "behavior_modification" in node.name:
                icon = self.ICONS.get("behavior_modification", "✨")
                name = self._color(f"Refining: {node.metadata.get('refinement_request', 'Modifying behavior')}", "magenta")
            else:
                name = self._color(f"Refinement: {node.name}", "magenta")
        elif node.type == "sketch_repair":
            icon = self.ICONS.get("sketch_repair", "🩹")
            if node.name == "sketch_repair_initiated":
                name = self._color(f"Sketch Repair: Auto-fixing execution errors", "yellow")
            else:
                name = self._color(f"Sketch Repair: {node.name}", "yellow")
        
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
                elif 'color_resolution' in node.name.lower() and node.llm_call.parsed_response:
                    parsed = node.llm_call.parsed_response
                    if isinstance(parsed, dict) and 'rgba_list' in parsed:
                        rgba = parsed['rgba_list']
                        if len(rgba) >= 3:
                            r, g, b = rgba[:3]
                            extra = f" → rgb({r:.2f}, {g:.2f}, {b:.2f})"
                        else:
                            extra = f" → {parsed.get('color_name', 'unknown')}"
                elif 'analyze_sketch' in node.name.lower() and node.llm_call.parsed_response:
                    parsed = node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        plan_length = len(parsed.get('implementation_plan', ''))
                        errors_found = parsed.get('errors_found', '')
                        if plan_length > 0:
                            extra = f" → {plan_length} char plan"
                            if errors_found:
                                extra += " (errors found)"
                        else:
                            extra = " → analysis complete"
                elif 'implement_refinement' in node.name.lower() and node.llm_call.parsed_response:
                    parsed = node.llm_call.parsed_response
                    if isinstance(parsed, dict):
                        changes = parsed.get('changes_summary', '')
                        if changes:
                            changes_preview = changes[:30] + "..." if len(changes) > 30 else changes
                            extra = f" → {changes_preview}"
                        else:
                            extra = " → implementation complete"
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
        elif node.type == "color_resolution" and node.output_data:
            self._show_color_resolution_output(node, depth + 1)
        elif node.type == "analysis" and (node.output_data or (node.llm_call and node.llm_call.parsed_response)):
            self._show_analysis_output(node, depth + 1)
        elif node.type == "implementation" and (node.output_data or (node.llm_call and node.llm_call.parsed_response)):
            self._show_implementation_output(node, depth + 1)
        elif node.type == "refinement" and node.output_data:
            self._show_refinement_output(node, depth + 1)
        elif node.type == "sketch_repair" and node.output_data:
            self._show_sketch_repair_output(node, depth + 1)
    
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
                # Color-code expert types for better visibility
                type_color = {
                    "force": "blue",
                    "interaction": "cyan", 
                    "temporal_update": "yellow",
                    "state_update": "green",
                    "utility": "magenta",
                    "visual": "red"
                }.get(expert_type, "dim")
                
                # Get expert type icon
                expert_icon = self.ICONS.get(expert_type, "📍")
                type_str = self._color(f"[{expert_type}]", type_color)
                weight_str = self._color(f"({priority})", "dim")
                
                print(f"{indent}  {expert_icon} {name_str} {type_str} {weight_str}")
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
        
        # Check if this is component-level state analysis
        description = node.input_data.get('description', '')
        if description:
            # Show what we're analyzing states for
            desc_preview = description[:100] + "..." if len(description) > 100 else description
            print(f"{indent}{self._color('Analyzing for:', 'dim')} {desc_preview}")
        
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
    
    def _show_color_resolution_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        output = node.output_data
        
        if output.get("resolved_color"):
            rgba = output["resolved_color"]
            color_name = output.get("color_name", "unknown")
            
            if len(rgba) >= 3:
                r, g, b = rgba[:3]
                alpha = rgba[3] if len(rgba) > 3 else 1.0
                
                # Create a visual representation with color name
                color_str = self._color(f"'{color_name}'", "bold")
                rgba_str = self._color(f"rgb({r:.3f}, {g:.3f}, {b:.3f})", "yellow")
                if alpha < 1.0:
                    rgba_str = self._color(f"rgba({r:.3f}, {g:.3f}, {b:.3f}, {alpha:.3f})", "yellow")
                
                print(f"{indent}Resolved: {color_str} → {rgba_str}")
                
                # Show hex approximation for convenience
                hex_r = int(r * 255)
                hex_g = int(g * 255)
                hex_b = int(b * 255)
                hex_str = self._color(f"#{hex_r:02x}{hex_g:02x}{hex_b:02x}", "dim")
                print(f"{indent}Hex approx: {hex_str}")
            else:
                print(f"{indent}Resolved: {self._color(color_name, 'yellow')} → {rgba}")
        elif node.llm_call and node.llm_call.parsed_response:
            parsed = node.llm_call.parsed_response
            if isinstance(parsed, dict):
                color_name = parsed.get('color_name', 'unknown')
                rgba_list = parsed.get('rgba_list', [])
                if rgba_list and len(rgba_list) >= 3:
                    r, g, b = rgba_list[:3]
                    color_str = self._color(f"'{color_name}'", "bold")
                    rgba_str = self._color(f"rgb({r:.3f}, {g:.3f}, {b:.3f})", "yellow")
                    print(f"{indent}LLM resolved: {color_str} → {rgba_str}")
        else:
            color_name = node.metadata.get("color_name", "unknown")
            print(f"{indent}{self._color(f'Resolving: {color_name}', 'yellow')}")
    
    def _show_refinement_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        output = node.output_data
        
        if output.get("changes_made"):
            changes = output["changes_made"]
            print(f"{indent}{self._color('Changes:', 'bold')} {self._color(changes, 'green')}")
        
        if output.get("warnings"):
            warnings = output["warnings"]
            print(f"{indent}{self._color('⚠️  Warnings:', 'yellow')} {warnings}")
        
        if output.get("validation_issues"):
            issues = output["validation_issues"]
            print(f"{indent}{self._color('⚠️  Validation:', 'yellow')} {issues}")
        
        if output.get("code_length"):
            length = output["code_length"]
            print(f"{indent}{self._color('Code size:', 'dim')} {length} chars")
    
    def _show_sketch_repair_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        output = node.output_data
        
        if output.get("repair_success"):
            print(f"{indent}{self._color('✅ Repair successful', 'green')}")
            if output.get("changes_made"):
                changes = output["changes_made"]
                print(f"{indent}{self._color('Changes:', 'bold')} {self._color(changes, 'green')}")
            if output.get("code_lines_changed"):
                lines_changed = output["code_lines_changed"]
                print(f"{indent}{self._color('Lines modified:', 'dim')} {lines_changed}")
            if output.get("final_code_length"):
                code_length = output["final_code_length"]
                print(f"{indent}{self._color('Final code size:', 'dim')} {code_length} chars")
        else:
            print(f"{indent}{self._color('❌ Repair failed', 'red')}")
            if output.get("error"):
                error_msg = output["error"]
                print(f"{indent}{self._color('Error:', 'red')} {error_msg}")
            if output.get("exception"):
                print(f"{indent}{self._color('Type:', 'dim')} Exception during repair process")
    
    def _show_analysis_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        
        # Check both output_data and LLM parsed response
        data = node.output_data or {}
        if node.llm_call and node.llm_call.parsed_response:
            llm_data = node.llm_call.parsed_response
            if isinstance(llm_data, dict):
                data.update(llm_data)
        
        if not data:
            return
            
        print(f"{indent}{self._color('🔍 Stage 1 Analysis Results:', 'bold')}")
        
        # Show implementation plan
        implementation_plan = data.get('implementation_plan', '')
        if implementation_plan:
            plan_preview = implementation_plan[:200] + "..." if len(implementation_plan) > 200 else implementation_plan
            print(f"{indent}  {self._color('Implementation Plan:', 'cyan')}")
            print(f"{indent}    {self._color(plan_preview, 'dim')}")
            print(f"{indent}    {self._color(f'[{len(implementation_plan)} total characters]', 'dim')}")
        
        # Show errors found
        errors_found = data.get('errors_found', '')
        if errors_found:
            print(f"{indent}  {self._color('⚠️  Errors Found:', 'red')}")
            errors_preview = errors_found[:150] + "..." if len(errors_found) > 150 else errors_found
            print(f"{indent}    {self._color(errors_preview, 'red')}")
        
        # Show architectural needs
        architectural_needs = data.get('architectural_needs', '')
        if architectural_needs:
            print(f"{indent}  {self._color('🏗️  Architectural Needs:', 'yellow')}")
            arch_preview = architectural_needs[:150] + "..." if len(architectural_needs) > 150 else architectural_needs
            print(f"{indent}    {self._color(arch_preview, 'yellow')}")
    
    def _show_implementation_output(self, node: TraceNode, depth: int):
        indent = self._get_indent(depth)
        
        # Check both output_data and LLM parsed response
        data = node.output_data or {}
        if node.llm_call and node.llm_call.parsed_response:
            llm_data = node.llm_call.parsed_response
            if isinstance(llm_data, dict):
                data.update(llm_data)
        
        if not data:
            return
            
        print(f"{indent}{self._color('⚙️ Stage 2 Implementation Results:', 'bold')}")
        
        # Show changes summary
        changes_summary = data.get('changes_summary', '')
        if changes_summary:
            print(f"{indent}  {self._color('✨ Changes Made:', 'green')}")
            changes_preview = changes_summary[:200] + "..." if len(changes_summary) > 200 else changes_summary
            print(f"{indent}    {self._color(changes_preview, 'green')}")
        
        # Show refined code stats
        refined_code = data.get('refined_code', '')
        if refined_code:
            code_lines = len(refined_code.split('\n'))
            code_chars = len(refined_code)
            print(f"{indent}  {self._color('Refined Code:', 'blue')}")
            print(f"{indent}    {self._color(f'Lines: {code_lines}', 'dim')} | {self._color(f'Characters: {code_chars}', 'dim')}")
            
            # Show first few lines as preview
            first_lines = refined_code.split('\n')[:3]
            for line in first_lines:
                if line.strip():  # Skip empty lines
                    line_preview = line[:80] + "..." if len(line) > 80 else line
                    print(f"{indent}    {self._color(line_preview, 'dim')}")
                    break  # Show only first non-empty line
    
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