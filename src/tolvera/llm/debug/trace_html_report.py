import json
import html
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from .diagram_generator import MermaidDiagramGenerator

class TraceHTMLReporter:
    
    def generate_report(self, trace_data: Dict[str, Any]) -> str:
        # Calculate summary statistics
        summary_stats = self._generate_summary_stats(trace_data)
        
        # Calculate effective duration (LLM processing time only)
        llm_calls = self._collect_llm_calls(trace_data)
        effective_duration = self._calculate_effective_duration(llm_calls)
        
        # Generate timeline
        timeline_segments = self._generate_timeline(trace_data)
        
        # Extract system prompt from first LLM call
        system_prompt = self._extract_system_prompt(trace_data)
        
        # Generate LLM call sections
        llm_call_sections = self._generate_llm_sections(trace_data)
        
        # Generate Mermaid diagrams
        diagram_generator = MermaidDiagramGenerator()
        # Convert dict to TraceNode-like structure for diagram generator
        from .tracing import TraceNode
        trace_node = self._dict_to_trace_node(trace_data)
        flow_diagram = diagram_generator.generate_raw(trace_node)
        sequence_diagram = diagram_generator.generate_sequence_raw(trace_node)
        
        # Build HTML with template parts
        html = self._build_html(
            summary_stats,
            timeline_segments,
            f"{effective_duration:.1f}",
            llm_call_sections,
            system_prompt,
            flow_diagram,
            sequence_diagram
        )
        
        return html
    
    def _build_html(self, summary_stats: str, timeline_segments: str, 
                    total_duration: str, llm_call_sections: str, system_prompt: str,
                    flow_diagram: str, sequence_diagram: str) -> str:
        """Build the complete HTML document."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tölvera LLM Trace Report</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}
        }});
    </script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <h1>Tölvera LLM Trace Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                {summary_stats}
            </div>
        </div>
        
        <div class="timeline">
            <h2>Execution Timeline</h2>
            <div class="timeline-bar">
                {timeline_segments}
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);">
                Total duration: {total_duration}ms
            </div>
            
            <!-- Color Legend -->
            <div style="margin-top: 15px; padding: 15px; background-color: var(--bg-secondary); border-radius: 4px;">
                <div style="font-weight: bold; margin-bottom: 8px; color: var(--text-primary);">Timeline Color Legend:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 12px;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #9C27B0; border-radius: 2px;"></div>
                        <span>Decomposition</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #4CAF50; border-radius: 2px;"></div>
                        <span>State Analysis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #2196F3; border-radius: 2px;"></div>
                        <span>Single Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #FF9800; border-radius: 2px;"></div>
                        <span>Interaction Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #E91E63; border-radius: 2px;"></div>
                        <span>Drawing Expert Synthesis</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="diagram-section">
            <h2>Trace Visualization</h2>
            <div class="diagram-tabs">
                <button class="tab-button active" onclick="showDiagram('flow')">Flow Diagram</button>
                <button class="tab-button" onclick="showDiagram('sequence')">Sequence Diagram</button>
            </div>
            <div id="flow-diagram" class="diagram-container active">
                <div class="diagram-actions">
                    <button onclick="fullscreenDiagram('flow')">⛶ Fullscreen</button>
                    <button onclick="downloadDiagram('flow')">⬇ Download SVG</button>
                </div>
                <div class="mermaid">
                    {flow_diagram}
                </div>
            </div>
            <div id="sequence-diagram" class="diagram-container" style="display: none;">
                <div class="diagram-actions">
                    <button onclick="fullscreenDiagram('sequence')">⛶ Fullscreen</button>
                    <button onclick="downloadDiagram('sequence')">⬇ Download SVG</button>
                </div>
                <div class="mermaid">
                    {sequence_diagram}
                </div>
            </div>
        </div>
        
        <div class="llm-calls">
            <h2>LLM Synthesis Calls</h2>
            {llm_call_sections}
        </div>
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>'''
    
    def _get_css(self) -> str:
        """Return the CSS styles."""
        return '''
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: #ffffff;
            --text-primary: #333333;
            --text-secondary: #666666;
            --accent: #2196F3;
            --success: #4CAF50;
            --error: #f44336;
            --border: #e0e0e0;
            --code-bg: #f8f8f8;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background-color: var(--bg-primary);
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: var(--bg-secondary);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1, h2, h3 {
            margin-top: 0;
        }
        
        .summary {
            background-color: var(--bg-primary);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-card {
            background-color: var(--bg-secondary);
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent);
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .timeline {
            margin: 30px 0;
            padding: 20px;
            background-color: var(--bg-primary);
            border-radius: 8px;
        }
        
        .timeline-bar {
            position: relative;
            height: 40px;
            margin: 10px 0;
            background-color: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .timeline-segment {
            position: absolute;
            height: 100%;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .llm-call {
            margin: 20px 0;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .llm-header {
            background-color: var(--bg-primary);
            padding: 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.2s;
        }
        
        .llm-header:hover {
            background-color: #e8e8e8;
        }
        
        .llm-title {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .llm-icon {
            font-size: 20px;
        }
        
        .llm-description {
            font-weight: 500;
        }
        
        .llm-arrow {
            color: var(--text-secondary);
            margin: 0 5px;
        }
        
        .llm-result {
            color: var(--success);
            font-family: monospace;
        }
        
        .llm-duration {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .llm-content {
            display: none;
            padding: 20px;
            border-top: 1px solid var(--border);
        }
        
        .llm-content.active {
            display: block;
        }
        
        .prompt-section, .code-section, .metadata-section {
            margin: 20px 0;
        }
        
        .section-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 12px;
        }
        
        .prompt-box, .code-box {
            background-color: var(--code-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            position: relative;
        }
        
        .code-box {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .copy-button {
            position: absolute;
            top: 5px;
            right: 5px;
            padding: 5px 10px;
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .code-box:hover .copy-button,
        .prompt-box:hover .copy-button {
            opacity: 1;
        }
        
        .copy-button:hover {
            background-color: #1976D2;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .metadata-item {
            display: flex;
            flex-direction: column;
        }
        
        .metadata-label {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .metadata-value {
            font-weight: 500;
        }
        
        .expand-icon {
            transition: transform 0.2s;
            display: inline-block;
        }
        
        .llm-call.expanded .expand-icon {
            transform: rotate(90deg);
        }
        
        .error {
            color: var(--error);
        }
        
        .success {
            color: var(--success);
        }
        
        .system-prompt-section {
            margin: 20px 0;
        }
        
        .system-prompt-section .collapsible-section {
            background-color: var(--bg-primary);
            border-radius: 8px;
        }
        
        .system-prompt-section .collapsible-header {
            padding: 15px 20px;
            font-size: 16px;
            font-weight: 600;
        }
        
        .system-prompt-section .collapsible-content {
            padding: 0 20px 20px 20px;
        }
        
        .collapsible-section {
            margin: 10px 0;
            border: 1px solid var(--border);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .collapsible-header {
            background-color: var(--bg-primary);
            padding: 10px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
            font-size: 14px;
        }
        
        .collapsible-header:hover {
            background-color: #e8e8e8;
        }
        
        .collapsible-content {
            display: none;
            padding: 15px;
            background-color: var(--code-bg);
            font-family: monospace;
            font-size: 13px;
            overflow-x: auto;
        }
        
        .collapsible-content.active {
            display: block;
        }
        
        .json-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        /* Syntax highlighting */
        .keyword { color: #0000ff; font-weight: bold; }
        .decorator { color: #aa22ff; }
        .function { color: #00aa00; }
        .string { color: #aa5500; }
        .comment { color: #888888; font-style: italic; }
        .number { color: #0088ff; }
        
        /* Diagram styles */
        .diagram-section {
            margin: 30px 0;
            background-color: var(--bg-primary);
            padding: 20px;
            border-radius: 8px;
        }
        
        .diagram-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
        }
        
        .tab-button {
            padding: 10px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-secondary);
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }
        
        .tab-button:hover {
            color: var(--text-primary);
        }
        
        .tab-button.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        
        .diagram-container {
            background-color: var(--bg-secondary);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            overflow: auto;
            max-height: 600px;
        }
        
        .diagram-container.active {
            display: block;
        }
        
        .diagram-actions {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 10px;
            z-index: 10;
        }
        
        .diagram-actions button {
            padding: 5px 10px;
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0.8;
            transition: opacity 0.2s;
        }
        
        .diagram-actions button:hover {
            opacity: 1;
        }
        
        .mermaid {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 300px;
        }
        
        .mermaid svg {
            max-width: 100%;
            height: auto;
        }
        
        /* Fullscreen styles */
        .fullscreen-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .fullscreen-content {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            max-width: 95%;
            max-height: 95%;
            overflow: auto;
            position: relative;
        }
        
        .fullscreen-close {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 10px 20px;
            background-color: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        '''
    
    def _get_javascript(self) -> str:
        """Return the JavaScript code."""
        return '''
        // Toggle main LLM call sections
        document.querySelectorAll('.llm-header').forEach(header => {
            header.addEventListener('click', () => {
                const call = header.parentElement;
                const content = call.querySelector('.llm-content');
                call.classList.toggle('expanded');
                content.classList.toggle('active');
            });
        });
        
        // Toggle collapsible subsections
        document.querySelectorAll('.collapsible-header').forEach(header => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                content.classList.toggle('active');
                const icon = header.querySelector('.expand-icon');
                if (icon) {
                    icon.style.transform = content.classList.contains('active') ? 'rotate(90deg)' : 'rotate(0deg)';
                }
            });
        });
        
        // Copy to clipboard functionality
        document.querySelectorAll('.copy-button').forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const codeBox = button.parentElement.querySelector('pre');
                const text = codeBox.textContent;
                navigator.clipboard.writeText(text).then(() => {
                    button.textContent = 'Copied!';
                    setTimeout(() => {
                        button.textContent = 'Copy';
                    }, 2000);
                });
            });
        });
        
        // Diagram tab switching
        window.showDiagram = function(type) {
            // Update tabs
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update diagram containers
            document.getElementById('flow-diagram').style.display = type === 'flow' ? 'block' : 'none';
            document.getElementById('sequence-diagram').style.display = type === 'sequence' ? 'block' : 'none';
        };
        
        // Fullscreen diagram functionality
        window.fullscreenDiagram = function(type) {
            const diagramContainer = document.getElementById(`${type}-diagram`);
            const mermaidSvg = diagramContainer.querySelector('svg');
            
            if (!mermaidSvg) return;
            
            // Create fullscreen overlay
            const overlay = document.createElement('div');
            overlay.className = 'fullscreen-overlay';
            
            const content = document.createElement('div');
            content.className = 'fullscreen-content';
            
            const closeBtn = document.createElement('button');
            closeBtn.className = 'fullscreen-close';
            closeBtn.textContent = '✕ Close';
            closeBtn.onclick = () => document.body.removeChild(overlay);
            
            const svgClone = mermaidSvg.cloneNode(true);
            svgClone.style.maxWidth = '100%';
            svgClone.style.maxHeight = '90vh';
            
            content.appendChild(closeBtn);
            content.appendChild(svgClone);
            overlay.appendChild(content);
            document.body.appendChild(overlay);
            
            // Close on ESC key
            const escHandler = (e) => {
                if (e.key === 'Escape') {
                    document.body.removeChild(overlay);
                    document.removeEventListener('keydown', escHandler);
                }
            };
            document.addEventListener('keydown', escHandler);
        };
        
        // Download diagram as SVG
        window.downloadDiagram = function(type) {
            const diagramContainer = document.getElementById(`${type}-diagram`);
            const mermaidSvg = diagramContainer.querySelector('svg');
            
            if (!mermaidSvg) return;
            
            // Get SVG content
            const svgData = new XMLSerializer().serializeToString(mermaidSvg);
            const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
            const svgUrl = URL.createObjectURL(svgBlob);
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = svgUrl;
            downloadLink.download = `tolvera-${type}-diagram.svg`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            // Clean up
            URL.revokeObjectURL(svgUrl);
        };
        
        '''
    
    def _generate_summary_stats(self, trace_data: Dict[str, Any]) -> str:
        """Generate summary statistics HTML."""
        # Count LLM calls
        llm_calls = self._collect_llm_calls(trace_data)
        successful_calls = sum(1 for call in llm_calls if call.get('status') == 'success')
        
        stats = [
            ("Duration", f"{(trace_data.get('duration_ms') or 0):.1f}ms", ""),
            ("LLM Calls", str(len(llm_calls)), ""),
            ("Success Rate", f"{(successful_calls/len(llm_calls)*100 if llm_calls else 0):.0f}%", ""),
            ("Timestamp", trace_data.get('timestamp', '').split('.')[0], "")
        ]
        
        html_parts = []
        for label, value, icon in stats:
            html_parts.append(f"""
                <div class="stat-card">
                    <div class="stat-value">{icon} {value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """)
        
        return "".join(html_parts)
    
    def _extract_system_prompt(self, trace_data: Dict[str, Any]) -> str:
        """Extract unique system prompts from all LLM calls."""
        llm_calls = self._collect_llm_calls(trace_data)
        
        # Collect unique system prompts
        unique_prompts = {}
        for call in llm_calls:
            if call.get('llm_call'):
                prompt = call['llm_call'].get('system_prompt', '')
                if prompt:
                    prompt_type = self._identify_prompt_type(prompt)
                    if prompt_type not in unique_prompts:
                        unique_prompts[prompt_type] = prompt
        
        if not unique_prompts:
            return 'No system prompts found'
        
        sections = []
        for prompt_type, prompt in unique_prompts.items():
            sections.append(f"### {prompt_type}\n\n{prompt}")
        
        return "\n\n---\n\n".join(sections)
    
    def _identify_prompt_type(self, prompt: str) -> str:
        """Identify the type of prompt based on content."""
        prompt_lower = prompt.lower()
        
        if "decomposing complex behavior" in prompt_lower:
            return "Behavior Decomposition Prompt"
        elif "drawing" in prompt_lower or "visual" in prompt_lower:
            return "Drawing Synthesis Prompt"
        elif "kernel" in prompt_lower and "integrate" in prompt_lower:
            return "Kernel Integration Prompt"
        elif "analyze what custom states" in prompt_lower or "temporal updates" in prompt_lower:
            return "State Analysis Prompt"
        elif "expert at analyzing particle behaviors" in prompt_lower:
            return "State Analysis Prompt"
        elif "tölvera core api" in prompt_lower:
            return "Expert Synthesis Prompt"
        else:
            return "System Prompt"
    
    def _generate_timeline(self, trace_data: Dict[str, Any]) -> str:
        """Generate timeline visualization."""
        llm_calls = self._collect_llm_calls(trace_data)
        if not llm_calls:
            return ""
        
        # Find the earliest LLM call start time to eliminate initial gap
        earliest_start = None
        latest_end = None
        
        for call in llm_calls:
            call_start = call.get('start_time', 0)
            call_duration = call.get('duration_ms', 0)
            call_end = call_start + (call_duration / 1000.0)  # Convert ms to seconds
            
            if earliest_start is None or call_start < earliest_start:
                earliest_start = call_start
            if latest_end is None or call_end > latest_end:
                latest_end = call_end
        
        # Calculate effective duration from first to last LLM call
        effective_duration = ((latest_end - earliest_start) * 1000) if earliest_start is not None and latest_end is not None else (trace_data.get('duration_ms') or 0)
        
        segments = []
        # Define semantic colors for different types of operations
        color_scheme = {
            'decomposition': '#9C27B0',      # Purple - for decomposition/analysis
            'state_analysis': '#4CAF50',     # Green - for state analysis  
            'synthesis_single': '#2196F3',   # Blue - for single-particle expert synthesis
            'synthesis_interaction': '#FF9800',  # Orange - for interaction expert synthesis
            'synthesis_drawing': '#E91E63',  # Pink - for drawing expert synthesis
            'default': '#607D8B'             # Gray - fallback
        }
        
        for call in llm_calls:
            # Calculate offset from the earliest LLM call (not trace start)
            start_offset = ((call.get('start_time', 0) - earliest_start) * 1000) if earliest_start is not None else 0
            duration = call.get('duration_ms') or 0
            width_percent = (duration / effective_duration) * 100 if effective_duration > 0 else 0
            left_percent = (start_offset / effective_duration) * 100 if effective_duration > 0 else 0
            
            # Determine what to show in timeline based on call type
            input_data = call.get('input_data', {})
            expert_name = input_data.get('expert_name', '')
            
            # Check if this is a decomposition call
            is_decomposition = call.get('name') == 'llm_decompose' or 'decompose' in call.get('name', '').lower()
            is_state_analysis = call.get('name') == 'llm_state_analysis' or 'state_analysis' in call.get('name', '').lower()
            
            # Determine call type and assign appropriate color
            if is_decomposition:
                # For decomposition, show "decomposition" as the label
                timeline_label = "decomposition"
                color = color_scheme['decomposition']
            elif is_state_analysis:
                # For state analysis, show brief result
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                if parsed and isinstance(parsed, dict):
                    global_states = parsed.get('global_states', [])
                    particle_states = parsed.get('particle_states', [])
                    species_states = parsed.get('species_states', [])
                    
                    global_count = len(global_states) if isinstance(global_states, list) else global_states if isinstance(global_states, int) else 0
                    particle_count = len(particle_states) if isinstance(particle_states, list) else particle_states if isinstance(particle_states, int) else 0
                    species_count = len(species_states) if isinstance(species_states, list) else species_states if isinstance(species_states, int) else 0
                    
                    total_states = global_count + particle_count + species_count
                    timeline_label = f"states({total_states})" if total_states > 0 else "states(0)"
                else:
                    timeline_label = "state_analysis"
                color = color_scheme['state_analysis']
            else:
                # For synthesis, determine type and assign appropriate color
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                description = input_data.get('description', '').lower()
                
                # Detect synthesis type for appropriate coloring
                is_interaction = parsed.get('is_interaction', False) if parsed else False
                is_drawing = 'draw' in description or 'trail' in description or 'glow' in description or 'visual' in description
                
                if is_drawing:
                    color = color_scheme['synthesis_drawing']
                elif is_interaction:
                    color = color_scheme['synthesis_interaction']  
                else:
                    color = color_scheme['synthesis_single']
                
                # Set timeline label
                if expert_name:
                    timeline_label = expert_name
                else:
                    # Fallback to parsing from response if available
                    timeline_label = parsed.get('name', 'synthesis') if parsed else 'synthesis'
            
            segments.append(f"""
                <div class="timeline-segment" style="
                    left: {left_percent:.1f}%;
                    width: {width_percent:.1f}%;
                    background-color: {color};
                ">
                    {html.escape(timeline_label)}
                </div>
            """)
        
        return "".join(segments)
    
    def _calculate_effective_duration(self, llm_calls: List[Dict[str, Any]]) -> float:
        """Calculate effective duration from first to last LLM call."""
        if not llm_calls:
            return 0.0
        
        earliest_start = None
        latest_end = None
        
        for call in llm_calls:
            call_start = call.get('start_time', 0)
            call_duration = call.get('duration_ms', 0)
            call_end = call_start + (call_duration / 1000.0)  # Convert ms to seconds
            
            if earliest_start is None or call_start < earliest_start:
                earliest_start = call_start
            if latest_end is None or call_end > latest_end:
                latest_end = call_end
        
        if earliest_start is not None and latest_end is not None:
            return (latest_end - earliest_start) * 1000  # Return in milliseconds
        
        return 0.0
    
    def _generate_llm_sections(self, trace_data: Dict[str, Any]) -> str:
        """Generate LLM call sections with full transparency."""
        llm_calls = self._collect_llm_calls(trace_data)
        sections = []
        
        for i, call in enumerate(llm_calls):
            llm_data = call.get('llm_call', {})
            if not llm_data:
                continue
            
            # Extract key information with null safety
            input_data = call.get('input_data', {})
            parsed_response = llm_data.get('parsed_response')
            
            # For synthesis calls, check if we have an expert_name in input_data
            synthesis_expert_name = input_data.get('expert_name', '')
            
            # Handle different types of LLM calls
            is_decomposition = call.get('name') == 'llm_decompose' or 'decompose' in call.get('name', '').lower()
            is_state_analysis = call.get('name') == 'llm_state_analysis' or 'state_analysis' in call.get('name', '').lower()
            
            if is_decomposition:
                # This is a decomposition call
                code = ''  # Decomposition doesn't generate code
                if parsed_response and parsed_response.get('components'):
                    components = parsed_response.get('components', [])
                    expert_name = f"decompose → {len(components)} experts"
                else:
                    expert_name = 'decompose'
            elif is_state_analysis:
                # This is a state analysis call
                code = ''  # State analysis doesn't generate code
                if parsed_response:
                    # Count actual states from lists
                    global_states = parsed_response.get('global_states', [])
                    particle_states = parsed_response.get('particle_states', [])
                    species_states = parsed_response.get('species_states', [])
                    temporal_updates = parsed_response.get('temporal_updates', [])
                    
                    global_count = len(global_states) if isinstance(global_states, list) else global_states if isinstance(global_states, int) else 0
                    particle_count = len(particle_states) if isinstance(particle_states, list) else particle_states if isinstance(particle_states, int) else 0
                    species_count = len(species_states) if isinstance(species_states, list) else species_states if isinstance(species_states, int) else 0
                    temporal_count = len(temporal_updates) if isinstance(temporal_updates, list) else temporal_updates if isinstance(temporal_updates, int) else 0
                    
                    total_states = global_count + particle_count + species_count
                    if total_states > 0:
                        parts = []
                        if global_count > 0:
                            parts.append(f"{global_count}G")
                        if particle_count > 0:
                            parts.append(f"{particle_count}P")
                        if species_count > 0:
                            parts.append(f"{species_count}S")
                        if temporal_count > 0:
                            parts.append(f"{temporal_count}T")
                        expert_name = f"state_analysis → {'+'.join(parts)}"
                    else:
                        expert_name = "state_analysis → no states"
                else:
                    expert_name = 'state_analysis'
            elif parsed_response is None:
                # Handle other LLM calls without parsed_response
                expert_name = 'processing'
                code = ''
            else:
                # For synthesis calls, prefer the expert_name from input_data if available
                if synthesis_expert_name:
                    expert_name = synthesis_expert_name
                elif parsed_response and parsed_response.get('name'):
                    expert_name = parsed_response.get('name')
                else:
                    # If no expert name found, default to 'synthesis'
                    expert_name = 'synthesis'
                code = parsed_response.get('code', '') or ''
            duration = call.get('duration_ms', 0) or 0  # Ensure duration is never None
            model = llm_data.get('model', 'Unknown') or 'Unknown'
            
            # Identify prompt type with null safety
            system_prompt = llm_data.get('system_prompt', '') or ''
            prompt_type = self._identify_prompt_type(system_prompt) or 'Unknown'
            
            # Format code with basic syntax highlighting
            formatted_code = self._format_code(code) or ''
            
            # Generate section HTML with collapsible subsections
            status_class = 'success' if call.get('status') == 'success' else 'error'
            section = f"""
            <div class="llm-call">
                <div class="llm-header">
                    <div class="llm-title">
                        <span class="llm-icon">🤖</span>
                        <span class="llm-description">{html.escape(expert_name)}{'' if is_decomposition else '()'}</span>
                        <span class="llm-arrow">→</span>
                        <span class="llm-result"></span>
                        <span style="margin-left: 10px; padding: 2px 8px; background: #e3f2fd; color: #1976d2; border-radius: 4px; font-size: 11px;">{prompt_type}</span>
                    </div>
                    <div>
                        <span class="llm-duration">{duration:.0f}ms</span>
                        <span class="expand-icon">▶</span>
                    </div>
                </div>
                
                <div class="llm-content">
                    <!-- User Prompt -->
                    <div class="collapsible-section">
                        <div class="collapsible-header">
                            <span>📝 User Prompt</span>
                            <span class="expand-icon">▶</span>
                        </div>
                        <div class="collapsible-content">
                            <pre>{html.escape(llm_data.get('user_prompt', '') or '')}</pre>
                        </div>
                    </div>
                    
                    <!-- Full Prompt -->
                    <div class="collapsible-section">
                        <div class="collapsible-header">
                            <span>📄 Full Prompt (System + User)</span>
                            <span class="expand-icon">▶</span>
                        </div>
                        <div class="collapsible-content">
                            <pre>{html.escape(llm_data.get('full_prompt', '') or '')}</pre>
                        </div>
                    </div>
                    
                    <!-- Raw Response -->
                    <div class="collapsible-section">
                        <div class="collapsible-header">
                            <span>🔄 Raw LLM Response</span>
                            <span class="expand-icon">▶</span>
                        </div>
                        <div class="collapsible-content">
                            <pre>{html.escape(llm_data.get('raw_response', 'No raw response available') or 'No raw response available')}</pre>
                        </div>
                    </div>
                    
                    <!-- Parsed Response -->
                    {f'''<div class="collapsible-section">
                        <div class="collapsible-header">
                            <span>🔍 Parsed Response (JSON)</span>
                            <span class="expand-icon">▶</span>
                        </div>
                        <div class="collapsible-content json-content">
                            <pre>{json.dumps(parsed_response, indent=2) if parsed_response else "No parsed response available"}</pre>
                        </div>
                    </div>''' if parsed_response else ''}
                    
                    <!-- State Analysis Details (if this is a state analysis call) -->
                    {self._generate_state_analysis_details(parsed_response, is_state_analysis) if is_state_analysis else ''}
                    
                    <!-- Decomposition Components (if applicable) -->
                    {self._generate_decomposition_section(parsed_response) if parsed_response and 'components' in parsed_response else ''}
                    
                    <!-- Generated Code (expanded by default) -->
                    {f'''<div class="code-section">
                        <div class="section-title">✨ Generated Code</div>
                        <div class="code-box">
                            <button class="copy-button">Copy</button>
                            <pre>{formatted_code}</pre>
                        </div>
                    </div>''' if not is_decomposition and not is_state_analysis and code else ''}
                    
                    <!-- Metadata -->
                    <div class="metadata-section">
                        <div class="section-title">📊 Metadata</div>
                        <div class="metadata-grid">
                            <div class="metadata-item">
                                <span class="metadata-label">Model</span>
                                <span class="metadata-value">{html.escape(model)}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Duration</span>
                                <span class="metadata-value">{duration:.0f}ms</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">API Call Time</span>
                                <span class="metadata-value">{(llm_data.get('api_call_ms') or 0):.0f}ms</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Type</span>
                                <span class="metadata-value">{'Interaction' if parsed_response and parsed_response.get('is_interaction') else 'Single' if parsed_response else 'Decomposition'}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Status</span>
                                <span class="metadata-value {status_class}">{call.get('status', 'unknown')}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Code Length</span>
                                <span class="metadata-value">{len(code)} chars</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            sections.append(section)
        
        return "".join(sections)
    
    def _collect_llm_calls(self, node: Dict[str, Any], calls: Optional[List] = None) -> List[Dict[str, Any]]:
        """Recursively collect all LLM call nodes."""
        if calls is None:
            calls = []
        
        if node.get('type') == 'llm_call' and node.get('llm_call'):
            calls.append(node)
        
        for child in node.get('children', []):
            self._collect_llm_calls(child, calls)
        
        return calls
    
    def _generate_decomposition_section(self, parsed_response: Dict[str, Any]) -> str:
        """Generate HTML section for decomposition components."""
        if not parsed_response or 'components' not in parsed_response:
            return ''
        
        components = parsed_response.get('components', [])
        is_simple = parsed_response.get('is_simple', False)
        interpretation = parsed_response.get('interpretation', '')
        context = parsed_response.get('context', {})
        
        section = f'''
        <div class="decomposition-section">
            <div class="section-title">🧩 Decomposition Results</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
                <div style="margin-bottom: 10px;">
                    <strong>Interpretation:</strong> {html.escape(interpretation)}
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Complexity:</strong> <span style="color: {'#4CAF50' if is_simple else '#FF9800'}">
                        {'Simple (single expert)' if is_simple else f'Complex ({len(components)} experts)'}
                    </span>
                </div>
                
                {'<div style="margin-top: 15px;"><strong>Components:</strong></div>' if components else ''}
                <div style="margin-left: 20px;">
        '''
        
        for i, comp in enumerate(components):
            expert_name = comp.get('expert_name', 'unknown')
            expert_type = comp.get('expert_type', 'unknown')
            description = comp.get('description', '')
            implementation = comp.get('implementation', '')
            priority = comp.get('priority', 1.0)
            
            section += f'''
                <div style="margin: 10px 0; padding: 10px; background-color: var(--bg-secondary); border-left: 3px solid var(--accent); border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{i+1}. {html.escape(expert_name)}</strong>
                        <span style="font-size: 12px; color: var(--text-secondary);">
                            Type: {expert_type} | Weight: {priority}
                        </span>
                    </div>
                    <div style="margin-top: 5px; color: var(--text-secondary);">
                        {html.escape(description)}
                    </div>
                    <div style="margin-top: 8px; padding: 8px; background-color: var(--code-bg); border-radius: 4px; font-family: monospace; font-size: 12px;">
                        <strong>Implementation:</strong><br>
                        {html.escape(implementation)}
                    </div>
                </div>
            '''
        
        section += '</div>'
        
        # Add context if present
        if context:
            constraints = context.get('constraints', [])
            relationships = context.get('relationships', [])
            
            if constraints or relationships:
                section += '''
                <div style="margin-top: 15px;">
                    <strong>Context Threading:</strong>
                    <ul style="margin: 5px 0 0 20px;">
                '''
                for constraint in constraints:
                    section += f'<li>{html.escape(constraint)}</li>'
                for rel in relationships:
                    section += f'<li>{html.escape(rel)}</li>'
                section += '</ul></div>'
        
        section += '</div></div>'
        return section
    
    def _generate_state_analysis_details(self, parsed_response: Dict[str, Any], is_state_analysis: bool) -> str:
        """Generate detailed HTML section for state analysis responses."""
        if not is_state_analysis or not parsed_response:
            return ''
        
        section = '''
        <div class="state-analysis-section">
            <div class="section-title">🧬 State Analysis Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show needs_states
        needs_states = parsed_response.get('needs_states', False)
        status_color = '#4CAF50' if needs_states else '#9E9E9E' 
        section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Needs Custom States:</strong> 
                    <span style="color: {status_color}; font-weight: bold;">{"Yes" if needs_states else "No"}</span>
                </div>
        '''
        
        if needs_states:
            # Show each category of states
            for category in ['global_states', 'particle_states', 'species_states']:
                states = parsed_response.get(category, [])
                if states:
                    category_name = category.replace('_states', '').capitalize()
                    section += f'''
                    <div style="margin: 15px 0;">
                        <h4 style="margin: 10px 0; color: var(--accent);">{category_name} States ({len(states)})</h4>
                        <div style="margin-left: 20px;">
                    '''
                    
                    for state in states:
                        if isinstance(state, dict):
                            name = html.escape(state.get('name', 'unnamed'))
                            state_type = html.escape(state.get('type', 'unknown'))
                            min_val = state.get('min', 'N/A')
                            max_val = state.get('max', 'N/A')
                            desc = html.escape(state.get('description', ''))
                            initial = state.get('initial', 'auto')
                            
                            section += f'''
                            <div style="margin: 10px 0; padding: 10px; background-color: var(--bg-secondary); border-left: 3px solid var(--accent); border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="color: var(--accent);">{name}</strong>
                                    <span style="font-family: monospace; color: var(--text-secondary); font-size: 12px;">{state_type}</span>
                                </div>
                                <div style="margin-top: 5px; font-size: 13px;">
                                    <span style="color: var(--text-secondary);">Range:</span> {min_val} - {max_val} | 
                                    <span style="color: var(--text-secondary);">Initial:</span> {initial}
                                </div>
                                {f'<div style="margin-top: 5px; font-style: italic; color: var(--text-secondary);">{desc}</div>' if desc else ''}
                            </div>
                            '''
                        else:
                            section += f'''
                            <div style="margin: 5px 0; padding: 5px; background-color: var(--bg-secondary); border-radius: 4px;">
                                {html.escape(str(state))}
                            </div>
                            '''
                    
                    section += '</div></div>'
            
            # Show temporal updates
            temporal_updates = parsed_response.get('temporal_updates', [])
            if temporal_updates:
                section += f'''
                <div style="margin: 15px 0;">
                    <h4 style="margin: 10px 0; color: var(--accent);">Temporal Updates ({len(temporal_updates)})</h4>
                    <div style="margin-left: 20px;">
                '''
                
                for update in temporal_updates:
                    if isinstance(update, dict):
                        state_name = html.escape(update.get('state_name', 'unknown'))
                        expression = html.escape(update.get('update_expression', 'unknown'))
                        desc = html.escape(update.get('description', ''))
                        
                        section += f'''
                        <div style="margin: 10px 0; padding: 10px; background-color: var(--bg-secondary); border-left: 3px solid #FF9800; border-radius: 4px;">
                            <div style="font-family: monospace; font-weight: bold; color: #FF9800;">
                                {state_name} = {expression}
                            </div>
                            {f'<div style="margin-top: 5px; font-style: italic; color: var(--text-secondary);">{desc}</div>' if desc else ''}
                        </div>
                        '''
                    else:
                        section += f'''
                        <div style="margin: 5px 0; padding: 5px; background-color: var(--bg-secondary); border-radius: 4px;">
                            {html.escape(str(update))}
                        </div>
                        '''
                
                section += '</div></div>'
        
        section += '</div></div>'
        return section
    
    def _format_code(self, code: str) -> str:
        """Apply basic syntax highlighting to Python code."""
        if not code:
            return ""
        
        # HTML escape first
        code = html.escape(code)
        
        # Basic Python syntax highlighting
        # Keywords
        keywords = ['def', 'return', 'if', 'else', 'elif', 'for', 'while', 'import', 
                   'from', 'class', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False']
        for kw in keywords:
            code = code.replace(f' {kw} ', f' <span class="keyword">{kw}</span> ')
            code = code.replace(f' {kw}(', f' <span class="keyword">{kw}</span>(')
            code = code.replace(f' {kw}:', f' <span class="keyword">{kw}</span>:')
        
        # Decorators
        code = code.replace('@ti.func', '<span class="decorator">@ti.func</span>')
        
        # Function names (basic pattern)
        import re
        code = re.sub(r'def (\w+)', r'def <span class="function">\1</span>', code)
        
        # Numbers
        code = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="number">\1</span>', code)
        
        # Comments
        lines = code.split('\n')
        formatted_lines = []
        for line in lines:
            if '#' in line:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    line = parts[0] + '<span class="comment">#' + parts[1] + '</span>'
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _dict_to_trace_node(self, data: Dict[str, Any]):
        """Convert dictionary data to TraceNode object for diagram generation."""
        from .tracing import TraceNode, LLMCallData
        
        node = TraceNode(
            id=data.get('id', ''),
            type=data.get('type', ''),
            name=data.get('name', ''),
            input_data=data.get('input_data', {}),
            output_data=data.get('output_data', {}),
            metadata=data.get('metadata', {}),
            status=data.get('status', 'success'),
            error=data.get('error', None),
            duration_ms=data.get('duration_ms', None),
            start_time=data.get('start_time', 0),
            end_time=data.get('end_time', 0)
        )
        
        # Convert timestamp string back to datetime if needed
        if 'timestamp' in data:
            from datetime import datetime
            node.timestamp = datetime.fromisoformat(data['timestamp'])
        
        # Convert LLM call data if present
        if data.get('llm_call'):
            llm_data = data['llm_call']
            node.llm_call = LLMCallData(**llm_data)
        
        # Recursively convert children
        for child_data in data.get('children', []):
            child_node = self._dict_to_trace_node(child_data)
            node.add_child(child_node)
        
        return node


def generate_html_report(json_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate an HTML report from a trace JSON file.
    
    Args:
        json_path: Path to the trace JSON file
        output_path: Optional output path for HTML file
        
    Returns:
        Path to the generated HTML file
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        trace_data = json.load(f)
    
    # Generate report
    reporter = TraceHTMLReporter()
    html_content = reporter.generate_report(trace_data)
    
    # Determine output path
    if output_path is None:
        json_path_obj = Path(json_path)
        output_path = json_path_obj.parent / f"{json_path_obj.stem}_report.html"
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trace_html_report.py <trace.json> [output.html]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = generate_html_report(json_file, output_file)
        print(f"✅ HTML report generated: {output_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)