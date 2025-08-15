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
            <h2>LLM Processing Timeline</h2>
            <div class="timeline-bar">
                {timeline_segments}
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);">
                Total processing time: {total_duration}ms (compact view - idle time removed)
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
                        <div style="width: 16px; height: 16px; background-color: #00BCD4; border-radius: 2px;"></div>
                        <span>State Initialization</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #FF5722; border-radius: 2px;"></div>
                        <span>Color Resolution</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #2196F3; border-radius: 2px;"></div>
                        <span>Force Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #FF9800; border-radius: 2px;"></div>
                        <span>Interaction Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #FFC107; border-radius: 2px;"></div>
                        <span>Temporal Update Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #795548; border-radius: 2px;"></div>
                        <span>Utility Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #E91E63; border-radius: 2px;"></div>
                        <span>Drawing Expert Synthesis</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #9C27B0; border-radius: 2px;"></div>
                        <span>Analysis (Stage 1)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #7C4DFF; border-radius: 2px;"></div>
                        <span>Implementation (Stage 2)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 16px; height: 16px; background-color: #F44336; border-radius: 2px;"></div>
                        <span>Error Correction</span>
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
            # ("Success Rate", f"{(successful_calls/len(llm_calls)*100 if llm_calls else 0):.0f}%", ""),
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
        
        # Check for specific prompt types in order of specificity
        if "decomposing complex behavior" in prompt_lower:
            return "Behavior Decomposition Prompt"
        elif "analyze what custom states" in prompt_lower or "analyzing particle behaviors and determining what states" in prompt_lower:
            return "State Analysis Prompt"
        elif "expert at analyzing particle behaviors" in prompt_lower and "states" in prompt_lower:
            return "State Analysis Prompt"
        elif "temporal updates" in prompt_lower and "states" in prompt_lower:
            return "State Analysis Prompt"
        elif "utility expert" in prompt_lower or "temporal update" in prompt_lower or "state update" in prompt_lower:
            return "Utility Expert Synthesis Prompt"
        elif "kernel" in prompt_lower and "integrate" in prompt_lower:
            return "Kernel Integration Prompt"
        elif "stage 1" in prompt_lower and "analyze" in prompt_lower and "implementation plan" in prompt_lower:
            return "Two-Stage Analysis Prompt (Stage 1)"
        elif "stage 2" in prompt_lower and "implement" in prompt_lower and "refactoring plan" in prompt_lower:
            return "Two-Stage Implementation Prompt (Stage 2)"
        elif "refining tölvera particle simulations" in prompt_lower or ("refine" in prompt_lower and "sketch" in prompt_lower):
            return "Refinement Prompt"
        # Check for drawing ONLY if it's explicitly about visual effects, not just if the word appears
        elif ("drawing" in prompt_lower or "visual" in prompt_lower) and ("trail" in prompt_lower or "glow" in prompt_lower or "visual effect" in prompt_lower or "draw_" in prompt_lower):
            return "Drawing Synthesis Prompt"
        elif "tölvera core api" in prompt_lower and "expert" in prompt_lower:
            # This is a force/interaction expert synthesis, not drawing
            return "Force Expert Synthesis Prompt"
        elif "expert function" in prompt_lower or "@ti.func" in prompt_lower:
            return "Expert Synthesis Prompt"
        else:
            return "System Prompt"
    
    def _generate_timeline(self, trace_data: Dict[str, Any]) -> str:
        """Generate compact timeline visualization showing only LLM processing periods."""
        llm_calls = self._collect_llm_calls(trace_data)
        if not llm_calls:
            return ""
        
        # Sort calls by start time
        llm_calls.sort(key=lambda x: x.get('start_time', 0))
        
        # Calculate total LLM processing time (sum of all durations, no gaps)
        total_processing_time = sum(call.get('duration_ms', 0) for call in llm_calls)
        
        if total_processing_time == 0:
            return ""
        
        segments = []
        # Define semantic colors for different types of operations
        color_scheme = {
            'decomposition': '#9C27B0',      # Purple - for decomposition/analysis
            'state_analysis': '#4CAF50',     # Green - for state analysis
            'state_initialization': '#00BCD4', # Cyan - for state initialization
            'color_resolution': '#FF5722',   # Deep Orange - for color resolution
            'synthesis_force': '#2196F3',    # Blue - for force expert synthesis
            'synthesis_interaction': '#FF9800',  # Orange - for interaction expert synthesis
            'synthesis_temporal': '#FFC107', # Amber - for temporal update synthesis
            'synthesis_utility': '#795548',  # Brown - for utility expert synthesis
            'synthesis_drawing': '#E91E63',  # Pink - for drawing expert synthesis
            'analysis_stage': '#9C27B0',     # Purple - for Stage 1 analysis
            'implementation_stage': '#7C4DFF', # Deep purple - for Stage 2 implementation
            'refinement': '#7C4DFF',         # Deep purple - for refinements (legacy)
            'error_correction': '#F44336',   # Red - for error corrections
            'default': '#607D8B'             # Gray - fallback
        }
        
        # Track cumulative position for compact timeline
        cumulative_position = 0
        
        for call in llm_calls:
            duration = call.get('duration_ms') or 0
            
            # Calculate width percentage based on this call's duration relative to total
            width_percent = (duration / total_processing_time) * 100 if total_processing_time > 0 else 0
            
            # Position starts where the previous segment ended (no gaps)
            left_percent = cumulative_position
            
            # Determine what to show in timeline based on call type
            input_data = call.get('input_data', {})
            expert_name = input_data.get('expert_name', '')
            
            # Check if this is a decomposition call
            is_decomposition = call.get('name') == 'llm_decompose' or 'decompose' in call.get('name', '').lower()
            is_state_analysis = call.get('name') == 'llm_state_analysis' or 'state_analysis' in call.get('name', '').lower()
            is_color_resolution = call.get('name') == 'llm_color_resolution' or 'color_resolution' in call.get('name', '').lower()
            is_refinement = 'refinement' in call.get('name', '').lower() or call.get('parent', {}).get('type') == 'refinement'
            is_error_correction = 'error_correction' in call.get('name', '').lower()
            is_sketch_repair = call.get('type') == 'sketch_repair' or 'sketch_repair' in call.get('name', '').lower()
            is_analysis_stage = 'analyze_sketch' in call.get('name', '').lower() or call.get('type') == 'analysis'
            is_implementation_stage = 'implement_refinement' in call.get('name', '').lower() or call.get('type') == 'implementation'
            
            # Determine call type and assign appropriate color
            if is_analysis_stage:
                # For Stage 1 analysis, show the analysis type
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                if parsed and parsed.get('implementation_plan'):
                    timeline_label = "analysis → plan"
                else:
                    timeline_label = "analysis"
                color = color_scheme['analysis_stage']
            elif is_implementation_stage:
                # For Stage 2 implementation, show the changes
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                if parsed and parsed.get('changes_summary'):
                    changes = parsed['changes_summary'][:20] + "..." if len(parsed['changes_summary']) > 20 else parsed['changes_summary']
                    timeline_label = f"implement → {changes}"
                else:
                    timeline_label = "implementation"
                color = color_scheme['implementation_stage']
            elif is_refinement:
                # For refinement, show the type and request
                if is_error_correction:
                    timeline_label = "error_fix"
                    color = color_scheme['error_correction']
                else:
                    timeline_label = "refinement"
                    color = color_scheme['refinement']
            elif is_sketch_repair:
                # For sketch repair, show repair result
                output_data = call.get('output_data', {})
                if output_data and output_data.get('repair_success'):
                    timeline_label = "sketch_repair_success"
                    color = '#4CAF50'  # Green for successful repair
                else:
                    timeline_label = "sketch_repair_failed" 
                    color = '#F44336'  # Red for failed repair
            elif is_decomposition:
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
            elif is_color_resolution:
                # For color resolution, show the color name being resolved
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                color_name = parsed.get('color_name') if parsed else call.get('metadata', {}).get('color_name')
                if color_name:
                    timeline_label = f"color({color_name})"
                else:
                    timeline_label = "color_resolution"
                color = color_scheme['color_resolution']
            else:
                # For synthesis, determine type and assign appropriate color
                parsed = call.get('llm_call', {}).get('parsed_response', {})
                description = input_data.get('description', '').lower()
                expert_type = input_data.get('expert_type') or call.get('metadata', {}).get('expert_type', '')
                
                # Check for state initialization
                is_state_init = 'init_' in call.get('name', '').lower() or 'initialize' in call.get('name', '').lower()
                
                # Detect synthesis type for appropriate coloring
                is_interaction = parsed.get('is_interaction', False) if parsed else False
                is_temporal = expert_type == 'temporal_update' or 'temporal' in expert_name.lower() or 'respawn' in expert_name.lower()
                is_utility = expert_type in ['utility', 'state_update'] or 'utility' in call.get('name', '').lower()
                is_drawing = ('draw' in description or 'trail' in description or 'glow' in description) and not is_temporal
                
                if is_state_init:
                    color = color_scheme['state_initialization']
                    timeline_label = f"init_{expert_name}" if expert_name else "state_init"
                elif is_temporal:
                    color = color_scheme['synthesis_temporal']
                elif is_utility:
                    color = color_scheme['synthesis_utility']
                elif is_drawing:
                    color = color_scheme['synthesis_drawing']
                elif is_interaction:
                    color = color_scheme['synthesis_interaction']  
                else:
                    # Default to force synthesis for regular experts
                    color = color_scheme['synthesis_force']
                
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
            
            # Move to next position for the next segment (compact timeline)
            cumulative_position += width_percent
        
        return "".join(segments)
    
    def _calculate_effective_duration(self, llm_calls: List[Dict[str, Any]]) -> float:
        """Calculate effective duration as sum of all LLM processing time (no gaps)."""
        if not llm_calls:
            return 0.0
        
        # Sum all LLM durations (compact timeline with no idle time)
        return sum(call.get('duration_ms', 0) for call in llm_calls)
    
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
            
            # Handle different types of calls (including non-LLM calls like sketch repair)
            is_decomposition = call.get('name') == 'llm_decompose' or 'decompose' in call.get('name', '').lower()
            is_state_analysis = call.get('name') == 'llm_state_analysis' or 'state_analysis' in call.get('name', '').lower()
            is_color_resolution = call.get('name') == 'llm_color_resolution' or 'color_resolution' in call.get('name', '').lower()
            is_refinement = 'refinement' in call.get('name', '').lower()
            is_sketch_repair = call.get('type') == 'sketch_repair' or 'sketch_repair' in call.get('name', '').lower()
            is_analysis_stage = 'analyze_sketch' in call.get('name', '').lower() or call.get('type') == 'analysis'
            is_implementation_stage = 'implement_refinement' in call.get('name', '').lower() or call.get('type') == 'implementation'
            
            # Check for utility/temporal expert synthesis
            expert_type = input_data.get('expert_type') or call.get('metadata', {}).get('expert_type', '')
            is_temporal = expert_type == 'temporal_update' or 'temporal' in synthesis_expert_name.lower()
            is_utility = expert_type in ['utility', 'state_update']
            is_state_init = 'init_' in call.get('name', '').lower() or 'initialize' in call.get('name', '').lower()
            is_drawing = ('draw' in input_data.get('description', '').lower() or 
                         'trail' in input_data.get('description', '').lower() or 
                         'glow' in input_data.get('description', '').lower() or
                         'visual' in input_data.get('description', '').lower()) and not is_temporal
            
            if is_decomposition:
                # This is a decomposition call
                code = ''  # Decomposition doesn't generate code
                if parsed_response and parsed_response.get('components'):
                    components = parsed_response.get('components', [])
                    
                    # Show expert types breakdown
                    expert_types = {}
                    for comp in components:
                        expert_type = comp.get("expert_type", "unknown")
                        expert_types[expert_type] = expert_types.get(expert_type, 0) + 1
                    
                    if expert_types:
                        type_summary = []
                        for etype, count in expert_types.items():
                            type_summary.append(f"{count}×{etype}")
                        expert_name = f"decompose → {len(components)} experts ({', '.join(type_summary)})"
                    else:
                        expert_name = f"decompose → {len(components)} experts"
                else:
                    expert_name = 'decompose'
            elif is_state_analysis:
                # This is a state analysis call
                code = ''  # State analysis doesn't generate code
                
                # Check if this is component-level analysis
                component_prefix = ""
                if input_data and input_data.get('description'):
                    desc = input_data.get('description', '')
                    # Check if this looks like a component description (usually shorter and specific)
                    if len(desc) < 150 and ('.' in desc or 'implementation' in desc.lower()):
                        component_prefix = "component: "
                
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
                        expert_name = f"{component_prefix}state_analysis → {'+'.join(parts)}"
                    else:
                        expert_name = f"{component_prefix}state_analysis → no states"
                else:
                    expert_name = f'{component_prefix}state_analysis'
            elif is_color_resolution:
                # This is a color resolution call
                code = ''  # Color resolution doesn't generate code
                if parsed_response:
                    color_name = parsed_response.get('color_name', 'unknown')
                    rgba_list = parsed_response.get('rgba_list', [])
                    if rgba_list and len(rgba_list) >= 3:
                        r, g, b = rgba_list[:3]
                        expert_name = f"color_resolution → '{color_name}' = rgb({r:.2f}, {g:.2f}, {b:.2f})"
                    else:
                        expert_name = f"color_resolution → '{color_name}'"
                else:
                    color_name = input_data.get('color_name', call.get('metadata', {}).get('color_name', 'unknown'))
                    expert_name = f"color_resolution → '{color_name}'"
            elif is_sketch_repair:
                # This is a sketch repair call (non-LLM event)
                code = ''  # Sketch repair doesn't have generated code in same format
                if call.get('output_data'):
                    output = call['output_data']
                    if output.get('repair_success'):
                        changes = output.get('changes_made', '')
                        expert_name = f"sketch_repair_success → {changes[:50]}..." if changes else "sketch_repair_success"
                    else:
                        error = output.get('error', '')
                        expert_name = f"sketch_repair_failed → {error[:50]}..." if error else "sketch_repair_failed"
                else:
                    expert_name = 'sketch_repair_initiated'
            elif is_analysis_stage:
                # This is a Stage 1 analysis call
                code = ''  # Analysis doesn't generate code, creates a plan
                if parsed_response:
                    plan_length = len(parsed_response.get('implementation_plan', ''))
                    errors_found = parsed_response.get('errors_found', '')
                    if plan_length > 0:
                        expert_name = f"analysis → plan ({plan_length} chars, {len(errors_found) > 0 and 'errors found' or 'no errors'})"
                    else:
                        expert_name = "analysis → planning"
                else:
                    expert_name = 'analysis'
            elif is_implementation_stage:
                # This is a Stage 2 implementation call
                code = parsed_response.get('refined_code', '') if parsed_response else ''
                if parsed_response and parsed_response.get('changes_summary'):
                    changes = parsed_response['changes_summary'][:50] + "..." if len(parsed_response['changes_summary']) > 50 else parsed_response['changes_summary']
                    expert_name = f"implementation → {changes}"
                else:
                    expert_name = 'implementation'
            elif is_refinement:
                # This is a refinement call
                code = parsed_response.get('refined_code', '') if parsed_response else ''
                changes = parsed_response.get('changes_made', '') if parsed_response else ''
                warnings = parsed_response.get('warnings', '') if parsed_response else ''
                
                # Determine refinement type
                if 'error_correction' in call.get('name', ''):
                    expert_name = f"error_correction → {changes[:50]}..." if changes else "error_correction"
                else:
                    expert_name = f"refinement → {changes[:50]}..." if changes else "refinement"
            elif is_state_init:
                # This is a state initialization call
                code = parsed_response.get('code', '') if parsed_response else ''
                expert_name = f"state_init → {synthesis_expert_name}" if synthesis_expert_name else "state_init"
            elif is_temporal:
                # This is a temporal update expert synthesis
                code = parsed_response.get('code', '') if parsed_response else ''
                expert_name = synthesis_expert_name or 'temporal_update'
                if not expert_name.startswith('temporal'):
                    expert_name = f"{expert_name} (temporal)"
            elif is_utility:
                # This is a utility expert synthesis
                code = parsed_response.get('code', '') if parsed_response else ''
                expert_name = synthesis_expert_name or 'utility_expert'
                if expert_type and expert_type not in expert_name:
                    expert_name = f"{expert_name} ({expert_type})"
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
                
                # Add expert type information if available and not already included
                if expert_type and expert_type not in expert_name:
                    expert_name += f" ({expert_type})"
                
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
                        <span class="llm-description">{html.escape(expert_name)}{'' if is_decomposition or is_state_analysis else '()'}</span>
                        <span class="llm-arrow">→</span>
                        <span class="llm-result"></span>
                    </div>
                    <div>
                        <span class="llm-duration">{duration:.0f}ms</span>
                        <span class="expand-icon">▶</span>
                    </div>
                </div>
                
                <div class="llm-content">
                    <!-- Prompt Type Badge -->
                    <div style="margin-bottom: 15px;">
                        <span style="padding: 4px 10px; background: #e3f2fd; color: #1976d2; border-radius: 4px; font-size: 12px; font-weight: 500;">
                            {prompt_type}
                        </span>
                    </div>
                    
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
                            <pre>{html.escape(llm_data.get('full_prompt', '') or (llm_data.get('system_prompt', '') + chr(10) + chr(10) + llm_data.get('user_prompt', '')) if llm_data.get('system_prompt') or llm_data.get('user_prompt') else 'No prompt data available')}</pre>
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
                    {'''<div class="collapsible-section">
                        <div class="collapsible-header">
                            <span>🔍 Parsed Response (JSON)</span>
                            <span class="expand-icon">▶</span>
                        </div>
                        <div class="collapsible-content json-content">
                            <pre>''' + (json.dumps(parsed_response, indent=2) if parsed_response else "No parsed response available") + '''</pre>
                        </div>
                    </div>''' if parsed_response else ''}
                    
                    <!-- State Analysis Details (if this is a state analysis call) -->
                    {self._generate_state_analysis_details(parsed_response, is_state_analysis, input_data) if is_state_analysis else ''}
                    
                    <!-- State Initialization Details (if this is a state init call) -->
                    {self._generate_state_init_details(parsed_response, input_data, is_state_init) if is_state_init else ''}
                    
                    <!-- Temporal/Utility Expert Details (if this is a temporal/utility synthesis) -->
                    {self._generate_temporal_utility_details(parsed_response, input_data, expert_type, is_temporal or is_utility) if (is_temporal or is_utility) else ''}
                    
                    <!-- Two-Stage Refinement Details (if this is analysis or implementation) -->
                    {self._generate_two_stage_details(parsed_response, input_data, is_analysis_stage, is_implementation_stage) if (is_analysis_stage or is_implementation_stage) else ''}
                    
                    <!-- Sketch Repair Details (if this is a sketch repair call) -->
                    {self._generate_sketch_repair_details(call.get('output_data', {}), input_data, is_sketch_repair) if is_sketch_repair else ''}
                    
                    <!-- Refinement Details (if this is a refinement call) -->
                    {self._generate_refinement_details(parsed_response, input_data, is_refinement) if is_refinement else ''}
                    
                    <!-- Decomposition Components (if applicable) -->
                    {self._generate_decomposition_section(parsed_response) if parsed_response and 'components' in parsed_response else ''}
                    
                    <!-- Generated Code (expanded by default) -->
                    {'''<div class="code-section">
                        <div class="section-title">✨ Generated Code</div>
                        <div class="code-box">
                            <button class="copy-button">Copy</button>
                            <pre>''' + formatted_code + '''</pre>
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
                                <span class="metadata-value">{
                                    'Analysis (Stage 1)' if is_analysis_stage else
                                    'Implementation (Stage 2)' if is_implementation_stage else
                                    'Temporal Update' if is_temporal else
                                    'Utility Expert' if is_utility else
                                    'State Init' if is_state_init else
                                    'Decomposition' if is_decomposition else
                                    'State Analysis' if is_state_analysis else
                                    'Color Resolution' if is_color_resolution else
                                    'Refinement' if is_refinement else
                                    'Interaction' if parsed_response and parsed_response.get('is_interaction') else
                                    'Force' if not (is_drawing or is_temporal or is_utility) else
                                    'Drawing' if 'draw' in expert_name.lower() or 'trail' in expert_name.lower() else
                                    'Single'
                                }</span>
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
                    <strong>Components:</strong> <span style="color: {'#4CAF50' if len(components) == 1 else '#FF9800'}">
                        {len(components)} expert{'s' if len(components) != 1 else ''}
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
    
    def _generate_state_analysis_details(self, parsed_response: Dict[str, Any], is_state_analysis: bool, input_data: Dict[str, Any] = None) -> str:
        """Generate detailed HTML section for state analysis responses."""
        if not is_state_analysis or not parsed_response:
            return ''
        
        section = '''
        <div class="state-analysis-section">
            <div class="section-title">🧬 State Analysis Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show what we're analyzing if available
        if input_data and input_data.get('description'):
            description = input_data.get('description', '')
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: var(--bg-secondary); border-radius: 4px;">
                    <strong>Analyzing for:</strong> {html.escape(description[:200])}{'...' if len(description) > 200 else ''}
                </div>
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
    
    def _generate_state_init_details(self, parsed_response: Dict[str, Any], input_data: Dict[str, Any], is_state_init: bool) -> str:
        """Generate detailed HTML section for state initialization."""
        if not is_state_init:
            return ''
        
        section = '''
        <div class="state-init-section">
            <div class="section-title">🔧 State Initialization Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show what states are being initialized
        if input_data.get('states_to_init'):
            states = input_data['states_to_init']
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>States Being Initialized:</strong> 
                    <span style="color: #00BCD4;">{len(states)} states</span>
                </div>
                <ul style="margin-left: 20px;">
            '''
            for state in states:
                section += f'<li>{html.escape(str(state))}</li>'
            section += '</ul>'
        
        # Show initialization code if available
        if parsed_response and parsed_response.get('code'):
            section += f'''
                <div style="margin-top: 15px;">
                    <strong>Initialization Code:</strong>
                    <pre style="background-color: var(--bg-secondary); padding: 10px; border-radius: 4px; overflow-x: auto;">
{html.escape(parsed_response['code'][:500])}{'...' if len(parsed_response['code']) > 500 else ''}
                    </pre>
                </div>
            '''
        
        section += '</div></div>'
        return section
    
    def _generate_temporal_utility_details(self, parsed_response: Dict[str, Any], input_data: Dict[str, Any], expert_type: str, is_temporal_utility: bool) -> str:
        """Generate detailed HTML section for temporal/utility expert synthesis."""
        if not is_temporal_utility:
            return ''
        
        section = '''
        <div class="temporal-utility-section">
            <div class="section-title">⏰ Temporal/Utility Expert Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show expert type
        section += f'''
            <div style="margin-bottom: 15px;">
                <strong>Expert Type:</strong> 
                <span style="color: #FFC107;">{html.escape(expert_type or 'Unknown')}</span>
            </div>
        '''
        
        # Show description
        if input_data.get('description'):
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Behavior Description:</strong><br>
                    <span style="font-style: italic; color: var(--text-secondary);">"{html.escape(input_data['description'])}"</span>
                </div>
            '''
        
        # Show which states this affects
        if input_data.get('affected_states'):
            states = input_data['affected_states']
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Affected States:</strong>
                    <ul style="margin-left: 20px;">
            '''
            for state in states:
                section += f'<li>{html.escape(str(state))}</li>'
            section += '</ul></div>'
        
        # Show update expression if temporal
        if expert_type == 'temporal_update' and input_data.get('update_expression'):
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: #FFF8E1; border-left: 3px solid #FFC107; border-radius: 4px;">
                    <strong>Update Expression:</strong><br>
                    <code style="font-family: monospace; color: #FF6F00;">
                        {html.escape(input_data['update_expression'])}
                    </code>
                </div>
            '''
        
        section += '</div></div>'
        return section
    
    def _generate_refinement_details(self, parsed_response: Dict[str, Any], input_data: Dict[str, Any], is_refinement: bool) -> str:
        """Generate detailed HTML section for refinement responses."""
        if not is_refinement or not parsed_response:
            return ''
        
        section = '''
        <div class="refinement-section">
            <div class="section-title">🔧 Refinement Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show changes made
        changes_made = parsed_response.get('changes_made', '')
        if changes_made:
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Changes Applied:</strong> 
                    <span style="color: #4CAF50;">{html.escape(changes_made)}</span>
                </div>
            '''
        
        # Show warnings
        warnings = parsed_response.get('warnings', '')
        if warnings:
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: #FFF8E1; border-left: 3px solid #FF9800; border-radius: 4px;">
                    <strong style="color: #FF9800;">⚠️ Warnings:</strong><br>
                    {html.escape(warnings)}
                </div>
            '''
        
        # Show refinement request
        refinement_request = input_data.get('refinement_request', '')
        if refinement_request:
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Original Request:</strong><br>
                    <span style="font-style: italic; color: var(--text-secondary);">"{html.escape(refinement_request)}"</span>
                </div>
            '''
        
        # Show error info if this was an error correction
        has_error = input_data.get('has_error', False)
        if has_error:
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: #FFEBEE; border-left: 3px solid #F44336; border-radius: 4px;">
                    <strong style="color: #F44336;">Error Fixed:</strong><br>
                    This refinement was triggered by an error in the sketch execution.
                </div>
            '''
        
        section += '</div></div>'
        return section
    
    def _generate_sketch_repair_details(self, output_data: Dict[str, Any], input_data: Dict[str, Any], is_sketch_repair: bool) -> str:
        """Generate detailed HTML section for sketch repair events."""
        if not is_sketch_repair:
            return ''
        
        section = '''
        <div class="sketch-repair-section">
            <div class="section-title">🩹 Sketch Repair Details</div>
            <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
        '''
        
        # Show trigger information
        if input_data.get('trigger'):
            trigger = input_data['trigger']
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Trigger:</strong> 
                    <span style="color: #FF9800;">{html.escape(trigger)}</span>
                </div>
            '''
        
        # Show error logs that prompted the repair
        if input_data.get('error_logs'):
            error_logs = input_data['error_logs']
            section += f'''
                <div style="margin-bottom: 15px;">
                    <strong>Error Logs:</strong>
                    <pre style="background-color: var(--bg-secondary); padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 200px;">
{html.escape(error_logs[:1000])}{'...' if len(error_logs) > 1000 else ''}
                    </pre>
                </div>
            '''
        
        # Show repair results
        if output_data.get('repair_success'):
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: #E8F5E9; border-left: 3px solid #4CAF50; border-radius: 4px;">
                    <strong style="color: #4CAF50;">✅ Repair Successful</strong>
                </div>
            '''
            
            if output_data.get('changes_made'):
                changes = output_data['changes_made']
                section += f'''
                    <div style="margin-bottom: 15px;">
                        <strong>Changes Applied:</strong> 
                        <span style="color: #4CAF50;">{html.escape(changes)}</span>
                    </div>
                '''
            
            if output_data.get('code_lines_changed'):
                lines_changed = output_data['code_lines_changed']
                section += f'''
                    <div style="margin-bottom: 15px;">
                        <strong>Lines Modified:</strong> 
                        <span style="color: #2196F3;">{lines_changed}</span>
                    </div>
                '''
            
            if output_data.get('final_code_length'):
                code_length = output_data['final_code_length']
                section += f'''
                    <div style="margin-bottom: 15px;">
                        <strong>Final Code Size:</strong> 
                        <span style="color: var(--text-secondary);">{code_length} characters</span>
                    </div>
                '''
        else:
            section += f'''
                <div style="margin-bottom: 15px; padding: 10px; background-color: #FFEBEE; border-left: 3px solid #F44336; border-radius: 4px;">
                    <strong style="color: #F44336;">❌ Repair Failed</strong>
                </div>
            '''
            
            if output_data.get('error'):
                error = output_data['error']
                section += f'''
                    <div style="margin-bottom: 15px;">
                        <strong>Error:</strong> 
                        <span style="color: #F44336;">{html.escape(error)}</span>
                    </div>
                '''
            
            if output_data.get('exception'):
                section += f'''
                    <div style="margin-bottom: 15px;">
                        <strong>Type:</strong> 
                        <span style="color: var(--text-secondary);">Exception during repair process</span>
                    </div>
                '''
        
        section += '</div></div>'
        return section
    
    def _generate_two_stage_details(self, parsed_response: Dict[str, Any], input_data: Dict[str, Any], is_analysis: bool, is_implementation: bool) -> str:
        """Generate detailed HTML section for two-stage refinement processes."""
        if not (is_analysis or is_implementation):
            return ''
        
        if is_analysis:
            section = '''
            <div class="analysis-stage-section">
                <div class="section-title">🔍 Stage 1: Analysis Details</div>
                <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
            '''
            
            # Show what's being analyzed
            if input_data.get('description'):
                description = input_data['description']
                section += f'''
                    <div style="margin-bottom: 15px; padding: 10px; background-color: var(--bg-secondary); border-radius: 4px;">
                        <strong>Analyzing Sketch for:</strong> {html.escape(description[:200])}{'...' if len(description) > 200 else ''}
                    </div>
                '''
            
            # Show analysis results
            if parsed_response:
                implementation_plan = parsed_response.get('implementation_plan', '')
                errors_found = parsed_response.get('errors_found', '')
                architectural_needs = parsed_response.get('architectural_needs', '')
                
                if implementation_plan:
                    plan_preview = implementation_plan[:300] + "..." if len(implementation_plan) > 300 else implementation_plan
                    section += f'''
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 10px 0; color: var(--accent);">Implementation Plan</h4>
                            <div style="padding: 10px; background-color: var(--bg-secondary); border-left: 3px solid var(--accent); border-radius: 4px;">
                                {html.escape(plan_preview)}
                            </div>
                        </div>
                    '''
                
                if errors_found:
                    section += f'''
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 10px 0; color: #F44336;">Errors Found</h4>
                            <div style="padding: 10px; background-color: #FFEBEE; border-left: 3px solid #F44336; border-radius: 4px;">
                                {html.escape(errors_found)}
                            </div>
                        </div>
                    '''
                
                if architectural_needs:
                    section += f'''
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 10px 0; color: #FF9800;">Architectural Needs</h4>
                            <div style="padding: 10px; background-color: #FFF3E0; border-left: 3px solid #FF9800; border-radius: 4px;">
                                {html.escape(architectural_needs)}
                            </div>
                        </div>
                    '''
            
            section += '</div></div>'
        
        elif is_implementation:
            section = '''
            <div class="implementation-stage-section">
                <div class="section-title">⚙️ Stage 2: Implementation Details</div>
                <div style="padding: 15px; background-color: var(--code-bg); border-radius: 4px;">
            '''
            
            # Show what's being implemented
            if input_data.get('description'):
                description = input_data['description']
                section += f'''
                    <div style="margin-bottom: 15px; padding: 10px; background-color: var(--bg-secondary); border-radius: 4px;">
                        <strong>Implementing for:</strong> {html.escape(description[:200])}{'...' if len(description) > 200 else ''}
                    </div>
                '''
            
            # Show implementation results
            if parsed_response:
                changes_summary = parsed_response.get('changes_summary', '')
                refined_code = parsed_response.get('refined_code', '')
                
                if changes_summary:
                    section += f'''
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 10px 0; color: var(--accent);">Changes Summary</h4>
                            <div style="padding: 10px; background-color: #E8F5E9; border-left: 3px solid #4CAF50; border-radius: 4px;">
                                {html.escape(changes_summary)}
                            </div>
                        </div>
                    '''
                
                if refined_code:
                    code_lines = len(refined_code.split('\n'))
                    code_chars = len(refined_code)
                    section += f'''
                        <div style="margin-bottom: 15px;">
                            <h4 style="margin: 10px 0; color: var(--accent);">Refined Code Stats</h4>
                            <div style="display: flex; gap: 20px; color: var(--text-secondary);">
                                <span>Lines: {code_lines}</span>
                                <span>Characters: {code_chars}</span>
                            </div>
                        </div>
                    '''
            
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