"""Trace collection and analysis for synthesis debugging.

This module provides comprehensive tracing functionality for tracking
the synthesis pipeline, including LLM calls, timing information, and
error tracking.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps
import threading

import logging
logger = logging.getLogger(__name__)


@dataclass
class TraceNode:
    """Node in the trace tree representing a single operation.
    
    Attributes:
        id (str): Unique identifier for the node.
        type (str): Type of operation (synthesis, llm_call, etc.).
        name (str): Name of the operation.
        timestamp (datetime): When the operation started.
        start_time (float): Start time in seconds.
        end_time (Optional[float]): End time in seconds.
        duration_ms (Optional[float]): Duration in milliseconds.
        input_data (Dict): Input parameters.
        output_data (Dict): Output results.
        metadata (Dict): Additional metadata.
        llm_call (Optional[LLMCallData]): LLM call details if applicable.
        parent_id (Optional[str]): Parent node ID.
        children (List[TraceNode]): Child nodes.
        status (str): Current status (running, success, error).
        error (Optional[str]): Error message if failed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_call: Optional['LLMCallData'] = None
    parent_id: Optional[str] = None
    children: List['TraceNode'] = field(default_factory=list)
    status: str = "running"
    error: Optional[str] = None
    
    def complete(self, status: str = "success", error: Optional[str] = None):
        """Mark the node as complete.
        
        Args:
            status (str): Final status. Defaults to "success".
            error (Optional[str]): Error message if failed.
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error
    
    def add_child(self, child: 'TraceNode') -> 'TraceNode':
        """Add a child node to this node.
        
        Args:
            child (TraceNode): Child node to add.
            
        Returns:
            TraceNode: The added child node.
        """
        child.parent_id = self.id
        self.children.append(child)
        return child
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation.
        """
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LLMCallData:
    model: str = ""
    provider: str = ""
    
    system_prompt: str = ""
    user_prompt: str = ""
    full_prompt: str = ""
    raw_response: str = ""
    parsed_response: Optional[Any] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    api_call_ms: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TraceCollector:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.traces: Dict[str, TraceNode] = {}
        self.current_trace: Optional[TraceNode] = None
        self._trace_stack: List[TraceNode] = []
        self.enabled = True
        self.console_output = True
        self.capture_llm_content = True
        self.on_node_start: Optional[Callable[[TraceNode], None]] = None
        self.on_node_complete: Optional[Callable[[TraceNode], None]] = None
    
    def start_trace(self, name: str, trace_type: str = "synthesis") -> TraceNode:
        if not self.enabled:
            return TraceNode()
        
        trace = TraceNode(
            type=trace_type,
            name=name,
            input_data={"description": name}
        )
        
        self.traces[trace.id] = trace
        self.current_trace = trace
        self._trace_stack = [trace]
        
        if self.on_node_start:
            self.on_node_start(trace)
        
        logger.info(f"Started trace: {name} (ID: {trace.id})")
        return trace
    
    @contextmanager
    def trace_node(self, name: str, node_type: str, **kwargs):
        if not self.enabled or not self.current_trace:
            yield None
            return
        
        parent = self._trace_stack[-1] if self._trace_stack else self.current_trace
        node = TraceNode(
            type=node_type,
            name=name,
            input_data=kwargs
        )
        
        parent.add_child(node)
        self._trace_stack.append(node)
        
        if self.on_node_start:
            self.on_node_start(node)
        
        try:
            yield node
            node.complete("success")
        except Exception as e:
            node.complete("error", str(e))
            raise
        finally:
            self._trace_stack.pop()
            if self.on_node_complete:
                self.on_node_complete(node)
    
    def log_llm_call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        **metadata
    ) -> LLMCallData:
        if not self.enabled or not self._trace_stack:
            return LLMCallData()
        
        current_node = self._trace_stack[-1]
        
        llm_data = LLMCallData(
            model=model,
            provider=self._get_provider(model),
            system_prompt=system_prompt if self.capture_llm_content else "[REDACTED]",
            user_prompt=user_prompt if self.capture_llm_content else "[REDACTED]",
            full_prompt=f"{system_prompt}\n\n{user_prompt}" if self.capture_llm_content else "[REDACTED]",
            raw_response=response if self.capture_llm_content else "[REDACTED]",
            **metadata
        )
        
        current_node.llm_call = llm_data
        current_node.metadata.update({
            "model": model,
            "prompt_length": len(system_prompt + user_prompt),
            "response_length": len(response)
        })
        
        return llm_data
    
    def log_routing_decision(self, decision: str, reason: str, confidence: float = 1.0):
        if not self.enabled or not self._trace_stack:
            return
        
        current_node = self._trace_stack[-1]
        current_node.metadata.update({
            "routing_decision": decision,
            "routing_reason": reason,
            "routing_confidence": confidence
        })
    
    def get_trace(self, trace_id: str) -> Optional[TraceNode]:
        return self.traces.get(trace_id)
    
    def get_current_trace(self) -> Optional[TraceNode]:
        return self.current_trace
    
    def export_trace(self, trace_id: str, format: str = "json") -> str:
        trace = self.get_trace(trace_id)
        if not trace:
            return ""
        
        if format == "json":
            return json.dumps(trace.to_dict(), indent=2, default=str)
        elif format == "mermaid":
            from .diagram_generator import generate_mermaid_diagram
            return generate_mermaid_diagram(trace)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def clear_traces(self):
        self.traces.clear()
        self.current_trace = None
        self._trace_stack.clear()
    
    def _get_provider(self, model: str) -> str:
        if "gemini" in model.lower():
            return "gemini"
        elif "claude" in model.lower():
            return "anthropic"
        elif "gpt" in model.lower():
            return "openai"
        else:
            return "ollama"


_collector = TraceCollector()


def get_collector() -> TraceCollector:
    return _collector


def trace_function(node_type: str = "function"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_collector()
            if not collector.enabled:
                return func(*args, **kwargs)
            
            name = func.__name__
            input_data = {
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200]
            }
            
            with collector.trace_node(name, node_type, **input_data) as node:
                result = func(*args, **kwargs)
                if node:
                    node.output_data = {
                        "result_type": type(result).__name__,
                        "result_preview": str(result)[:200]
                    }
                return result
        
        return wrapper
    return decorator


trace_synthesis = trace_function("synthesis")
trace_decomposition = trace_function("decomposition")
trace_routing = trace_function("routing")
trace_parsing = trace_function("parsing")