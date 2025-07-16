import logging
import re
from typing import Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class BoundaryMode(Enum):
    NONE = "none"  # Particles just stop at screen edges
    WRAP = "wrap"  # Particles wrap around screen edges (toroidal)
    BOUNCE = "bounce"  # Particles bounce off screen edges
    ABSORB = "absorb"  # Particles are absorbed/removed at edges
    

class BoundaryManager:
    
    def __init__(self):
        # Could be implemented in an LLM call, but I thought this might work a bit quicker than waiting on a model to get this info
        self.boundary_keywords = {
            BoundaryMode.WRAP: [
                r'\bwrap\b', r'\bwrapping\b', r'\btoroidal\b', r'\btorus\b',
                r'\bloop\b', r'\blooping\b', r'\bcontinuous space\b',
                r'\bperiodic boundary\b', r'\bperiodic boundaries\b'
            ],
            BoundaryMode.BOUNCE: [
                r'\bbounce\b', r'\bbouncing\b', r'\breflect\b', r'\breflection\b',
                r'\bwalls?\b', r'\bbarrier\b', r'\bcollide with edges?\b',
                r'\belastic collision\b', r'\bbounce off\b'
            ],
            BoundaryMode.ABSORB: [
                r'\babsorb\b', r'\babsorbed?\b', r'\bremove at edges?\b',
                r'\bdisappear\b', r'\bdelete at boundary\b', r'\bkill at edges?\b',
                r'\bdestroy at boundary\b', r'\bexit screen\b'
            ],
            BoundaryMode.NONE: [
                r'\bno boundary\b', r'\bunbounded\b', r'\binfinite space\b',
                r'\boff-screen\b', r'\boffscreen\b', r'\bno walls?\b',
                r'\bopen space\b', r'\bcan leave\b', r'\bescape\b'
            ]
        }
    
    def analyze_boundary_requirements(self, description: str) -> Tuple[BoundaryMode, float]:
        desc_lower = description.lower()
        
        matches = {}
        # Again, could put the LLM call from poe_ollama here too if needed
        for mode, patterns in self.boundary_keywords.items():
            count = sum(1 for pattern in patterns if re.search(pattern, desc_lower))
            if count > 0:
                matches[mode] = count
        
        if matches:
            best_mode = max(matches, key=matches.get)
            confidence = min(1.0, matches[best_mode] / 2.0)  # Cap at 1.0
            logger.info(f"Detected boundary mode: {best_mode.value} (confidence: {confidence:.2f})")
            return best_mode, confidence
        
        logger.info("No boundary behavior detected, defaulting to NONE")
        return BoundaryMode.NONE, 0.5
    
    def get_boundary_code(self, mode: BoundaryMode, use_new_pos: bool = True) -> str:
        if mode == BoundaryMode.NONE:
            # No boundary handling needed
            return ""
        
        elif mode == BoundaryMode.WRAP:
            if use_new_pos:
                return """            # Wrap around boundaries
            if new_pos[0] < 0:
                new_pos[0] += tv.x
            elif new_pos[0] > tv.x:
                new_pos[0] -= tv.x
                
            if new_pos[1] < 0:
                new_pos[1] += tv.y
            elif new_pos[1] > tv.y:
                new_pos[1] -= tv.y
            
            tv.p.field[i].pos = new_pos"""
            else:
                return """            # Wrap particles around screen edges
            if tv.p.field[i].pos[0] > tv.x:
                tv.p.field[i].pos[0] = 0.0
            elif tv.p.field[i].pos[0] < 0.0:
                tv.p.field[i].pos[0] = tv.x
                
            if tv.p.field[i].pos[1] > tv.y:
                tv.p.field[i].pos[1] = 0.0
            elif tv.p.field[i].pos[1] < 0.0:
                tv.p.field[i].pos[1] = tv.y"""
        
        elif mode == BoundaryMode.BOUNCE:
            if use_new_pos:
                return """            # Bounce off boundaries
            if new_pos[0] < 0:
                new_pos[0] = -new_pos[0]
                tv.p.field[i].vel[0] = -tv.p.field[i].vel[0] * 0.8  # Energy loss
            elif new_pos[0] > tv.x:
                new_pos[0] = 2 * tv.x - new_pos[0]
                tv.p.field[i].vel[0] = -tv.p.field[i].vel[0] * 0.8
                
            if new_pos[1] < 0:
                new_pos[1] = -new_pos[1]
                tv.p.field[i].vel[1] = -tv.p.field[i].vel[1] * 0.8
            elif new_pos[1] > tv.y:
                new_pos[1] = 2 * tv.y - new_pos[1]
                tv.p.field[i].vel[1] = -tv.p.field[i].vel[1] * 0.8
            
            tv.p.field[i].pos = new_pos"""
            else:
                return """            # Bounce particles off screen edges
            if tv.p.field[i].pos[0] < 0:
                tv.p.field[i].pos[0] = -tv.p.field[i].pos[0]
                tv.p.field[i].vel[0] = -tv.p.field[i].vel[0] * 0.8
            elif tv.p.field[i].pos[0] > tv.x:
                tv.p.field[i].pos[0] = 2 * tv.x - tv.p.field[i].pos[0]
                tv.p.field[i].vel[0] = -tv.p.field[i].vel[0] * 0.8
                
            if tv.p.field[i].pos[1] < 0:
                tv.p.field[i].pos[1] = -tv.p.field[i].pos[1]
                tv.p.field[i].vel[1] = -tv.p.field[i].vel[1] * 0.8
            elif tv.p.field[i].pos[1] > tv.y:
                tv.p.field[i].pos[1] = 2 * tv.y - tv.p.field[i].pos[1]
                tv.p.field[i].vel[1] = -tv.p.field[i].vel[1] * 0.8"""
        
        elif mode == BoundaryMode.ABSORB:
            if use_new_pos:
                return """            # Check boundaries and deactivate if outside
            if new_pos[0] < 0 or new_pos[0] > tv.x or new_pos[1] < 0 or new_pos[1] > tv.y:
                tv.p.field[i].active = 0.0
            else:
                tv.p.field[i].pos = new_pos"""
            else:
                return """            # Deactivate particles that go off-screen
            if tv.p.field[i].pos[0] < 0 or tv.p.field[i].pos[0] > tv.x or tv.p.field[i].pos[1] < 0 or tv.p.field[i].pos[1] > tv.y:
                tv.p.field[i].active = 0.0"""
        
        return ""