"""
CSV logging for PoE LLM interactions.
"""

import csv
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PoECSVLogger:
    
    def __init__(self, log_file: str = "poe_llm_interactions.csv"):
        self.log_file = log_file
        self.fieldnames = [
            'timestamp',
            'type',  # 'expert' or 'kernel'
            'user_description',
            'llm_prompt',
            'raw_llm_response',
            'extracted_code',
            'success',
            'errors',
            'model_name',
            'expert_name',
            'synthesis_time_ms',
            'included_experts',  # For kernel synthesis: list of expert names
            'expert_codes'  # For kernel synthesis: JSON dict of expert_name: code
        ]
        
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
        
        logger.info(f"Initialized PoE CSV logger: {log_file}")
    
    def log_synthesis_attempt(self, 
                            user_description: str,
                            llm_prompt: str,
                            raw_response: str,
                            extracted_code: str,
                            success: bool,
                            errors: list,
                            model_name: str,
                            expert_name: Optional[str],
                            synthesis_time_ms: float,
                            synthesis_type: str = "expert",
                            included_experts: Optional[list] = None,
                            expert_codes: Optional[dict] = None):
        row = {
            'timestamp': datetime.now().isoformat(),
            'type': synthesis_type,
            'user_description': user_description,
            'llm_prompt': llm_prompt,
            'raw_llm_response': raw_response,
            'extracted_code': extracted_code,
            'success': success,
            'errors': '|'.join(errors) if errors else '',
            'model_name': model_name,
            'expert_name': expert_name or '',
            'synthesis_time_ms': round(synthesis_time_ms, 2),
            'included_experts': json.dumps(included_experts) if included_experts else '',
            'expert_codes': json.dumps(expert_codes) if expert_codes else ''
        }
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                writer.writerow(row)
            
            logger.debug(f"Logged synthesis attempt for '{user_description}' - Success: {success}")
        except Exception as e:
            logger.error(f"Failed to write to CSV log: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.log_file):
            return {"error": "Log file not found"}
        
        total_attempts = 0
        successful = 0
        failed = 0
        models_used = set()
        common_errors = {}
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_attempts += 1
                    if row['success'].lower() == 'true':
                        successful += 1
                    else:
                        failed += 1
                        
                        # Track error types
                        if row['errors']:
                            for error in row['errors'].split('|'):
                                error = error.strip()
                                if error:
                                    common_errors[error] = common_errors.get(error, 0) + 1
                    
                    if row['model_name']:
                        models_used.add(row['model_name'])
            
            return {
                "total_attempts": total_attempts,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_attempts * 100) if total_attempts > 0 else 0,
                "models_used": list(models_used),
                "common_errors": dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        except Exception as e:
            return {"error": f"Failed to read log file: {e}"}


_global_logger = None

def get_logger(log_file: str = "poe_llm_interactions.csv") -> PoECSVLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = PoECSVLogger(log_file)
    return _global_logger