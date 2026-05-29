r"""LLM-driven refiner for the SuperCollider companion patch.

Mirrors ``SketchRefiner.refine_sketch`` but targets the ``.scd`` companion
that ``emit_companion`` (in ``tolvera.llm.sc.emitter``) generates alongside
each Python sketch. The Textual UI calls this when the user is focused on
the SC editor tab — refinements never re-run the full Bedrock orchestrator
pipeline, they just rewrite the ``.scd``.

The system prompt embeds the SC patch contract (port 5000, OSC address
template, available synths, mapping ranges) so the model knows what to
preserve. Post-process step: strip markdown fences, undo over-escaped
backslashes (Bedrock sometimes double-escapes ``\identifier`` symbols),
then validate that every species id still has its three OSCdef bindings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..debug.tracing import LLMCallData, get_collector
from ..prompts.prompt_loader import get_prompt_loader
from .conversation_manager import ConversationManager
from .llm_factory import ModelFactory


class SCRefinementResponse(BaseModel):
    """Structured output from the SC refinement agent."""

    refined_code: str = Field(description="The complete refined .scd patch content.")
    changes_made: str = Field(description="Brief summary of what was changed and why.")
    warnings: Optional[str] = Field(
        None, description="Any concerns about the refinement (e.g. dropped responder)."
    )


class SCRefiner:
    """Refine an existing SuperCollider companion patch via LLM."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name, api_key)
        self.provider = ModelFactory.get_provider_for_model(model_name)
        self.refinement_agent: Optional[Agent] = None
        self.conversation_manager = ConversationManager(
            max_history_length=10, max_context_tokens=2000
        )

    async def _create_agent(self) -> Agent:
        # Load the SC system prompt verbatim (no template substitution): the
        # prompt itself is self-contained and references no Tölvera context
        # patterns, so passing kwargs would just be brittle.
        loader = get_prompt_loader()
        system_prompt = loader.load_prompt("sc_refinement/system.txt")
        return Agent(
            self.model,
            output_type=SCRefinementResponse,
            output_retries=2,
            system_prompt=system_prompt,
            # Returns a full .scd patch; give reasoning-heavy models room so the
            # tool call isn't truncated into a validation failure.
            model_settings={"max_tokens": 12000},
        )

    async def refine_sc_patch(
        self,
        scd_code: str,
        request: str,
        species_ids: Optional[Iterable[int]] = None,
        error_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a natural-language refinement to a ``.scd`` patch.

        Args:
            scd_code: Current patch contents.
            request: User's natural-language change request.
            species_ids: Ids the Python sketch is still broadcasting for. Used
                to validate that the refined patch keeps the OSC contract.
            error_info: Optional sclang error log to fold into the prompt.

        Returns:
            ``{success, refined_code, changes_made, warnings, error}``.
        """
        if self.refinement_agent is None:
            self.refinement_agent = await self._create_agent()

        species_list: List[int] = list(species_ids) if species_ids is not None else []
        species_repr = ", ".join(str(s) for s in species_list) if species_list else "(unknown)"

        loader = get_prompt_loader()
        prompt = loader.load_prompt(
            "sc_refinement/user.txt",
            request=request,
            species_ids=species_repr,
            conversation_context=self.conversation_manager.get_conversation_context() or "(none)",
            error_section=(error_info or "(none)"),
            scd_code=scd_code,
        )

        collector = get_collector()
        with collector.trace_node(
            "refine_sc_patch", "refinement", request=request, has_error=bool(error_info)
        ) as node:
            try:
                with collector.trace_node(
                    "sc_refinement_llm_call", "llm_call", model=self.model_name
                ) as llm_node:
                    result = await self.refinement_agent.run(prompt)
                    refined_code = self._sanitize_sc_code(result.output.refined_code)
                    changes_made = result.output.changes_made
                    warnings = result.output.warnings

                    validation = self._validate_sc_patch(refined_code, species_list)
                    if validation:
                        warnings = (warnings + "; " if warnings else "") + validation

                    if llm_node:
                        llm_node.llm_call = LLMCallData(
                            model=self.model_name,
                            provider=self.provider,
                            user_prompt=prompt,
                            system_prompt="SC refinement system prompt",
                            full_prompt=prompt,
                            parsed_response={
                                "changes_made": changes_made,
                                "warnings": warnings,
                            },
                        )

                self.conversation_manager.add_conversation_entry(
                    user_request=request,
                    agent_response_summary=changes_made,
                    interaction_type="sc_refinement",
                    success=True,
                    pydantic_messages=(
                        result.new_messages() if hasattr(result, "new_messages") else None
                    ),
                )

                return {
                    "success": True,
                    "refined_code": refined_code,
                    "changes_made": changes_made,
                    "warnings": warnings,
                }
            except Exception as exc:
                if node:
                    node.set_error(str(exc))
                self.conversation_manager.add_conversation_entry(
                    user_request=request,
                    agent_response_summary=f"Failed: {exc}",
                    interaction_type="sc_refinement",
                    success=False,
                    error_info=str(exc),
                )
                return {
                    "success": False,
                    "error": str(exc),
                    "refined_code": scd_code,
                    "changes_made": "No changes due to error",
                    "warnings": f"SC refinement failed: {exc}",
                }

    def clear_conversation_history(self) -> None:
        self.conversation_manager.clear_history()

    @staticmethod
    def _sanitize_sc_code(code: str) -> str:
        """Undo common LLM-output quirks: markdown fences and over-escaping.

        Bedrock Claude occasionally returns ``\\\\freq`` instead of ``\\freq``
        when emitting SC symbols inside a JSON-typed output field. Form-feeds
        (``\\u000c``) sometimes leak through too. Strip both back to canonical
        SuperCollider syntax.
        """
        code = code.strip()

        # Drop a leading code fence if present.
        for fence in ("```supercollider", "```sclang", "```scd", "```"):
            if code.startswith(fence):
                code = code[len(fence) :].lstrip("\n")
                break

        # Drop a trailing code fence.
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3].rstrip()

        # Drop trailing triple-quotes if the model wrapped it like a docstring.
        for trail in ('"""', "'''"):
            if code.rstrip().endswith(trail):
                code = code.rstrip()[: -len(trail)].rstrip()

        # Fix double-escaped backslashes (\\freq → \freq) but leave a literal
        # ``\\\\`` (rare in SC) intact by only collapsing when followed by an
        # identifier character.
        import re

        code = re.sub(r"\\\\(?=[A-Za-z_])", r"\\", code)

        # Remove stray form-feed contamination from JSON encoding.
        code = code.replace("\x0c", "").replace("\\u000c", "")

        return code.strip() + "\n"

    @staticmethod
    def _validate_sc_patch(code: str, species_ids: List[int]) -> Optional[str]:
        """Lightweight structural check on the refined patch.

        Returns a warning string if the OSC contract appears broken; ``None``
        otherwise. This NEVER hard-fails the refinement — it lets the user
        decide whether to accept a patch that may drop sound for a species.
        """
        issues: List[str] = []

        if "s.waitForBoot" not in code:
            issues.append("missing s.waitForBoot block")
        if "ServerQuit.add" not in code:
            issues.append("missing ServerQuit cleanup")
        if "~tolveraSynths" not in code:
            issues.append("missing ~tolveraSynths registry")

        for sid in species_ids:
            for metric in ("x", "y", "vel"):
                addr = f"/metrics/{sid}/{metric}"
                if addr not in code:
                    issues.append(f"dropped OSCdef for {addr}")

        # OSCdef without an explicit recvPort listens on sclang's default
        # port (57120), not the patch's port 5000 — silent species despite
        # everything looking fine in the code. Catch that early.
        if "/metrics/" in code and ", 5000)" not in code:
            issues.append(
                "OSCdef(s) missing recvPort 5000 — packets sent by the Python "
                "sketch won't reach the synths (silent audio)"
            )

        if not issues:
            return None
        return "validation: " + "; ".join(issues)
