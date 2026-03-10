import re
import uuid
from typing import Tuple, List, Dict, Any, Optional


# ── System prompt ─────────────────────────────────────────────────────────────
# Optimised for Qwen3.5-4B:
#   - Thinking mode is DISABLED on the server side (enable_thinking=False).
#   - Very explicit one-line format rule so the model doesn't drift into prose.
#   - Few-shot examples in the prompt anchor the output format.
#   - Explicit list of termination conditions prevents the model from stalling.
SYSTEM_PROMPT = """\
You are a shopping agent operating in a text-only WebShop environment.
Your job: follow the instruction, browse the shop, and buy exactly the right product.

## Action Space
- search[<keywords>]       — submit a keyword search
- click[<exact text>]      — click an item or button using its EXACT text from the page
                             (product IDs like B09QQWRW2M, options like "large", nav like "buy now")

## Output Format  ← YOU MUST FOLLOW THIS EXACTLY, EVERY TURN
Thought: <one-sentence reasoning; cite earlier obs with [[obs:N]] if relevant>
Action: <exactly one action, no extra text>

## Examples of correct output
Thought: I need to find a slim-fit jumpsuit in green; I'll start with a broad search.
Action: search[women slim fit jumpsuit green]

Thought: [[obs:3]] shows product B07XYZABC which looks close; I'll click to inspect.
Action: click[B07XYZABC]

Thought: Size and color match; I'll complete the purchase.
Action: click[buy now]

## Rules
- Output ONLY the Thought + Action block. No extra commentary.
- NEVER repeat an action that already returned the same page.
- When you see "Thank you for shopping", the episode is over — output nothing.
- If you are unsure, try a more specific search rather than clicking randomly.
"""

FORMAT_RETRY_PROMPT = """\
Your last response did not contain a valid Action line.
Please respond with ONLY:
Thought: <brief reasoning>
Action: <one action>
"""


class ReActWebShopAgent:
    """
    Vanilla ReAct agent for WebShop/WebStore.

    LLM backend: any object with a .chat(messages) -> str method.
    In production this is OpenAIChatClient pointing at the local
    Qwen3.5-4B Flask server (HTTP POST to /v1/chat/completions).
    """

    def __init__(
        self,
        llm_client,
        logger,
        policy_version: str = "react_v1",
        max_format_retries: int = 2,
        success_reward_threshold: float = 1.0,
    ):
        self.llm = llm_client
        self.logger = logger
        self.policy_version = policy_version
        self.max_format_retries = max_format_retries
        self.success_reward_threshold = success_reward_threshold

    # ── Parsing ───────────────────────────────────────────────────────────────

    def parse_response(self, text: str) -> Tuple[str, str, List[int]]:
        """Return (thought, action, ref_ids) from a raw LLM output."""
        thought, action = "", ""

        t_match = re.search(r'Thought:(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'Action:\s*(.*)', text, re.IGNORECASE)

        if t_match:
            thought = t_match.group(1).strip()
        if a_match:
            # Take only the first line of the action field
            action = a_match.group(1).strip().split("\n")[0].strip()
            # Strip wrapping quotes if the model added them
            action = action.strip('"\'')

        # Normalise common case mistakes: "Buy Now" → "buy now"
        action_lower = action.lower()
        for keyword in ("buy now", "back to search", "next >", "< prev"):
            if action_lower == keyword:
                action = keyword
                break

        # Extract explicit [[obs:N]] references from thought
        ref_ids = [int(m.group(1)) for m in re.finditer(r'\[\[obs:(\d+)\]\]', thought)]

        return thought, action, ref_ids

    def _llm_with_retry(self, history: list) -> Tuple[str, str, List[int]]:
        """
        Call LLM and parse. If no action is found, retry up to max_format_retries
        times with a corrective prompt before giving up.
        """
        for attempt in range(self.max_format_retries + 1):
            try:
                raw = self.llm.chat(history)
            except Exception as e:
                return f"[LLM error: {e}]", "", []

            thought, action, ref_ids = self.parse_response(raw)

            if action:
                return thought, action, ref_ids

            # No action parsed — insert retry message
            if attempt < self.max_format_retries:
                history = history + [
                    {"role": "assistant", "content": raw},
                    {"role": "user",      "content": FORMAT_RETRY_PROMPT},
                ]

        # Exhausted retries
        return thought, "", ref_ids

    # ── Episode runner ────────────────────────────────────────────────────────

    def run_episode(
        self,
        env,
        task_id: str,
        instruction: str,
        initial_obs: str,
        max_steps: int = 15,
    ) -> Dict[str, Any]:
        """
        Run one full episode.

        Args:
            env:         environment object with .step(action) → (obs, reward, done, info)
            task_id:     string identifier for logging
            instruction: natural-language shopping instruction
            initial_obs: the first observation text from env.reset()
            max_steps:   hard cap on number of agent steps

        Returns:
            dict with keys: is_success, reward, steps, termination_reason, episode_id
        """
        episode_id = str(uuid.uuid4())
        seed = getattr(env, 'seed', 0)

        self.logger.start_episode(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            policy_version=self.policy_version,
            toolset_version="webshop_text_v1",
        )

        # Log initial observation
        self.logger.log_node(
            node_type="OBS_TOOL",
            payload={"raw_text": initial_obs, "tool_name": "env_reset"},
            call_id="step_init",
        )

        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Instruction: {instruction}\n\nObservation:\n{initial_obs}"},
        ]

        final_reward = 0.0
        termination  = "max_steps"
        steps_taken  = 0

        for step in range(max_steps):
            steps_taken = step + 1

            # ── LLM call (with format-retry) ──────────────────────────────
            thought, action, ref_ids = self._llm_with_retry(history)

            action_call_id = str(uuid.uuid4())

            self.logger.log_node(
                node_type="THOUGHT",
                payload={"raw_text": thought},
                ref_ids=ref_ids,
            )
            self.logger.log_node(
                node_type="ACT_TOOL",
                payload={"action_str": action},
                call_id=action_call_id,
            )

            if not action:
                termination = "format_failure"
                break

            # ── Environment step ──────────────────────────────────────────
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                obs, reward, done = f"[Env error: {e}]", 0.0, True
                termination = "env_error"

            final_reward = reward
            tool_name = action.split("[")[0] if "[" in action else action

            self.logger.log_node(
                node_type="OBS_TOOL",
                payload={"raw_text": obs, "tool_name": tool_name},
                call_id=action_call_id,
            )

            history.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {action}"})
            history.append({"role": "user",      "content": f"Observation:\n{obs}"})

            if done:
                termination = "done"
                break

        is_success = final_reward >= self.success_reward_threshold

        self.logger.end_episode(
            is_success=is_success,
            total_cost=0.0,
            termination_reason=termination,
        )

        return {
            "episode_id":        episode_id,
            "task_id":           task_id,
            "is_success":        is_success,
            "reward":            final_reward,
            "steps":             steps_taken,
            "termination_reason": termination,
        }
