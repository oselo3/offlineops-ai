"""
core/agents/engine.py

A minimal, dependency-free tool-calling agent loop.

The agent receives a user query + retrieved context, decides whether to
call a tool, executes it, and feeds the result back into the LLM until
it produces a final answer or hits max_iterations.

Design principle: no LangChain, no LlamaIndex — just a clean loop that
is easy to read, audit, and extend.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Any

from core.inference.ollama_client import OllamaClient, Message


# ------------------------------------------------------------------ #
# Tool registry                                                        #
# ------------------------------------------------------------------ #

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema of expected params
    fn: Callable[..., str]    # must return a string result


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schema(self) -> str:
        """Return a formatted string describing all available tools."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  Parameters: {json.dumps(t.parameters)}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Agent output                                                         #
# ------------------------------------------------------------------ #

@dataclass
class AgentStep:
    type: str         # "thought" | "tool_call" | "tool_result" | "answer"
    content: str
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    model: str
    total_latency_ms: float


# ------------------------------------------------------------------ #
# System prompt                                                        #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are OfflineOps AI — an infrastructure operations assistant.
You help engineers diagnose and resolve infrastructure issues using provided documentation context and available tools.

You operate in a loop:
1. Think about the problem (start with "Thought:")
2. If you need more information from the system, call a tool using this exact JSON format on its own line:
   TOOL_CALL: {{"tool": "tool_name", "args": {{"param": "value"}}}}
3. You will receive the tool result, then continue thinking.
4. When you have enough information, provide your final answer starting with "Answer:"

Available tools:
{tool_schema}

Rules:
- Be concise and precise.
- Only call tools when necessary — prefer the documentation context when sufficient.
- Never guess system state — if you need it, call a tool.
- If a tool call fails, explain why and suggest manual steps.
- Always cite which documentation source supports your answer when applicable.
"""


# ------------------------------------------------------------------ #
# Agent engine                                                         #
# ------------------------------------------------------------------ #

class AgentEngine:
    def __init__(
        self,
        ollama: OllamaClient,
        registry: ToolRegistry,
        model: str = "llama3.1:8b",
        max_iterations: int = 5,
    ):
        self.ollama = ollama
        self.registry = registry
        self.model = model
        self.max_iterations = max_iterations

    def run(self, query: str, context: str = "") -> AgentResult:
        """
        Run the agent loop for a given query.

        Args:
            query: The user's natural language question.
            context: Retrieved RAG context (pre-formatted).

        Returns:
            AgentResult with the final answer and full reasoning trace.
        """
        import time
        t0 = time.perf_counter()

        system = SYSTEM_PROMPT.format(tool_schema=self.registry.schema())

        user_message = f"""Documentation context:
{context if context else "(no context retrieved)"}

Question: {query}"""

        messages = [Message(role="user", content=user_message)]
        steps: list[AgentStep] = []
        total_latency = 0.0

        for iteration in range(self.max_iterations):
            response = self.ollama.chat(
                messages=messages,
                model=self.model,
                system=system,
                temperature=0.0,
            )
            total_latency += response.latency_ms
            reply = response.content.strip()

            # Append assistant message to history
            messages.append(Message(role="assistant", content=reply))

            # Check for final answer
            if "Answer:" in reply:
                answer = reply.split("Answer:", 1)[1].strip()
                steps.append(AgentStep(type="answer", content=answer))
                break

            # Check for tool call
            tool_call_match = re.search(
                r"TOOL_CALL:\s*(\{.*?\})", reply, re.DOTALL
            )
            if tool_call_match:
                try:
                    call = json.loads(tool_call_match.group(1))
                    tool_name = call["tool"]
                    tool_args = call.get("args", {})

                    steps.append(AgentStep(
                        type="tool_call",
                        content=reply,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    ))

                    tool = self.registry.get(tool_name)
                    if tool:
                        result = tool.fn(**tool_args)
                    else:
                        result = f"ERROR: Tool '{tool_name}' not found."

                    steps.append(AgentStep(
                        type="tool_result",
                        content=result,
                        tool_name=tool_name,
                    ))

                    # Feed result back into the conversation
                    messages.append(Message(
                        role="user",
                        content=f"Tool result for {tool_name}:\n{result}"
                    ))

                except (json.JSONDecodeError, KeyError) as e:
                    steps.append(AgentStep(
                        type="tool_result",
                        content=f"ERROR parsing tool call: {e}",
                    ))
            else:
                # Model responded without a tool call or answer — treat as thought
                steps.append(AgentStep(type="thought", content=reply))

        else:
            # Hit max iterations without a clean answer
            steps.append(AgentStep(
                type="answer",
                content="I reached my reasoning limit. Please review the steps above and try rephrasing your question.",
            ))

        answer = next(
            (s.content for s in reversed(steps) if s.type == "answer"),
            "No answer produced."
        )

        return AgentResult(
            answer=answer,
            steps=steps,
            model=self.model,
            total_latency_ms=round(time.perf_counter() - t0, 2) * 1000,
        )
