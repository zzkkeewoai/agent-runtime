"""Agent 运行时 —— 显式状态机 + 三个硬约束 + 全步留痕

设计目标（对应裸 ReAct 的五个问题）：
    ① 路由漂移   → 流程固化成状态，模型只能在节点内决策
    ② 死循环     → 重复调用检测（连续 N 次相同调用 → 判定卡死）
    ③ 成本失控   → token 预算 + 步数上限
    ④ 挂了重来   → TaskRun 全量落盘（轨迹可回放）
    ⑤ 不可观测   → 每一步都记录（思考/动作/观察/token/耗时）

关键设计：**触发约束不直接失败，而是"优雅收尾"**
    跑了一半直接报错，用户什么都拿不到；而"基于已有信息给出部分结论"
    对用户是有价值的。所以达到上限时进入 FORCE_ANSWER 状态收尾。
"""
import json
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain, ForceAnswerBrain, Observation  # noqa: E402
from gateway import ToolCallRequest, ToolGateway  # noqa: E402


class AgentState(str, Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FORCE_ANSWER = "force_answer"
    DONE = "done"


class FinishReason(str, Enum):
    ANSWERED = "answered"            # 正常给出答案
    NO_ACTION = "no_action"          # 模型既不调工具也不作答（异常）
    MAX_STEPS = "max_steps"          # 步数超限 → 已优雅收尾
    TOKEN_BUDGET = "token_budget"    # 预算超限 → 已优雅收尾
    LOOP_DETECTED = "loop_detected"  # 重复调用卡死 → 已优雅收尾
    USAGE_ERROR = "usage_error"      # 模型给出了非法动作


@dataclass
class Step:
    """一步轨迹（可回放的最小单位）"""

    index: int
    state: str
    thought: str = ""
    tool_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    tool_status: Optional[str] = None
    observation: Optional[str] = None
    tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class TaskRun:
    """一次任务的完整轨迹"""

    task_id: str
    goal: str
    role: str
    status: str                      # completed / aborted
    finish_reason: str
    final_answer: Optional[str]
    steps: List[Step] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    tool_calls: int = 0
    denied_calls: int = 0
    pending_approvals: int = 0
    created_at: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["step_count"] = self.step_count
        d["completed"] = self.completed
        return d


# ============================================================
# 轨迹存储
# ============================================================
class TraceStore:
    """轨迹落盘（JSONL；真实场景落 Postgres，并按 task_id 建索引）。

    轨迹是 Agent 可观测性的最小要求：
      · 出问题能回放（哪一步开始偏的）
      · 能做评测（同一任务不同版本对比）
      · 能算指标（步数、工具调用次数、拒绝次数、token 消耗）
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def save(self, run: TaskRun) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def load(self, task_id: str) -> Optional[Dict[str, Any]]:
        for row in self.load_all():
            if row.get("task_id") == task_id:
                return row
        return None

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    def stats(self) -> Dict[str, Any]:
        rows = self.load_all()
        if not rows:
            return {"tasks": 0}
        by_reason: Dict[str, int] = {}
        for r in rows:
            by_reason[r["finish_reason"]] = by_reason.get(r["finish_reason"], 0) + 1
        return {
            "tasks": len(rows),
            "completed": sum(1 for r in rows if r.get("completed")),
            "success_rate": round(sum(1 for r in rows if r.get("completed")) / len(rows), 3),
            "by_finish_reason": by_reason,
            "avg_steps": round(sum(r["step_count"] for r in rows) / len(rows), 2),
            "avg_tokens": round(sum(r["total_tokens"] for r in rows) / len(rows), 1),
            "avg_latency_ms": round(sum(r["total_latency_ms"] for r in rows) / len(rows), 2),
        }


# ============================================================
# 运行时
# ============================================================
class AgentRuntime:
    """Agent 执行循环（状态机）"""

    def __init__(
        self,
        brain: Brain,
        gateway: ToolGateway,
        max_steps: int = 8,
        token_budget: int = 5000,
        loop_threshold: int = 3,
        trace_store: Optional[TraceStore] = None,
    ):
        self.brain = brain
        self.gateway = gateway
        self.max_steps = max_steps
        self.token_budget = token_budget
        # 连续多少次相同调用算"卡死"
        self.loop_threshold = loop_threshold
        self.trace_store = trace_store

    # ----------------------------------------------------------
    def run(self, goal: str, role: str, task_id: Optional[str] = None) -> TaskRun:
        task_id = task_id or f"T-{uuid.uuid4().hex[:8]}"
        started = time.time()

        steps: List[Step] = []
        observations: List[Observation] = []
        total_tokens = 0
        reason = FinishReason.ANSWERED
        answer: Optional[str] = None
        recent_calls: Deque[Tuple[str, str]] = deque(maxlen=self.loop_threshold)

        for step_index in range(self.max_steps):
            t_step = time.time()

            # ---------- THINKING：模型决策 ----------
            decision = self.brain.decide(goal, observations, step_index)
            total_tokens += decision.tokens
            step = Step(
                index=step_index,
                state=AgentState.THINKING.value,
                thought=decision.thought,
                tokens=decision.tokens,
            )

            # 预算检查（在动作之前判断，避免多花一次工具调用的成本）
            if total_tokens > self.token_budget:
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.TOKEN_BUDGET
                break

            # 模型直接作答 → 正常结束
            if decision.answer:
                answer = decision.answer
                step.observation = "(模型直接作答)"
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.ANSWERED
                break

            # 既没动作也没答案 → 认为模型输出非法
            if not decision.action:
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.NO_ACTION
                break

            tool_name = decision.action.get("tool", "")
            args = decision.action.get("args", {}) or {}

            # ---------- 死循环检测 ----------
            signature = (tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            recent_calls.append(signature)
            if len(recent_calls) == self.loop_threshold and len(set(recent_calls)) == 1:
                step.observation = f"(检测到连续 {self.loop_threshold} 次相同调用: {tool_name})"
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.LOOP_DETECTED
                break

            # ---------- ACTING：所有动作都过工具网关 ----------
            result = self.gateway.call(
                ToolCallRequest(
                    task_id=task_id,
                    role=role,
                    tool_name=tool_name,
                    args=args,
                    step=step_index,
                )
            )
            step.tool_name = tool_name
            step.args = args
            step.tool_status = result.status

            # ---------- OBSERVING：把结果转成观察 ----------
            observation = Observation(
                step=step_index,
                tool_name=tool_name,
                args=args,
                status=result.status,
                result=result.result,
                reason=result.reason,
            )
            observations.append(observation)
            step.observation = self._render_observation(result)
            step.latency_ms = round((time.time() - t_step) * 1000, 2)
            steps.append(step)
        else:
            # for 正常走完（没有 break）→ 步数用尽
            reason = FinishReason.MAX_STEPS

        # ---------- 优雅收尾：触发约束时不直接失败，而是产出一个结论 ----------
        if reason != FinishReason.ANSWERED:
            answer, tail_step = self._graceful_finish(goal, observations, reason, len(steps))
            steps.append(tail_step)
            total_tokens += tail_step.tokens
            status = "aborted"
        else:
            status = "completed"

        run = TaskRun(
            task_id=task_id,
            goal=goal,
            role=role,
            status=status,
            finish_reason=reason.value,
            final_answer=answer,
            steps=steps,
            total_tokens=total_tokens,
            total_latency_ms=round((time.time() - started) * 1000, 2),
            tool_calls=sum(1 for s in steps if s.tool_name),
            denied_calls=sum(1 for s in steps if s.tool_status == "denied"),
            pending_approvals=sum(1 for s in steps if s.tool_status == "pending_approval"),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # ---------- 落盘（可回放） ----------
        if self.trace_store:
            self.trace_store.save(run)
        return run

    # ----------------------------------------------------------
    @staticmethod
    def _render_observation(result) -> str:
        """把工具结果渲染成模型能读的观察文本。

        注意：**被拒绝的调用也要作为观察反馈给模型**——
        否则模型不知道"这个工具不能调"，会一直重试同一个被拒的调用
        （这是 Agent 卡死的常见原因之一）。
        """
        if result.status == "executed":
            return f"工具 {result.tool_name} 执行成功，结果：{result.result}"
        if result.status == "cached":
            return f"工具 {result.tool_name} 命中幂等缓存，结果：{result.result}"
        if result.status == "pending_approval":
            return (
                f"工具 {result.tool_name} 属于高危操作，已提交人工审批（approval_id={result.approval_id}），"
                f"请勿重复提交，可先用其他只读工具继续推进或直接给出阶段性结论。"
            )
        return f"工具 {result.tool_name} 未执行（{result.status}）：{result.reason}"

    def _graceful_finish(
        self, goal: str, observations: List[Observation], reason: FinishReason, next_index: int
    ) -> Tuple[str, Step]:
        """优雅收尾：用专用 brain 产出一个"基于现有信息"的结论。

        为什么不在收尾阶段继续用原 brain：
            原 brain 可能继续要求调工具（它不知道已经没预算了），
            会导致收尾阶段又触发约束。收尾必须用一个"只输出答案"的实现。
        """
        t0 = time.time()
        finisher = ForceAnswerBrain(
            answer=f"任务因 {reason.value} 提前终止"
        )
        decision = finisher.decide(goal, observations, next_index)
        step = Step(
            index=next_index,
            state=AgentState.FORCE_ANSWER.value,
            thought=decision.thought,
            observation="(约束触发，进入优雅收尾)",
            tokens=decision.tokens,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )
        return decision.answer or "无法给出结论。", step
