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
    """Agent 状态。

    【ACTING 是瞬态，不会出现在落盘轨迹里】——这点必须说清楚，
    否则会让人以为状态机漏实现了：
        工具是同步执行的，一次调用不跨越步边界，所以"正在执行"这个状态
        只存在于调用的那一瞬间，没有机会被写进某一步的 state 字段。
        落盘的 state 记录的是**每一步结束时**的状态：
            THINKING      这一步没走到动作就结束了（被预算 / 循环约束中止）
            OBSERVING     调完工具、拿到观察，循环继续
            DONE          模型直接给出答案，任务正常结束
            FORCE_ANSWER  触发约束 → 进入优雅收尾
    """

    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FORCE_ANSWER = "force_answer"
    DONE = "done"


def detect_loop(history: List[Tuple[str, str]], threshold: int) -> Tuple[bool, str]:
    """判断工具调用序列是否已经陷入死循环。

    覆盖两种模式 —— 原实现只覆盖了第一种：

      ① 原地打转：连续 threshold 次**完全相同**的调用
      ② 周期性交替：A,B,A,B... —— 窗口里签名种类不止一种，但序列严格周期重复

    【为什么第二种必须单独判】
        原实现的判定条件是 `len(set(recent_calls)) == 1`，
        也就是说：只要窗口里出现第二种签名，判定就永远不成立。
        于是一个在两个工具之间来回摇摆的 Agent（什么都没推进）
        会一路跑满 max_steps 才停下 —— 中间烧掉的每一次调用和 token 全是白费。
        这类循环在真实系统里很常见：模型"查一下 A，发现要看 B，查完 B 又觉得该看 A"。

    为什么要求**严格周期**，而不是"签名种类少就算"：
        正常的多步任务也会反复用同几个工具。
        只有序列严格按周期重复（window[i] == window[i-period] 对全部 i 成立），
        才说明它没有任何推进。判得太宽松会把正常任务误杀。

    签名里包含**完整参数**，所以"对不同实体做同样的操作"（args 不同）
    本来就是不同签名，天然不会被误判成循环。
    """
    items = list(history)
    if len(items) < threshold:
        return False, ""

    # ① 原地打转
    tail = items[-threshold:]
    if len(set(tail)) == 1:
        return True, f"连续 {threshold} 次完全相同的调用（{tail[0][0]}）"

    # ② 周期性交替（至少要凑够 2 个周期才够判）
    span = threshold * 2
    if len(items) >= span:
        window = items[-span:]
        for period in range(1, threshold + 1):
            if all(window[i] == window[i - period] for i in range(period, len(window))):
                names = "→".join(dict.fromkeys(w[0] for w in window[:period]))
                return (
                    True,
                    f"检测到周期为 {period} 的重复调用序列（{names}），该循环没有推进任何信息",
                )
    return False, ""


class FinishReason(str, Enum):
    ANSWERED = "answered"            # 正常给出答案
    NO_ACTION = "no_action"          # 模型既没调工具也没作答（什么都没给）
    MAX_STEPS = "max_steps"          # 步数超限 → 已优雅收尾
    TOKEN_BUDGET = "token_budget"    # 预算超限 → 已优雅收尾
    LOOP_DETECTED = "loop_detected"  # 重复调用卡死 → 已优雅收尾
    USAGE_ERROR = "usage_error"      # 给了动作但格式非法（非 dict / 缺 tool 字段）


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
        # ---------- 参数下界校验（fail-fast） ----------
        # 为什么必须在这里拦，而不是"信任调用方"：
        #   这些参数配错时**不会报错，只会静默地做错事**，而且错的方向很隐蔽：
        #     loop_threshold = 1 → 一次正常调用就被判成"连续 1 次相同" →
        #         **正常任务被误杀**，steps=2 且工具一次都没执行；
        #     loop_threshold = 0 → deque(maxlen=0) 窗口永远为空 →
        #         **循环检测静默失效**，无限循环要跑满 max_steps 才被兜住。
        #   两种都是实测出来的（不是推演）。合法下界是 2：
        #     模式①「连续相同」至少需要 2 个样本；
        #     模式②「严格周期」需要 2×threshold 个样本，threshold>=2 才有意义。
        if loop_threshold < 2:
            raise ValueError(
                f"loop_threshold 必须 >= 2（收到 {loop_threshold}）："
                f"配 1 会把一次正常调用误判成死循环，配 0 会让循环检测静默失效"
            )
        if max_steps < 2:
            raise ValueError(f"max_steps 必须 >= 2（收到 {max_steps}）")
        if token_budget <= 0:
            raise ValueError(f"token_budget 必须 > 0（收到 {token_budget}）")

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
        # 窗口开到 2 倍阈值：判定"周期为 2 的交替"至少需要 4 个样本，
        # 窗口只留 threshold 个会漏掉 A,B,A,B 这类循环（这正是原实现的缺口）
        recent_calls: Deque[Tuple[str, str]] = deque(maxlen=self.loop_threshold * 2)

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
                step.state = AgentState.DONE.value
                reason = FinishReason.ANSWERED
                break

            # 既没动作也没答案 → 模型什么都没给
            if not decision.action:
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.NO_ACTION
                break

            # 有动作但格式非法 → 必须先校验再取字段。
            # 直接写 `decision.action.get("tool", "")` 是有风险的：
            # 真实 LLM 完全可能吐出 action="query_order"（字符串）或 {"tool": null}，
            # 那时 .get() 会抛 AttributeError，把整个任务崩掉 ——
            # 而"模型输出格式不对"恰恰是 Agent 最常见的失败之一，必须优雅处理。
            action = decision.action
            if not isinstance(action, dict) or not action.get("tool"):
                step.observation = "(模型输出的动作格式非法：action 必须是含 tool 字段的对象)"
                step.latency_ms = round((time.time() - t_step) * 1000, 2)
                steps.append(step)
                reason = FinishReason.USAGE_ERROR
                break

            tool_name = str(action["tool"])
            args = action.get("args", {}) or {}

            # ---------- 死循环检测 ----------
            signature = (tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            recent_calls.append(signature)
            looped, why = detect_loop(list(recent_calls), self.loop_threshold)
            if looped:
                step.observation = f"(检测到死循环，已中止: {why})"
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
            # 这一步走完了「动作 + 观察」，落盘为 OBSERVING（循环继续）
            step.state = AgentState.OBSERVING.value
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
