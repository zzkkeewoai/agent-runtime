"""确定性回放 —— 把一条历史轨迹原样搬回本地重跑

【为什么需要它】
    Agent 是出了名的"难复现"：模型每次输出可能不同、工具每次返回也可能不同。
    线上出问题时如果只能"重新跑一遍"，你根本分不清是——
      代码改了？模型抽风了？还是数据变了？
    有了轨迹回放，就能把当时那次执行**原样搬回本地**，
    不依赖真实数据库、邮箱、支付系统，反复调试同一个失败现场。

【它回放的是什么（这点必须说清）】
    回放的**不是模型**，而是**工具执行层** —— `ReplayGateway` 顶替真实网关，
    按轨迹顺序返回当时记录的工具结果，**一个都不真实执行**。

    这么切分是有意的：**把确定的部分和不确定的部分分开**
      · 系统链路（状态机 / 三道约束 / 网关顺序 / 观察渲染）是确定的 → 回放必须逐字一致
      · 模型是不确定的 → 将来接真实 LLM 时，不一致就只可能来自模型

    所以回放器在这里的作用是：**先证明系统链路是确定的**。
    这样以后出现差异时，第一个该怀疑的对象才轮到模型 ——
    而不是像现在这样"每次都不一样，不知道从哪查起"。

【它发现不了什么（诚实说）】
    回放假设"环境没变"。如果工具本身行为变了（接口返回格式改了、业务规则改了），
    回放**不会报错** —— 因为它读的是记录，不是真实返回。
    所以它是"复现当时现场"的工具，**不是回归测试工具**。
"""
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain, ScriptedBrain  # noqa: E402
from gateway import ToolCallRequest, ToolCallResult  # noqa: E402
from runtime import AgentRuntime, Step  # noqa: E402


class ReplayDivergence(Exception):
    """回放过程中发现不可继续的偏差（比如重放比原轨迹多调了一次工具）。"""


# ============================================================
# 回放网关
# ============================================================
class ReplayGateway:
    """回放专用网关：不执行任何工具，按顺序返回轨迹里记录的原始结果。

    它和真实 `ToolGateway` 的接口一致（都提供 `call(req) -> ToolCallResult`），
    所以 `AgentRuntime` 根本不知道自己在被回放 ——
    **这是"回放不侵入业务代码"的前提**：没有在 runtime 里写一行 `if replay:`。
    """

    def __init__(self, steps: List[Step]):
        # 只取真正发起了工具调用的步
        self._planned: List[Step] = [s for s in steps if s.tool_name]
        self._cursor = 0

    @property
    def consumed(self) -> int:
        return self._cursor

    @property
    def total(self) -> int:
        return len(self._planned)

    def call(self, req: ToolCallRequest) -> ToolCallResult:
        if self._cursor >= len(self._planned):
            raise ReplayDivergence(
                f"回放越界：轨迹里只记录了 {len(self._planned)} 次工具调用，"
                f"但重放时又发起了第 {self._cursor + 1} 次（工具 {req.tool_name}）。"
                f"这说明重放时的决策路径和原轨迹不同 —— 正是回放要发现的东西。"
            )
        planned = self._planned[self._cursor]
        if planned.tool_name != req.tool_name:
            raise ReplayDivergence(
                f"第 {self._cursor + 1} 次工具调用对不上："
                f"轨迹记录的是 {planned.tool_name}，重放时调的是 {req.tool_name}"
            )
        if planned.args != req.args:
            raise ReplayDivergence(
                f"第 {self._cursor + 1} 次工具调用的参数对不上："
                f"轨迹 {planned.args} vs 重放 {req.args}"
            )
        self._cursor += 1
        # 用**完整快照**重建这次调用结果 —— reason 和 approval_id 都会出现在
        # 渲染后的观察文本里，缺任何一个，回放出的轨迹就和原来不一样
        snap = planned.tool_call_result or {}
        return ToolCallResult(
            status=snap.get("status") or planned.tool_status or "error",
            tool_name=snap.get("tool_name") or planned.tool_name or "",
            result=snap.get("result"),
            reason=snap.get("reason", ""),
            approval_id=snap.get("approval_id"),
        )


# ============================================================
# 从轨迹反推 brain
# ============================================================
def brain_from_trace(record: Dict[str, Any]) -> Brain:
    """从一条轨迹里反推出可重放的 brain 脚本。

    【为什么需要它】这样回放**只需要一条轨迹**，不需要当初那份脚本代码。
    否则"回放"就要依赖"你手上还有当时那份 brain 实现"，实用性大打折扣。

    【必须还原的两个细节 —— 第一版我在这里踩了坑，8 条轨迹重放失败】
      1. **用 `raw_action` 而不是"tool_name + args"反拼**。
         非法 action（字符串、tool 为 null）反拼不出来 —— 而那正是
         `usage_error` 这类失败要回放的东西。反拼的结果是把它们统统变成
         "模型直接作答"，结束原因也随之从 usage_error 变成 answered。
      2. **判断原脚本是不是"无限重复"型**。
         如果整条轨迹里**没有任何一步是模型主动作答**（没有 done 步），
         说明当初那个脚本是 `repeat_last=True`（被反复执行到触发约束为止）。
         我第一版写死了 `repeat_last=False`，于是重放到第二次就"脚本用完 → 兜底作答"，
         把 `loop_detected` 全部重放成了 `answered`。
    """
    steps = record.get("steps", [])
    script: List[Dict[str, Any]] = []

    for s in steps:
        thought = s.get("thought", "")
        raw = s.get("raw_action")
        if raw is not None:
            script.append({"thought": thought, "action": raw})
        elif s.get("state") == "done":
            script.append({"thought": thought, "answer": record.get("final_answer")})
        else:
            # 既没有 action、也不是作答步 —— 对应"模型什么都没做"那种情况。
            # 脚本里必须原样占一个位置，否则后续步骤整体错位。
            script.append({"thought": thought})

    has_done = any(s.get("state") == "done" for s in steps)

    # 3. **每步 token 数也要还原**。
    #    它决定"预算在第几步耗尽"，进而决定结束原因 ——
    #    拿默认的 120 去重放一个 tokens_per_step=200 的任务，预算就不会按时触发，
    #    原本的 token_budget 会被重放成 loop_detected（这是实测踩到的第 4 个坑）。
    #    轨迹里每步都记了 tokens，取第一个"真正由模型产生"的步即可
    #    （收尾步是 ForceAnswerBrain 产生的，token 数不是模型配置值，要排除）。
    scripted_tokens = [
        s["tokens"]
        for s in steps
        if (s.get("raw_action") is not None or s.get("state") == "done") and s.get("tokens")
    ]
    tokens_per_step = scripted_tokens[0] if scripted_tokens else 120

    return ScriptedBrain(script, repeat_last=not has_done, tokens_per_step=tokens_per_step)


# ============================================================
# 比对
# ============================================================
@dataclass
class Divergence:
    """一处不一致"""

    step: int
    field: str
    expected: Any
    actual: Any

    def render(self) -> str:
        pos = "整体" if self.step < 0 else f"第 {self.step} 步"
        return (
            f"{pos} · {self.field}\n"
            f"      轨迹：{self.expected!r}\n"
            f"      重放：{self.actual!r}"
        )


@dataclass
class ReplayReport:
    """一次回放的比对结果"""

    task_id: str
    ok: bool = True
    steps_original: int = 0
    steps_replayed: int = 0
    tool_calls_original: int = 0
    tool_calls_replayed: int = 0
    finish_reason_original: str = ""
    finish_reason_replayed: str = ""
    tokens_original: int = 0
    tokens_replayed: int = 0
    divergences: List[Divergence] = field(default_factory=list)
    error: Optional[str] = None

    def render(self) -> str:
        head = f"{'✓' if self.ok else '✗'} 回放 {self.task_id}"
        lines = [head]
        if self.error:
            lines.append(f"    中断：{self.error}")
            return "\n".join(lines)
        lines.append(
            f"    步数        {self.steps_original} → {self.steps_replayed}"
            + ("  一致" if self.steps_original == self.steps_replayed else "  **不一致**")
        )
        lines.append(
            f"    工具调用    {self.tool_calls_original} → {self.tool_calls_replayed}"
            + ("  一致" if self.tool_calls_original == self.tool_calls_replayed else "  **不一致**")
        )
        lines.append(
            f"    结束原因    {self.finish_reason_original} → {self.finish_reason_replayed}"
            + ("  一致" if self.finish_reason_original == self.finish_reason_replayed else "  **不一致**")
        )
        lines.append(
            f"    token       {self.tokens_original} → {self.tokens_replayed}"
            + ("  一致" if self.tokens_original == self.tokens_replayed else "  **不一致**")
        )
        if self.divergences:
            lines.append(f"    发现 {len(self.divergences)} 处差异：")
            for d in self.divergences:
                lines.append("      " + d.render())
        return "\n".join(lines)


_STEP_FIELDS = set(Step.__dataclass_fields__.keys())


def _compare_steps(original: List[Dict[str, Any]], replayed: List[Dict[str, Any]]) -> List[Divergence]:
    """逐步比对两边的轨迹。

    比对的字段是**行为字段**（思考、动作、参数、状态、观察），
    不比耗时（latency 天然有浮动，比它没意义，只会制造假告警）。
    """
    out: List[Divergence] = []
    for i, (a, b) in enumerate(zip(original, replayed)):
        for f in ("state", "thought", "tool_name", "args", "tool_status", "observation"):
            va, vb = a.get(f), b.get(f)
            if va != vb:
                out.append(Divergence(step=i, field=f, expected=va, actual=vb))
    # 长度不一致本身就是差异
    if len(original) != len(replayed):
        out.append(
            Divergence(
                step=min(len(original), len(replayed)),
                field="step_count",
                expected=len(original),
                actual=len(replayed),
            )
        )
    return out


# ============================================================
# 对外入口
# ============================================================
def replay(record: Dict[str, Any], trace_store=None) -> ReplayReport:
    """回放一条轨迹并逐字段比对。

    参数 `record` 就是 `TraceStore.load(task_id)` 返回的那条 dict
    （也就是 traces.jsonl 里的一行）。
    """
    task_id = record.get("task_id", "?")
    original_steps = record.get("steps", [])

    missing = [
        s["index"] for s in original_steps if s.get("tool_name") and not s.get("tool_call_result")
    ]
    if missing:
        return ReplayReport(
            task_id=task_id,
            ok=False,
            error=(
                f"轨迹缺少 tool_call_result（第 {missing} 步）——这是早期版本落的轨迹，"
                f"当时没有记录工具结果的完整快照，无法完整回放。重新跑一次即可生成可回放的轨迹。"
            ),
        )

    gateway = ReplayGateway([Step(**{k: v for k, v in s.items() if k in _STEP_FIELDS}) for s in original_steps])
    brain = brain_from_trace(record)

    runtime = AgentRuntime(
        brain=brain,
        gateway=gateway,  # type: ignore[arg-type]  # 回放网关与真网关接口一致
        max_steps=record.get("max_steps") or 8,
        token_budget=record.get("token_budget") or 5000,
        loop_threshold=record.get("loop_threshold") or 3,
        trace_store=trace_store,
    )

    try:
        new_run = runtime.run(record["goal"], role=record["role"], task_id=task_id)
    except ReplayDivergence as e:
        return ReplayReport(task_id=task_id, ok=False, error=str(e))

    new = new_run.to_dict()
    divergences = _compare_steps(original_steps, new.get("steps", []))

    # 【顶层指标也必须判】只比 steps 会漏掉一种很隐蔽的不一致：
    #   每一步看起来都对，但**结束原因变了 / 预算在第几步耗尽变了**。
    #   比如约束触发顺序一旦被改动，轨迹的步内容可能仍然一致，结束原因却不同 ——
    #   这在"改了约束逻辑"之后正是最该被抓出来的回归。
    # 不比对 latency：耗时天然有浮动，比它只会制造假告警。
    summary_fields = [
        ("finish_reason", record.get("finish_reason", ""), new.get("finish_reason", "")),
        ("step_count", len(original_steps), len(new.get("steps", []))),
        ("tool_calls", record.get("tool_calls", 0), new.get("tool_calls", 0)),
        ("total_tokens", record.get("total_tokens", 0), new.get("total_tokens", 0)),
    ]
    top_divergences = [
        Divergence(step=-1, field=f, expected=a, actual=b)
        for f, a, b in summary_fields
        if a != b
    ]
    all_divergences = divergences + top_divergences

    return ReplayReport(
        task_id=task_id,
        ok=not all_divergences,
        steps_original=len(original_steps),
        steps_replayed=len(new.get("steps", [])),
        tool_calls_original=record.get("tool_calls", 0),
        tool_calls_replayed=new.get("tool_calls", 0),
        finish_reason_original=record.get("finish_reason", ""),
        finish_reason_replayed=new.get("finish_reason", ""),
        tokens_original=record.get("total_tokens", 0),
        tokens_replayed=new.get("total_tokens", 0),
        divergences=all_divergences,
    )


def replay_all(trace_path: str) -> List[ReplayReport]:
    """回放一个轨迹文件里的全部任务 —— 这是它最有用的用法。

    **把历史轨迹当成一套回归测试**：改了代码之后全部回放一遍，
    任何一处行为变化都会被 `_compare_steps` 抓出来，精确到"第几步、哪个字段"。
    而且**不需要真实数据库、不需要 API Key、不需要模型** ——
    所以它是可以在 CI 里跑的。
    """
    if not os.path.exists(trace_path):
        return []
    reports: List[ReplayReport] = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reports.append(replay(json.loads(line)))
    return reports


def main() -> int:
    """命令行入口：python replay.py [traces.jsonl]"""
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "traces.jsonl"
    )
    reports = replay_all(path)
    if not reports:
        print(f"没有可回放的轨迹：{path}")
        return 1

    print("=" * 74)
    print(f"确定性回放 · {path}")
    print("=" * 74)
    for r in reports:
        print(r.render())
        print()

    ok = sum(1 for r in reports if r.ok)
    print("-" * 74)
    print(f"结果：{ok}/{len(reports)} 条轨迹可确定性重放")
    if ok != len(reports):
        print("（不一致的轨迹会精确指出是第几步、哪个字段对不上）")
    return 0 if ok == len(reports) else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
