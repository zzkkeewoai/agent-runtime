"""Agent 失败归因 —— 九分类 + 优先级判定

归因回答的问题不是"任务成不成功"，而是【错在哪一环、该改哪里】：

    ① POLICY_VIOLATION   调用了禁止的工具          → 该收紧工具权限/改 prompt
    ② PERMISSION_DENIED  越权尝试被拒              → 该改角色配置或任务分派
    ③ PARAMETER_ERROR    参数被校验拦下            → 该改工具描述/加 few-shot
    ④ ENVIRONMENT_ERROR  工具执行超时/异常         → 该修下游服务或加重试
    ⑤ BUDGET_EXHAUSTED   预算耗尽                  → 该优化流程或提高预算
    ⑥ LOOP_DETECTED      陷入重复调用              → 该改 prompt/加中间结果提示
    ⑦ MAX_STEPS          步数超支                  → 该简化任务拆解
    ⑧ NO_ACTION          模型既不动手也不作答      → 该改 prompt/工具描述
    ⑨ PLANNING_ERROR     期望的工具没被调用        → 该改规划能力

为什么必须有【优先级】：
    一条失败轨迹往往同时命中多个特征（既越权、又超步数、参数还错）。
    如果只报"最先扫描到的那个"，就会把安全问题埋在效率问题下面。
    **安全问题的优先级永远最高** —— 这是企业级系统的基本要求。

判定原则：**确定性规则 + 只读轨迹**，不调用 LLM。
    因为归因本身必须可靠、可复现、零成本 —— LLM 归因适合做补充，不能做主力。
"""
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from runtime import FinishReason, TaskRun  # noqa: E402


class Attribution(str, Enum):
    """归因类别（含成功）"""

    SUCCESS = "success"
    POLICY_VIOLATION = "policy_violation"
    PERMISSION_DENIED = "permission_denied"
    PARAMETER_ERROR = "parameter_error"
    ENVIRONMENT_ERROR = "environment_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOOP_DETECTED = "loop_detected"
    MAX_STEPS = "max_steps"
    NO_ACTION = "no_action"
    PLANNING_ERROR = "planning_error"


# 严重程度排序（数字越小越严重），用于"一条轨迹命中多个特征时取最严重的"
SEVERITY_ORDER = [
    Attribution.POLICY_VIOLATION,
    Attribution.PERMISSION_DENIED,
    Attribution.PARAMETER_ERROR,
    Attribution.ENVIRONMENT_ERROR,
    Attribution.BUDGET_EXHAUSTED,
    Attribution.LOOP_DETECTED,
    Attribution.MAX_STEPS,
    Attribution.NO_ACTION,
    Attribution.PLANNING_ERROR,
    Attribution.SUCCESS,
]

# 每类归因对应的"下一步该做什么" —— 归因的最终目的是指导优化
FIX_HINT = {
    Attribution.POLICY_VIOLATION: "收紧工具权限白名单，并在系统提示里明确禁止的工具",
    Attribution.PERMISSION_DENIED: "检查角色配置是否与任务匹配，或在 prompt 里说明可用工具范围",
    Attribution.PARAMETER_ERROR: "优化工具参数描述、补充 few-shot 示例，降低模型幻觉参数概率",
    Attribution.ENVIRONMENT_ERROR: "排查下游服务稳定性，并按错误类型区分是否可重试",
    Attribution.BUDGET_EXHAUSTED: "简化任务拆解或提高预算；检查是否存在无效的工具调用",
    Attribution.LOOP_DETECTED: "在观察里补充中间结果提示，或调整 prompt 避免重复动作",
    Attribution.MAX_STEPS: "拆分任务或减少工具粒度",
    Attribution.NO_ACTION: "检查工具描述是否清晰，或补充示例说明该用什么工具",
    Attribution.PLANNING_ERROR: "优化规划提示，明确可用工具与期望的调用顺序",
    Attribution.SUCCESS: "—",
}


@dataclass
class AttributionResult:
    case_id: str
    category: Attribution
    detail: str
    fix_hint: str
    all_signals: List[str] = field(default_factory=list)


def _collect_signals(run: TaskRun, case: Dict[str, Any]) -> Dict[str, Any]:
    """扫描一次任务轨迹，收集所有归因信号。

    只读轨迹（run.steps）+ 用例期望（case），不做任何 LLM 调用。
    """
    signals: Dict[str, Any] = {
        "policy_violations": [],   # 调用了被禁止的工具
        "permission_denied": [],   # 被权限拒
        "invalid_arguments": [],   # 参数被拒
        "environment_errors": [],  # 超时/异常
        "used_tools": [],
        "expected_tools": case.get("expect", {}).get("tools_used", []) or [],
        # 注意：forbidden_tools（系统级安全策略，归因用）与
        # expect.must_not_use（任务级业务规则，判定用）是两个不同的概念，不能混用。
        "forbidden_tools": case.get("forbidden_tools", []) or [],
        "finish_reason": run.finish_reason,
    }

    for step in run.steps:
        if step.tool_name:
            signals["used_tools"].append(step.tool_name)
        status = step.tool_status
        obs = step.observation or ""

        if status == "denied":
            if "[permission_denied]" in obs:
                signals["permission_denied"].append(step.tool_name)
            elif "[invalid_arguments]" in obs:
                signals["invalid_arguments"].append(step.tool_name)
        if status in ("timeout", "error"):
            signals["environment_errors"].append(step.tool_name)

    # 策略违规：模型尝试调用了明确禁止的工具（即使被拦，也是严重信号）
    for forbidden in signals["forbidden_tools"]:
        if forbidden in signals["used_tools"]:
            signals["policy_violations"].append(forbidden)

    # 期望的工具没被调用
    missing = [t for t in signals["expected_tools"] if t not in signals["used_tools"]]
    signals["missing_expected_tools"] = missing
    return signals


def attribute(run: TaskRun, case: Dict[str, Any]) -> AttributionResult:
    """给一次任务执行做归因（按优先级返回最严重的一类）。"""
    case_id = case.get("case_id", run.task_id)
    s = _collect_signals(run, case)
    all_signals = [k for k, v in s.items() if v and k not in ("used_tools", "expected_tools", "finish_reason")]

    # 逐级判定（严重 → 轻微）
    if s["policy_violations"]:
        return AttributionResult(
            case_id,
            Attribution.POLICY_VIOLATION,
            f"模型调用了禁止的工具: {s['policy_violations']}",
            FIX_HINT[Attribution.POLICY_VIOLATION],
            all_signals,
        )

    if s["permission_denied"]:
        return AttributionResult(
            case_id,
            Attribution.PERMISSION_DENIED,
            f"越权尝试被拒: {s['permission_denied']}（角色 {run.role}）",
            FIX_HINT[Attribution.PERMISSION_DENIED],
            all_signals,
        )

    if s["invalid_arguments"]:
        return AttributionResult(
            case_id,
            Attribution.PARAMETER_ERROR,
            f"参数校验失败: {s['invalid_arguments']}",
            FIX_HINT[Attribution.PARAMETER_ERROR],
            all_signals,
        )

    if s["environment_errors"]:
        return AttributionResult(
            case_id,
            Attribution.ENVIRONMENT_ERROR,
            f"工具执行异常: {s['environment_errors']}",
            FIX_HINT[Attribution.ENVIRONMENT_ERROR],
            all_signals,
        )

    reason = s["finish_reason"]
    if reason == FinishReason.TOKEN_BUDGET.value:
        return AttributionResult(
            case_id, Attribution.BUDGET_EXHAUSTED,
            f"token 耗尽（{run.total_tokens} tokens，{run.step_count} 步）",
            FIX_HINT[Attribution.BUDGET_EXHAUSTED], all_signals,
        )
    if reason == FinishReason.LOOP_DETECTED.value:
        return AttributionResult(
            case_id, Attribution.LOOP_DETECTED,
            "陷入重复调用", FIX_HINT[Attribution.LOOP_DETECTED], all_signals,
        )
    if reason == FinishReason.MAX_STEPS.value:
        return AttributionResult(
            case_id, Attribution.MAX_STEPS,
            f"步数超支（{run.step_count} 步）", FIX_HINT[Attribution.MAX_STEPS], all_signals,
        )
    if reason == FinishReason.NO_ACTION.value:
        return AttributionResult(
            case_id, Attribution.NO_ACTION,
            "模型既未调用工具也未作答", FIX_HINT[Attribution.NO_ACTION], all_signals,
        )

    if s["missing_expected_tools"]:
        return AttributionResult(
            case_id, Attribution.PLANNING_ERROR,
            f"期望工具未被调用: {s['missing_expected_tools']}（实际: {s['used_tools']}）",
            FIX_HINT[Attribution.PLANNING_ERROR], all_signals,
        )

    if not run.completed:
        return AttributionResult(
            case_id, Attribution.PLANNING_ERROR,
            f"未完成（{reason}）", FIX_HINT[Attribution.PLANNING_ERROR], all_signals,
        )

    return AttributionResult(case_id, Attribution.SUCCESS, "任务完成且过程合规", "—", all_signals)


# ============================================================
# 汇总
# ============================================================
def summarize_attributions(items: List[AttributionResult]) -> Dict[str, int]:
    """归因分布 —— 这张表就是"该往哪优化"的结论。"""
    dist: Dict[str, int] = {}
    for it in items:
        key = it.category.value
        dist[key] = dist.get(key, 0) + 1
    # 按严重程度排序输出
    order = {a.value: i for i, a in enumerate(SEVERITY_ORDER)}
    return dict(sorted(dist.items(), key=lambda kv: order.get(kv[0], 99)))


def failure_categories(items: List[AttributionResult]) -> List[Attribution]:
    """列出实际出现的失败类别（按严重程度排序）"""
    seen = {it.category for it in items if it.category != Attribution.SUCCESS}
    return [a for a in SEVERITY_ORDER if a in seen and a != Attribution.SUCCESS]
