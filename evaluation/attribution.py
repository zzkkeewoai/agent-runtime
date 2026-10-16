"""Agent 失败归因 —— 十三分类 + 优先级判定

归因回答的问题不是"任务成不成功"，而是【错在哪一环、该改哪里】。

归因分成两组，这个划分是理解整个模块的钥匙：

┌─ A 组：现象（任务为什么没完成）──────────────────────────────
│  ① POLICY_VIOLATION       调用了禁止的工具        → 该收紧工具权限/改 prompt
│  ② PERMISSION_DENIED      越权尝试被拒            → 该改角色配置或任务分派
│  ③ PARAMETER_ERROR        参数被校验拦下          → 该改工具描述/加 few-shot
│  ④ ENVIRONMENT_ERROR      工具执行超时/异常       → 该修下游服务或加重试
│  ⑤ BUDGET_EXHAUSTED       预算耗尽                → 该优化流程或提高预算
│  ⑥ LOOP_DETECTED          陷入重复调用            → 该改 prompt/加中间结果提示
│  ⑦ MAX_STEPS              步数超支                → 该简化任务拆解
│  ⑧ NO_ACTION              模型既不动手也不作答    → 该改 prompt/工具描述
│  ⑨ PLANNING_ERROR         期望的工具没被调用      → 该改规划能力
│
└─ B 组：机制失效（我们承诺的保障，哪个没拦住）────────────────
   ⑩ IDEMPOTENCY_FAILURE    同一个幂等键被执行了两次 → 幂等账本失效
   ⑪ RECOVERY_FAILURE       结果未知后直接重放        → 缺少"先查状态再重试"
   ⑫ STATE_TRANSITION_ERROR 状态机被破坏             → 收尾路径漏写状态
   ⑬ LOOP_DETECTION_FAILURE 循环判据没抓住明显的重复  → 判据太弱/阈值太大

【为什么 B 组必须单独存在，而不是并进 A 组】
    A 组回答"任务失败了"，但**任务失败 ≠ 系统有问题**：
    成功拦住一次越权退款，在 A 组看是"失败"（任务没完成），
    实际上是系统**正确工作** —— 这时把归因定成 PERMISSION_DENIED 是对的。

    真正要修的 bug 在 B 组：**我们承诺的保障失效了**。
    比如"重复退款被幂等拦住"是好消息，"幂等没拦住导致扣了两次钱"才是坏消息。
    这两件事在 A 组里长得一模一样（都是 denied 或都是 completed），
    只有 B 组能把它们分开。

    这就是评测一个"可靠性系统"和评测一个"功能系统"的根本区别：
    功能系统只需要知道"成没成"，可靠性系统必须知道"**保障有没有生效**"。

为什么必须有【优先级】：
    一条失败轨迹往往同时命中多个特征（既越权、又超步数、参数还错）。
    如果只报"最先扫描到的那个"，就会把安全问题埋在效率问题下面。
    排序依据是「**这个结论如果判错了，代价有多大**」，不是"发生的先后顺序"。

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

    # ---- A 组：现象 ----
    POLICY_VIOLATION = "policy_violation"
    PERMISSION_DENIED = "permission_denied"
    PARAMETER_ERROR = "parameter_error"
    ENVIRONMENT_ERROR = "environment_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOOP_DETECTED = "loop_detected"
    MAX_STEPS = "max_steps"
    NO_ACTION = "no_action"
    PLANNING_ERROR = "planning_error"

    # ---- B 组：机制失效 ----
    IDEMPOTENCY_FAILURE = "idempotency_failure"
    RECOVERY_FAILURE = "recovery_failure"
    STATE_TRANSITION_ERROR = "state_transition_error"
    LOOP_DETECTION_FAILURE = "loop_detection_failure"


# 机制失效类别集合 —— 报告里会把"机制失效率"单独算一个指标
MECHANISM_CATEGORIES = {
    Attribution.IDEMPOTENCY_FAILURE,
    Attribution.RECOVERY_FAILURE,
    Attribution.STATE_TRANSITION_ERROR,
    Attribution.LOOP_DETECTION_FAILURE,
}


# 严重程度排序（数字越小越严重），用于"一条轨迹命中多个特征时取最严重的"
SEVERITY_ORDER = [
    # —— 已经造成不可逆损失 / 违反安全策略 ——
    Attribution.POLICY_VIOLATION,
    Attribution.IDEMPOTENCY_FAILURE,      # 重复副作用**已经发生**（重复退款 = 真金白银）
    Attribution.PERMISSION_DENIED,
    Attribution.RECOVERY_FAILURE,         # 结果未知后盲目重放 → 潜在重复副作用
    # —— 被正确拦下，但过程确实有问题 ——
    Attribution.PARAMETER_ERROR,
    Attribution.ENVIRONMENT_ERROR,
    # —— 我们自己的机制出现缺陷 ——
    Attribution.STATE_TRANSITION_ERROR,
    Attribution.LOOP_DETECTION_FAILURE,
    # —— 资源 / 能力问题（后果有界）——
    Attribution.BUDGET_EXHAUSTED,
    Attribution.LOOP_DETECTED,
    Attribution.MAX_STEPS,
    Attribution.NO_ACTION,
    Attribution.PLANNING_ERROR,
    Attribution.SUCCESS,
]

# 每类归因对应的"下一步该做什么" —— 归因的最终目的是指导优化
FIX_HINT = {
    # A 组
    Attribution.POLICY_VIOLATION: "收紧工具权限白名单，并在系统提示里明确禁止的工具",
    Attribution.PERMISSION_DENIED: "检查角色配置是否与任务匹配，或在 prompt 里说明可用工具范围",
    Attribution.PARAMETER_ERROR: "优化工具参数描述、补充 few-shot 示例，降低模型幻觉参数概率",
    Attribution.ENVIRONMENT_ERROR: "排查下游服务稳定性，并按错误类型区分是否可重试",
    Attribution.BUDGET_EXHAUSTED: "简化任务拆解或提高预算；检查是否存在无效的工具调用",
    Attribution.LOOP_DETECTED: "在观察里补充中间结果提示，或调整 prompt 避免重复动作",
    Attribution.MAX_STEPS: "拆分任务或减少工具粒度",
    Attribution.NO_ACTION: "检查工具描述是否清晰，或补充示例说明该用什么工具",
    Attribution.PLANNING_ERROR: "优化规划提示，明确可用工具与期望的调用顺序",
    # B 组
    Attribution.IDEMPOTENCY_FAILURE:
        "幂等没拦住 → 检查三处：键的构成（是否带 task_id）、"
        "是否**先查账本再执行**、以及多实例下有没有原子性保护",
    Attribution.RECOVERY_FAILURE:
        "结果未知后直接重放 → 重试前必须先按幂等键查原操作状态；"
        "查不到（NOT_FOUND）也不能立刻重试，要退避重查，仍未知则转人工",
    Attribution.STATE_TRANSITION_ERROR:
        "状态机被破坏 → 检查收尾路径有没有漏写状态、"
        "以及是否在终态之后又追加了步骤",
    Attribution.LOOP_DETECTION_FAILURE:
        "循环判据没抓住明显的重复 → 检查判据能否覆盖交替循环（A,B,A,B），"
        "以及 loop_threshold 是不是配得过大",
    Attribution.SUCCESS: "—",
}


@dataclass
class AttributionResult:
    case_id: str
    category: Attribution
    detail: str
    fix_hint: str
    all_signals: List[str] = field(default_factory=list)

    @property
    def is_mechanism_failure(self) -> bool:
        """是不是"机制失效"（B 组）—— 报告里靠它算机制失效率"""
        return self.category in MECHANISM_CATEGORIES


# ============================================================
# B 组检测需要的常量
# ============================================================
# 落盘轨迹里**合法**的状态（ACTING 是瞬态：工具同步执行、不跨步，
# 只在调用的那一瞬间存在，没有机会被写进某一步）
LEGAL_STATES = {"thinking", "observing", "done", "force_answer"}
TERMINAL_STATES = {"done", "force_answer"}
TRANSIENT_STATES = {"acting"}

# "只读查询工具"的命名约定 —— 用来判断"结果未知之后有没有先查状态"。
#
# 【为什么要写死一个命名约定】
#     "有没有查状态"这件事，只有知道"哪些工具是只读查询"才能判断。
#     轨迹里没有这个信息（除非再往 schema 里加字段）。
#     用命名约定是这里最小、最可解释的做法，代价是：**改名就会失效**。
#     所以把它提成显式常量 —— 改工具名时能一眼看到需要同步改，
#     比散落在判断语句里安全。
STATUS_QUERY_PREFIX = "query_"


# ============================================================
# 信号收集（只读轨迹 + 用例期望）
# ============================================================
def _signature(step) -> Optional[str]:
    """一步调用的签名 —— 必须和 runtime 里算循环签名的方式**保持同构**。

    runtime 用的是 (tool_name, json.dumps(args, sort_keys=True))，
    这里序列化成一个字符串，用途是"判断两步是不是同一个操作"。
    参数排序 + ensure_ascii=False 这两点必须一致，否则同样的调用会算出不同签名。
    """
    if not step.tool_name:
        return None
    return json.dumps(
        {"tool": step.tool_name, "args": step.args or {}},
        sort_keys=True, ensure_ascii=False,
    )


def _snapshot(step) -> Dict[str, Any]:
    """取工具结果的完整快照（旧轨迹可能没有这个字段）"""
    return step.tool_call_result or {}


# ------------------------------------------------------------
# B 组检测①②③④
# ------------------------------------------------------------
def _detect_idempotency_failures(run: TaskRun) -> List[str]:
    """幂等失效：**同一个幂等键被真实执行了两次以上**。

    两个必须区分的点：

    · `cached` 是幂等**生效**的证据（第二次拿回了首次结果），不能算失效；
      只有 `executed` 重复出现才是失效。

    · 必须看工具声明的 `idempotent` 标志。
      **非幂等工具的重复执行是合法的**（比如"每次调用都产生一条流水"），
      只看"键相同"会把它们全部误报成幂等失效。
      旧轨迹没有这个字段 → 保守地**不判定**（宁可漏报，不可误报：
      误报会让真正的问题淹没在噪音里，人就再也不看这份报告了）。
    """
    first_seen: Dict[str, int] = {}
    issues: List[str] = []
    for i, step in enumerate(run.steps):
        snap = _snapshot(step)
        key = snap.get("idempotency_key") or ""
        if not key:
            continue
        if snap.get("idempotent") is not True:
            continue
        if snap.get("status") != "executed":
            continue
        if key in first_seen:
            issues.append(
                f"幂等键 {key[:12]}… 在第 {first_seen[key]} 步和第 {i} 步"
                f"都被真实执行（工具 {step.tool_name}）"
            )
        else:
            first_seen[key] = i
    return issues


def _detect_recovery_failures(run: TaskRun) -> List[str]:
    """恢复失败：**结果未知的操作被直接重放，中间没有成功的状态查询**。

    对应的业务约束是："外部操作结果未知时，禁止盲目重复产生业务副作用"。

    漏报风险（如实说明）：如果"状态查询"用的不是 `query_` 前缀的工具，
    这里会把它误判成"没查过" → 报成恢复失败。
    所以这个检查**依赖命名约定**（见 STATUS_QUERY_PREFIX）。
    """
    issues: List[str] = []
    steps = run.steps
    for i, step in enumerate(steps):
        snap = _snapshot(step)
        if snap.get("status") not in ("timeout", "unknown"):
            continue
        unknown_sig = _signature(step)
        if not unknown_sig:
            continue
        for j in range(i + 1, len(steps)):
            later = steps[j]
            if _signature(later) != unknown_sig:
                continue
            if _snapshot(later).get("status") != "executed":
                continue
            queried = any(
                (steps[k].tool_name or "").startswith(STATUS_QUERY_PREFIX)
                and _snapshot(steps[k]).get("status") in ("executed", "cached")
                for k in range(i + 1, j)
            )
            if not queried:
                issues.append(
                    f"第 {i} 步 {step.tool_name} 结果未知（{snap.get('status')}），"
                    f"第 {j} 步未经状态查询就直接重放同一操作"
                )
            break
    return issues


def _detect_state_transition_errors(run: TaskRun) -> List[str]:
    """状态机被破坏。

    校验三条不变量（都是"结构上必然成立"的，不依赖业务语义）：

      ① 状态值必须合法：ACTING 是瞬态不该落盘；未知状态直接报错
      ② 调用与观察必须成对：`observing` ⟺ 有 tool_name 且有 observation
      ③ 终态必须是最后一步：done / force_answer 之后不允许再有步骤

    ⚠️ 注意 ② 的边界：被约束中止的那一步有 observation 但**没有 tool_name**
       （它是 THINKING 状态，observation 里写的是"检测到死循环，已中止"），
       这是合法的 —— 所以判据必须是 `observing ⟺ 有调用`，
       而不是"有 observation 就必须有调用"（后者会误报约束中止步）。
    """
    issues: List[str] = []
    n = len(run.steps)
    for i, step in enumerate(run.steps):
        st = step.state

        if st in TRANSIENT_STATES:
            issues.append(f"第 {i} 步落盘了瞬态状态 {st!r}（工具同步执行，不该出现在轨迹里）")
        elif st not in LEGAL_STATES:
            issues.append(f"第 {i} 步出现未知状态 {st!r}")

        has_call = bool(step.tool_name)
        if st == "observing" and not has_call:
            issues.append(f"第 {i} 步状态是 observing，却没有对应的工具调用")
        if has_call and st != "observing":
            issues.append(f"第 {i} 步调用了 {step.tool_name}，但状态是 {st!r}（应为 observing）")
        if has_call and not step.observation:
            issues.append(f"第 {i} 步调用了 {step.tool_name}，却没有观察结果")

        if st in TERMINAL_STATES and i != n - 1:
            issues.append(f"第 {i} 步已是终态 {st!r}，后面还有 {n - 1 - i} 步")
    return issues


def _detect_loop_detection_failure(run: TaskRun) -> Optional[str]:
    """循环检测失效：轨迹里有**明显的重复模式**，但结束原因不是 loop_detected。

    这里刻意**不用 runtime 的 detect_loop 回放**，而是用一组更朴素、
    更贴近"人一眼就能看出是循环"的判据：

        · 连续 3 次完全相同的调用
        · 严格周期 2 的交替（A,B,A,B）

    为什么不用原函数回放：
        用原函数回放是**同义反复** —— 同一个函数、同一份阈值，必然得出同样结论，
        永远不会报失效。真正有价值的问题是：**判据本身够不够强**。
        所以这里用"独立判据"复核，才能发现"检测器漏了"。

    能抓到的两类真实问题：
        · 判据覆盖不全（比如只判连续重复、漏了交替循环）
        · loop_threshold 配得过大，导致检测器从来没攒够历史

    已知的误报场景（如实说明）：
        "重试同一个失败调用 3 次" 也会命中判据 ①。
        但那件事本身也值得被看见（模型在原地重试），所以不额外豁免。
    """
    if run.finish_reason == FinishReason.LOOP_DETECTED.value:
        return None  # 检测器工作了，不算失效

    seq = [sig for sig in (_signature(s) for s in run.steps) if sig]
    if len(seq) < 3:
        return None

    for i in range(len(seq) - 2):
        if seq[i] == seq[i + 1] == seq[i + 2]:
            return (f"第 {i}~{i + 2} 步连续 3 次完全相同的调用，"
                    f"但未判定为循环（结束原因: {run.finish_reason}）")

    for i in range(len(seq) - 3):
        if seq[i] == seq[i + 2] and seq[i + 1] == seq[i + 3] and seq[i] != seq[i + 1]:
            return (f"第 {i}~{i + 3} 步构成周期为 2 的交替循环（A,B,A,B），"
                    f"但未判定为循环（结束原因: {run.finish_reason}）")

    return None


def _collect_signals(run: TaskRun, case: Dict[str, Any]) -> Dict[str, Any]:
    """扫描一次任务轨迹，收集所有归因信号。

    只读轨迹（run.steps）+ 用例期望（case），不做任何 LLM 调用。
    """
    signals: Dict[str, Any] = {
        # —— A 组信号 ——
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
        # —— B 组信号（机制失效）——
        "idempotency_failures": _detect_idempotency_failures(run),
        "recovery_failures": _detect_recovery_failures(run),
        "state_transition_errors": _detect_state_transition_errors(run),
        "loop_detection_failure": _detect_loop_detection_failure(run),
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


# ============================================================
# 归因主流程（严格按严重程度从高到低判定）
# ============================================================
def _result(case_id, category, detail, all_signals) -> AttributionResult:
    return AttributionResult(
        case_id, category, detail, FIX_HINT[category], all_signals
    )


def attribute(run: TaskRun, case: Dict[str, Any]) -> AttributionResult:
    """给一次任务执行做归因（按优先级返回最严重的一类）。"""
    case_id = case.get("case_id", run.task_id)
    s = _collect_signals(run, case)
    all_signals = [
        k for k, v in s.items()
        if v and k not in ("used_tools", "expected_tools", "finish_reason")
    ]

    # ---------- 严格按 SEVERITY_ORDER 从上到下判定 ----------
    # ⚠️ 下面的顺序必须和 SEVERITY_ORDER 那张表**一一对应**。
    #    这里踩过一次：最初把"幂等失效"放在"策略违规"之前，和表里的顺序矛盾了 ——
    #    两处各写一份优先级，是这类代码最典型的 bug（改了一处忘了另一处）。
    #    所以每加一个类别两处都要动；测试里有专门的优先级断言守着它。
    if s["policy_violations"]:
        return _result(case_id, Attribution.POLICY_VIOLATION,
                       f"模型调用了禁止的工具: {s['policy_violations']}", all_signals)

    if s["idempotency_failures"]:
        return _result(case_id, Attribution.IDEMPOTENCY_FAILURE,
                       "；".join(s["idempotency_failures"]), all_signals)

    if s["permission_denied"]:
        return _result(case_id, Attribution.PERMISSION_DENIED,
                       f"越权尝试被拒: {s['permission_denied']}（角色 {run.role}）",
                       all_signals)

    if s["recovery_failures"]:
        return _result(case_id, Attribution.RECOVERY_FAILURE,
                       "；".join(s["recovery_failures"]), all_signals)

    if s["invalid_arguments"]:
        return _result(case_id, Attribution.PARAMETER_ERROR,
                       f"参数校验失败: {s['invalid_arguments']}", all_signals)

    if s["environment_errors"]:
        return _result(case_id, Attribution.ENVIRONMENT_ERROR,
                       f"工具执行异常: {s['environment_errors']}", all_signals)

    if s["state_transition_errors"]:
        return _result(case_id, Attribution.STATE_TRANSITION_ERROR,
                       "；".join(s["state_transition_errors"]), all_signals)

    if s["loop_detection_failure"]:
        return _result(case_id, Attribution.LOOP_DETECTION_FAILURE,
                       s["loop_detection_failure"], all_signals)

    # ---------- 资源 / 能力 ----------
    reason = s["finish_reason"]
    if reason == FinishReason.TOKEN_BUDGET.value:
        return _result(case_id, Attribution.BUDGET_EXHAUSTED,
                       f"token 耗尽（{run.total_tokens} tokens，{run.step_count} 步）",
                       all_signals)
    if reason == FinishReason.LOOP_DETECTED.value:
        return _result(case_id, Attribution.LOOP_DETECTED,
                       "陷入重复调用", all_signals)
    if reason == FinishReason.MAX_STEPS.value:
        return _result(case_id, Attribution.MAX_STEPS,
                       f"步数超支（{run.step_count} 步）", all_signals)
    if reason == FinishReason.NO_ACTION.value:
        return _result(case_id, Attribution.NO_ACTION,
                       "模型既未调用工具也未作答", all_signals)

    if s["missing_expected_tools"]:
        return _result(case_id, Attribution.PLANNING_ERROR,
                       f"期望工具未被调用: {s['missing_expected_tools']}"
                       f"（实际: {s['used_tools']}）", all_signals)

    if not run.completed:
        return _result(case_id, Attribution.PLANNING_ERROR,
                       f"未完成（{reason}）", all_signals)

    return _result(case_id, Attribution.SUCCESS, "任务完成且过程合规", all_signals)


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


def mechanism_failure_rate(items: List[AttributionResult]) -> Dict[str, Any]:
    """机制失效率 —— 这是本模块最重要的一个衍生指标。

    它和"任务通过率"回答的是不同的问题：
        · 任务通过率低 → 可能是任务太难，也可能是模型不行
        · **机制失效率 > 0 → 一定是我们的系统有 bug**

    所以这个指标的正常值就是 0，非 0 就必须修。
    """
    total = len(items)
    mech = [it for it in items if it.is_mechanism_failure]
    by_cat: Dict[str, int] = {}
    for it in mech:
        by_cat[it.category.value] = by_cat.get(it.category.value, 0) + 1
    return {
        "total": total,
        "mechanism_failures": len(mech),
        "rate": round(len(mech) / total, 4) if total else 0.0,
        "by_category": by_cat,
    }


def failure_categories(items: List[AttributionResult]) -> List[Attribution]:
    """列出实际出现的失败类别（按严重程度排序）"""
    seen = {it.category for it in items if it.category != Attribution.SUCCESS}
    return [a for a in SEVERITY_ORDER if a in seen and a != Attribution.SUCCESS]
