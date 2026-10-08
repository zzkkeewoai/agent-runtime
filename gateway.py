"""工具网关 —— 所有工具调用的唯一入口

五道关，顺序不可变：

    ① 工具存在性    → 未注册的工具直接拒
    ② 权限校验      → 角色不在白名单直接拒
    ③ 参数校验      → Pydantic schema 挡住幻觉参数
    ④ 计算幂等键    → 供"幂等检查"与"审批去重"共同使用
    ⑤ 幂等检查      → 同一任务同一调用若已执行过，返回首次结果（不再要求审批）
    ⑥ 高危审批      → 未执行过的高危调用才挂起（HITL）；**按幂等键去重**
    ⑦ 沙箱执行      → 超时保护 + 异常隔离
    ⑧ 审计留痕      → 谁/何时/调了什么/结果/耗时

为什么顺序不能变（其中第 2 条是我写反过、被测试抓出来的）：
    · 权限必须在参数校验前：无权限的调用没必要浪费校验成本，也不该把
      "参数格式对不对"这种信息透给无权限的调用方；
    · **幂等检查必须在审批前**：幂等不是"能不能做"，而是"是否已经做过"。
      已经成功执行过的调用不该再要求审批（审批早就发生过了）。
      —— 第一版我把审批放在幂等之前，理由是"先问能不能做，再问做没做过"，
      这个推理是错的，测试立刻抓到了（审批后重试又走了审批分支）；
    · 幂等键必须在审批前算好：审批单本身也要幂等，否则重试会创建一堆
      一模一样的待办，审批人烦了就会无脑点通过，审核形同虚设；
    · 幂等检查必须在执行前：否则就已经重复执行了。
"""
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from registry import ToolRegistry, ToolSpec

# ============================================================
# 请求 / 结果
# ============================================================
@dataclass
class ToolCallRequest:
    """一次工具调用请求（由 Agent 生成）"""

    task_id: str
    role: str
    tool_name: str
    args: Dict[str, Any]
    step: int = 0


@dataclass
class ToolCallResult:
    status: str  # executed / cached / denied / pending_approval / timeout / error
    tool_name: str
    result: Any = None
    reason: str = ""
    audit_id: str = ""
    idempotency_key: str = ""
    approval_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("executed", "cached")


# ============================================================
# 审计存储
# ============================================================
class AuditStore:
    """审计留痕（JSONL，真实场景落 Postgres）。

    审计不是"顺便记个日志"，它是合规要求：
    任何对生产数据的写操作都必须能回答"谁在什么时候通过什么方式改了什么"。
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: Dict[str, Any]) -> str:
        audit_id = f"AUD-{uuid.uuid4().hex[:10]}"
        entry = dict(entry)
        entry["audit_id"] = audit_id
        entry.setdefault("timestamp", datetime.now().isoformat())
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return audit_id

    def load_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    def stats(self) -> Dict[str, Any]:
        """审计统计 —— 这就是"越权拦截率"等指标的来源。"""
        rows = self.load_all()
        by_status: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for r in rows:
            by_status[r.get("decision", "?")] = by_status.get(r.get("decision", "?"), 0) + 1
            if r.get("reason"):
                by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        total = len(rows)
        denied = by_status.get("denied", 0)
        return {
            "total": total,
            "by_status": by_status,
            "by_reason": by_reason,
            "denied_count": denied,
            "allowed_count": by_status.get("executed", 0) + by_status.get("cached", 0),
            "pending_approval": by_status.get("pending_approval", 0),
        }


# ============================================================
# 幂等账本
# ============================================================
class IdempotencyLedger:
    """记录"已执行过的调用"，防止重试导致重复副作用。

    幂等键 = hash(task_id + tool_name + 规范化参数)
      · 带 task_id：区分"两个不同任务的相同操作"（要执行两次）
                    和"同一任务重试"（只能执行一次）
      · 参数规范化（sort_keys）：保证 {"a":1,"b":2} 与 {"b":2,"a":1} 得到同一个键

    真实场景应该落 Redis/DB（多实例共享），这里用内存 + 可选落盘。
    """

    def __init__(self, path: Optional[str] = None):
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.path = path

    @staticmethod
    def build_key(task_id: str, tool_name: str, args: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"task_id": task_id, "tool": tool_name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def put(self, key: str, result: Any) -> None:
        with self._lock:
            self._store[key] = result
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")


# ============================================================
# 审批队列（HITL）
# ============================================================
class ApprovalStore:
    """待审批队列。

    ⚠️ 真实场景必须持久化（Postgres）——因为审批可能等几分钟到几小时，
    进程重启后"正在等审批"的任务不能丢。这里用内存 + JSONL 演示接口设计。
    """

    def __init__(self, path: Optional[str] = None):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self.path = path

    def create(
        self, req: ToolCallRequest, spec: ToolSpec, args: Dict[str, Any], idempotency_key: str = ""
    ) -> Tuple[str, bool]:
        """创建审批单（**按幂等键去重**）。

        返回 (approval_id, reused)：
          reused=True 表示复用了已存在的待审批单（没有新建）。

        为什么必须去重（这是修掉的一个真实缺陷）：
            Agent 重试非常常见（网络抖动、超时重发、上层重跑）。
            如果每次重试都新建审批单，会出现：
              · 审批人看到一堆一模一样的待办 → 烦 → 无脑点通过 → 审核失效
              · 同一次操作被批多次 → 可能重复执行
            所以审批单本身也要幂等：同一幂等键只保留一个待审批单。
        """
        if idempotency_key:
            existing = self.find_pending_by_key(idempotency_key)
            if existing:
                return existing, True

        approval_id = f"APV-{uuid.uuid4().hex[:8]}"
        self._pending[approval_id] = {
            "approval_id": approval_id,
            "task_id": req.task_id,
            "role": req.role,
            "tool_name": spec.name,
            "args": args,
            "idempotency_key": idempotency_key,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "decided_at": None,
            "approver": None,
            "note": "",
        }
        return approval_id, False

    def find_pending_by_key(self, idempotency_key: str) -> Optional[str]:
        """按幂等键查找已存在的待审批单"""
        for aid, item in self._pending.items():
            if item["status"] == "pending" and item.get("idempotency_key") == idempotency_key:
                return aid
        return None

    def get(self, approval_id: str) -> Optional[Dict[str, Any]]:
        return self._pending.get(approval_id)

    def decide(self, approval_id: str, approved: bool, approver: str, note: str = "") -> Dict[str, Any]:
        item = self._pending.get(approval_id)
        if not item:
            raise KeyError(f"审批单不存在: {approval_id}")
        item["status"] = "approved" if approved else "rejected"
        item["decided_at"] = datetime.now().isoformat()
        item["approver"] = approver
        item["note"] = note
        return item

    def list_pending(self) -> List[Dict[str, Any]]:
        return [v for v in self._pending.values() if v["status"] == "pending"]


# ============================================================
# 工具网关
# ============================================================
class ToolGateway:
    """五道关 + 执行 + 审计"""

    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditStore,
        approvals: Optional[ApprovalStore] = None,
        ledger: Optional[IdempotencyLedger] = None,
    ):
        self.registry = registry
        self.audit = audit
        self.approvals = approvals or ApprovalStore()
        self.ledger = ledger or IdempotencyLedger()

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def call(self, req: ToolCallRequest) -> ToolCallResult:
        started = time.time()

        # ---------- ① 工具存在性 ----------
        spec = self.registry.get(req.tool_name)
        if spec is None:
            return self._deny(req, "tool_not_found", f"工具未注册: {req.tool_name}", started)

        # ---------- ② 权限校验 ----------
        if req.role not in spec.allowed_roles:
            return self._deny(
                req,
                "permission_denied",
                f"角色 {req.role} 无权调用 {spec.name}（允许: {sorted(spec.allowed_roles)}）",
                started,
            )

        # ---------- ③ 参数校验 ----------
        try:
            args = spec.args_model(**req.args)
        except ValidationError as e:
            detail = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
            return self._deny(req, "invalid_arguments", f"参数校验失败 → {detail}", started)

        args_dict = args.model_dump()

        # ---------- ④ 计算幂等键（审批去重与执行判断都要用） ----------
        idem_key = IdempotencyLedger.build_key(req.task_id, spec.name, args_dict)

        # ---------- ⑤ 幂等检查（必须在审批之前！）----------
        # 【顺序设计修正】我第一版把"审批"放在"幂等"之前，理由是"先问能不能做，
        # 再问做没做过" —— 这个推理是错的。测试立刻抓到了：
        #   高危调用 → 审批通过 → 执行成功 → 再重试同一调用
        #   → 期望返回缓存，实际又走了审批分支（因为它先判断 needs_approval）
        #
        # 正确逻辑：**幂等不是"能不能做"，而是"是否已经做过"**。
        # 已经成功执行过的调用，不该再要求审批（审批早就发生过了）。
        # 所以顺序必须是：幂等检查 → 高危审批 → 执行。
        if spec.idempotent and self.ledger.has(idem_key):
            cached = self.ledger.get(idem_key)
            audit_id = self.audit.record(
                {
                    "task_id": req.task_id,
                    "role": req.role,
                    "tool_name": spec.name,
                    "args": args_dict,
                    "decision": "cached",
                    "reason": "duplicate_call_within_same_task",
                    "idempotency_key": idem_key,
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            return ToolCallResult(
                status="cached",
                tool_name=spec.name,
                result=cached,
                reason="同一任务内重复调用，返回首次执行结果（幂等）",
                audit_id=audit_id,
                idempotency_key=idem_key,
            )

        # ---------- ⑥ 高危审批（按幂等键去重） ----------
        if spec.needs_approval(args):
            approval_id, reused = self.approvals.create(req, spec, args_dict, idem_key)
            audit_id = self.audit.record(
                {
                    "task_id": req.task_id,
                    "role": req.role,
                    "tool_name": spec.name,
                    "args": args_dict,
                    "decision": "pending_approval",
                    "reason": "approval_reused" if reused else "high_risk_requires_approval",
                    "approval_id": approval_id,
                    "idempotency_key": idem_key,
                    "side_effect": spec.side_effect,
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            return ToolCallResult(
                status="pending_approval",
                tool_name=spec.name,
                reason=(
                    "高危操作，复用已存在的待审批单（幂等，避免重复审批）"
                    if reused
                    else "高危操作，已提交人工审批"
                ),
                audit_id=audit_id,
                idempotency_key=idem_key,
                approval_id=approval_id,
            )

        # ---------- ⑥ 执行 ----------
        return self._execute_and_audit(req, spec, args, args_dict, idem_key, started)

    # ----------------------------------------------------------
    # 审批通过与拒绝
    # ----------------------------------------------------------
    def approve_and_run(self, approval_id: str, approver: str, run: bool = True) -> ToolCallResult:
        """人工审批后执行（或拒绝）。

        真实场景里审批可能发生在几分钟后、由另一个进程处理——
        所以这里通过 approval_id 从存储里取出原始请求再执行。
        """
        item = self.approvals.get(approval_id)
        if item is None:
            raise KeyError(f"审批单不存在: {approval_id}")

        started = time.time()
        self.approvals.decide(approval_id, approved=run, approver=approver)

        if not run:
            audit_id = self.audit.record(
                {
                    "task_id": item["task_id"],
                    "role": item["role"],
                    "tool_name": item["tool_name"],
                    "args": item["args"],
                    "decision": "denied",
                    "reason": "rejected_by_human",
                    "approver": approver,
                    "approval_id": approval_id,
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            return ToolCallResult(
                status="denied",
                tool_name=item["tool_name"],
                reason=f"人工拒绝（审批人: {approver}）",
                audit_id=audit_id,
                approval_id=approval_id,
            )

        spec = self.registry.get(item["tool_name"])
        if spec is None:
            raise KeyError(f"工具已不存在: {item['tool_name']}")
        args = spec.args_model(**item["args"])
        args_dict = args.model_dump()
        idem_key = IdempotencyLedger.build_key(item["task_id"], spec.name, args_dict)
        req = ToolCallRequest(
            task_id=item["task_id"], role=item["role"], tool_name=spec.name, args=item["args"]
        )
        result = self._execute_and_audit(
            req, spec, args, args_dict, idem_key, started, extra={"approver": approver, "approval_id": approval_id}
        )
        return result

    # ----------------------------------------------------------
    # 内部：执行 + 审计
    # ----------------------------------------------------------
    def _execute_and_audit(
        self,
        req: ToolCallRequest,
        spec: ToolSpec,
        args,
        args_dict: Dict[str, Any],
        idem_key: str,
        started: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ToolCallResult:
        status, result, reason = "executed", None, ""
        try:
            result = self._run_with_timeout(spec, args)
            if spec.idempotent:
                self.ledger.put(idem_key, result)
        except FuturesTimeoutError:
            status, reason = "timeout", f"执行超时（>{spec.timeout_seconds}s）"
        except Exception as e:  # noqa: BLE001
            status, reason = "error", f"{type(e).__name__}: {e}"

        entry: Dict[str, Any] = {
            "task_id": req.task_id,
            "role": req.role,
            "tool_name": spec.name,
            "args": args_dict,
            "decision": status,
            "reason": reason,
            "idempotency_key": idem_key,
            "side_effect": spec.side_effect,
            "duration_ms": round((time.time() - started) * 1000, 2),
        }
        if extra:
            entry.update(extra)
        audit_id = self.audit.record(entry)

        return ToolCallResult(
            status=status,
            tool_name=spec.name,
            result=result,
            reason=reason,
            audit_id=audit_id,
            idempotency_key=idem_key,
        )

    @staticmethod
    def _run_with_timeout(spec: ToolSpec, args):
        """带超时执行工具。

        ⚠️ 这里有个真实的坑：如果写成
              with ThreadPoolExecutor(max_workers=1) as ex:
                  return ex.submit(spec.handler, args).result(timeout=...)
          超时后【不会真的返回】——因为 ThreadPoolExecutor 的上下文管理器退出时
          会调用 shutdown(wait=True)，**阻塞等待任务跑完**，超时保护形同虚设。

        正确做法：不用 with，超时后 shutdown(wait=False) 立即放弃等待。
        """
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(spec.handler, args)
        try:
            return future.result(timeout=spec.timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------
    # 内部：拒绝并审计
    # ----------------------------------------------------------
    def _deny(self, req: ToolCallRequest, reason_code: str, message: str, started: float) -> ToolCallResult:
        audit_id = self.audit.record(
            {
                "task_id": req.task_id,
                "role": req.role,
                "tool_name": req.tool_name,
                "args": req.args,
                "decision": "denied",
                "reason": reason_code,
                "detail": message,
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
        )
        return ToolCallResult(
            status="denied",
            tool_name=req.tool_name,
            reason=f"[{reason_code}] {message}",
            audit_id=audit_id,
        )
