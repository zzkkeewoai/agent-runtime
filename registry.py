"""工具注册中心 —— 工具的"身份证"

每个工具必须声明三件企业级最关心的事：
  1. **参数 schema**（Pydantic）→ 挡住模型幻觉出的参数
  2. **权限白名单**（allowed_roles）→ 谁能调
  3. **高危规则**（requires_approval / approval_rule）→ 什么情况需要人工审批

另外声明两个工程属性：
  · timeout_seconds  超时（防某个工具挂死拖垮服务）
  · idempotent       是否可安全重放（决定重试策略是否安全）

为什么要有"注册中心"而不是直接用一个函数字典：
    注册中心是**策略的单一事实来源** —— 权限、高危、超时、参数 schema 都在这里声明，
    网关只负责按声明执行。新增工具时**自动继承全部安全策略**，不会漏。
    如果策略散落在各个工具内部，迟早会出现"某个工具忘了加权限检查"的漏洞。
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Type

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """一个工具的完整声明"""

    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[[BaseModel], Any]
    allowed_roles: Set[str]

    # ---- 高危控制 ----
    requires_approval: bool = False
    # 动态审批规则：按参数决定是否高危（如"退款金额 > 500 才要审批"）
    approval_rule: Optional[Callable[[BaseModel], bool]] = None

    # ---- 工程属性 ----
    timeout_seconds: float = 5.0
    idempotent: bool = True
    side_effect: bool = False  # 是否产生副作用（写库 / 对外发送）

    def needs_approval(self, args: BaseModel) -> bool:
        """判断本次调用是否需要人工审批。

        静态规则（requires_approval）优先；否则看动态规则。
        为什么要支持动态：只做静态的话，要么全部审批（人烦了就无脑点通过，
        等于没审），要么全不审批（风险失控）。
        """
        if self.requires_approval:
            return True
        if self.approval_rule is not None:
            return bool(self.approval_rule(args))
        return False


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._tools[spec.name] = spec

    def register_all(self, specs: List[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return sorted(self._tools)

    def to_function_schemas(self) -> List[Dict[str, Any]]:
        """转成 Function Calling 格式的 schema（这是给模型看的部分）。

        注意：description 的质量直接影响模型"选对工具"的概率——
        工具描述写不清，模型就会选错工具，这是 Agent 失败的高频原因之一。
        """
        out: List[Dict[str, Any]] = []
        for spec in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.args_model.model_json_schema(),
                    },
                }
            )
        return out
