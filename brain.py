"""模型接口（Brain）+ 可控的 Mock 实现

为什么要把模型抽象成接口：
    Agent 运行时不应该依赖具体模型实现 —— 这样：
      · 测试时注入可控的 Mock（零 API key、可复现、可故意制造异常）
      · 生产时换成真实 LLM，**运行时逻辑一行都不用改**
    这也是为什么这套测试能在没有 API key 的情况下跑出完整结果。

为什么 Mock 要"可控"而不是"随机"：
    要测试运行时的约束（死循环检测、预算中断、优雅收尾），
    必须能【故意制造】这些异常场景。随机 Mock 做得到"跑"，但做不到"稳定复现异常"。
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Decision:
    """模型的一次决策：要么调工具，要么直接作答"""

    thought: str
    action: Optional[Dict[str, Any]] = None  # {"tool": "...", "args": {...}}
    answer: Optional[str] = None
    tokens: int = 0  # 本次决策消耗的 token（真实场景由模型返回）


@dataclass
class Observation:
    """一次工具执行后的观察结果"""

    step: int
    tool_name: str
    args: Dict[str, Any]
    status: str
    result: Any = None
    reason: str = ""


class Brain:
    """模型接口（生产环境接真实 LLM）"""

    def decide(self, goal: str, observations: List[Observation], step_index: int) -> Decision:
        raise NotImplementedError


class ScriptedBrain(Brain):
    """按脚本执行的 Mock 大脑 —— 用于稳定复现各种场景。

    用法：
        brain = ScriptedBrain([
            {"thought": "先查订单", "action": {"tool": "query_order", "args": {...}}},
            {"thought": "有结果了", "answer": "订单状态是已支付"},
        ])

    当脚本用完后：
      · 默认重复最后一个决策（用来测"死循环"）
      · repeat_last=False 时返回兜底答案（用来测正常结束）
    """

    def __init__(
        self,
        script: List[Dict[str, Any]],
        repeat_last: bool = True,
        tokens_per_step: int = 120,
        answer_when_exhausted: str = "已根据现有信息给出结论。",
    ):
        self.script = script
        self.repeat_last = repeat_last
        self.tokens_per_step = tokens_per_step
        self.answer_when_exhausted = answer_when_exhausted

    def decide(self, goal: str, observations: List[Observation], step_index: int) -> Decision:
        if step_index < len(self.script):
            item = self.script[step_index]
        elif self.repeat_last and self.script:
            item = self.script[-1]
        else:
            return Decision(
                thought="脚本已结束，基于现有信息作答",
                answer=self.answer_when_exhausted,
                tokens=self.tokens_per_step,
            )

        return Decision(
            thought=item.get("thought", ""),
            action=item.get("action"),
            answer=item.get("answer"),
            tokens=self.tokens_per_step,
        )


class ForceAnswerBrain(Brain):
    """受限场景专用：无论问什么，只给答案不调工具。

    运行时在"优雅收尾"阶段会把 brain 换成它（或调用 force_answer 方法），
    保证收尾阶段一定产出一个答案而不是继续调工具。
    """

    def __init__(self, answer: str = "任务已达执行上限，以下是基于现有信息的结论。"):
        self.answer = answer

    def decide(self, goal: str, observations: List[Observation], step_index: int) -> Decision:
        summary = "；".join(f"{o.tool_name}={o.result}" for o in observations[-3:]) or "无工具结果"
        return Decision(
            thought="达到约束上限，基于现有观察作答",
            answer=f"{self.answer}（依据：{summary}）",
            tokens=30,
        )
