# Agent Runtime 快速上手指南

给 vibe coding 同学的实用路线图 —— 不看长文档，直接上手玩。

---

## 🎯 第一步：跑起来（5 分钟）

**目标**：看到测试全部通过，建立"它能跑"的信心。

```bash
# 1. 安装依赖（只有一个）
pip install pydantic

# 2. 跑三个核心测试
python tests/test_gateway.py      # 工具网关（25 项断言）
python tests/test_runtime.py      # Agent 运行时（40 项断言）
python evaluation/run_eval.py     # 任务评测（10 条任务）
```

**看到什么**：
- `25/25 通过` / `40/40 通过` / `归因准确率 10/10 = 100%`
- 一堆 `✅` 和绿色的数字

**此时你知道了**：系统基本功能是 work 的。

---

## 🎮 第二步：玩一个最小 Demo（10 分钟）

**目标**：自己写一个任务，看 Agent 怎么跑的。

创建 `playground.py`：

```python
from brain import ScriptedBrain
from gateway import ToolGateway
from runtime import AgentRuntime
from builtin_tools import get_builtin_tools

# 1. 准备工具
registry, gateway = get_builtin_tools()

# 2. 写一个简单脚本：先查订单，再回答
brain = ScriptedBrain(
    script=[
        {
            "thought": "用户要查订单，我先调工具",
            "action": {"tool": "query_order", "args": {"order_id": "ORD-12345"}}
        },
        {
            "thought": "查到了，告诉用户",
            "answer": "您的订单状态是：已发货"
        }
    ],
    repeat_last=False  # 跑完脚本就结束
)

# 3. 跑起来
runtime = AgentRuntime(brain=brain, gateway=gateway)
result = runtime.run(
    task_id="DEMO-001",
    role="agent_staff",  # 客服专员角色
    goal="查询订单 ORD-12345 的状态"
)

# 4. 看结果
print(f"\n✅ 任务状态: {result.finish_reason}")
print(f"📝 最终答案: {result.final_answer}")
print(f"📊 用了 {len(result.steps)} 步")

# 5. 看详细过程
for i, step in enumerate(result.steps, 1):
    print(f"\n--- 第 {i} 步 ---")
    print(f"思考: {step.thought}")
    if step.action:
        print(f"动作: {step.action}")
    print(f"观察: {step.observation}")
```

**运行**：
```bash
python playground.py
```

**你会看到**：
```
✅ 任务状态: answered
📝 最终答案: 您的订单状态是：已发货
📊 用了 2 步

--- 第 1 步 ---
思考: 用户要查订单，我先调工具
动作: {'tool': 'query_order', 'args': {'order_id': 'ORD-12345'}}
观察: [executed] 订单 ORD-12345 状态: shipped (已发货)

--- 第 2 步 ---
思考: 查到了，告诉用户
动作: None
观察: 您的订单状态是：已发货
```

**此时你知道了**：
- Agent 怎么"思考 → 动作 → 观察"循环的
- 每一步都记录了什么

---

## 🔥 第三步：玩坏它（15 分钟）

**目标**：故意制造各种异常场景，看系统怎么拦的。

### 实验 1：越权操作

```python
# 把角色改成 agent_reader（只读）
runtime = AgentRuntime(brain=brain, gateway=gateway)
result = runtime.run(
    task_id="DEMO-002",
    role="agent_reader",  # ❌ 只读角色
    goal="发起退款"
)
```

**期望看到**：`[permission_denied]` 被拒绝

---

### 实验 2：重复调用（幂等测试）

```python
brain = ScriptedBrain(
    script=[
        {"thought": "发起退款", "action": {"tool": "issue_refund", "args": {"order_id": "ORD-999", "amount": 100}}},
        {"thought": "再发起一次", "action": {"tool": "issue_refund", "args": {"order_id": "ORD-999", "amount": 100}}},
        {"answer": "退款完成"}
    ],
    repeat_last=False
)
```

**期望看到**：第二次调用返回 `[cached]`，不会真的扣两次钱

---

### 实验 3：死循环（让它疯狂重复）

```python
brain = ScriptedBrain(
    script=[
        {"thought": "查订单", "action": {"tool": "query_order", "args": {"order_id": "ORD-111"}}}
    ],
    repeat_last=True  # ⚠️ 永远重复最后一个动作
)

runtime = AgentRuntime(
    brain=brain,
    gateway=gateway,
    max_steps=10,
    loop_threshold=3  # 连续 3 次相同调用就判定死循环
)
```

**期望看到**：跑 3-4 步就停下，结束原因是 `loop_detected`

---

### 实验 4：参数错误

```python
brain = ScriptedBrain(
    script=[
        {"thought": "查订单", "action": {"tool": "query_order", "args": {"order_id": "不合法的订单号 哈哈哈"}}}
    ],
    repeat_last=False
)
```

**期望看到**：`[invalid_arguments]` 参数校验失败

---

## 📚 第四步：看关键代码（30 分钟）

现在你玩了一遍，知道它"能做什么"了。接下来看"它是怎么做的"。

**只看这 4 个文件，按顺序**：

### 1. [registry.py](registry.py)（5 分钟）
**看什么**：
- `ToolSpec` 类：一个工具要声明什么（权限、参数 schema、超时、是否高危）
- `needs_approval()` 方法：怎么判断"这次调用要不要审批"

**关键点**：工具的权限、高危规则都在这里统一声明，不散落在各处。

---

### 2. [gateway.py](gateway.py)（10 分钟）
**看什么**：
- `ToolGateway.call()` 方法：8 步流水线，顺序不可变
- 幂等键怎么算的：`_make_idempotency_key()`
- 审批单怎么去重的：`_create_or_find_approval()`

**关键点**：
- 第 22-28 行注释：为什么顺序不能变（这是我踩过坑改出来的）
- 幂等键必须带 `task_id`（否则两笔真实业务会被误判为重复）

---

### 3. [runtime.py](runtime.py)（10 分钟）
**看什么**：
- `AgentRuntime.run()` 方法：状态机主循环
- `detect_loop()` 函数：怎么判断死循环（支持连续重复 + 周期性交替）
- `_force_answer()` 方法：触发约束后怎么优雅收尾

**关键点**：
- 第 56-99 行：循环检测的两种模式（原地打转 vs 交替循环）
- 收尾时要换 brain（`ForceAnswerBrain`），否则它还会继续调工具

---

### 4. [evaluation/attribution.py](evaluation/attribution.py)（5 分钟）
**看什么**：
- `Attribution` 枚举：十三种归因类别
- `SEVERITY_ORDER` 列表：优先级排序（为什么幂等失效比参数错误更严重）
- `attribute()` 函数：怎么从轨迹判断归因

**关键点**：
- 归因分成 A 组（现象）和 B 组（机制失效）
- B 组归因非 0 = 保障机制出 bug 了

---

## 🎓 第五步：看测试用例（理解边界场景）

**目标**：知道系统在边界情况下怎么表现。

推荐看这几个测试：

### 1. `tests/test_gateway.py`
- 第 200-230 行：幂等测试（同一任务重复调用 vs 不同任务相同参数）
- 第 250-280 行：审批单去重（重试不会创建多个审批单）

### 2. `tests/test_runtime.py`
- 第 150-180 行：交替循环检测（`A,B,A,B...` 这种也要抓住）
- 第 300-330 行：参数下界校验（`loop_threshold=1` 会误杀正常任务）

### 3. `tests/test_attribution_mechanism.py`
- 第 80-120 行：幂等失效 vs 幂等生效的区别
- 第 200-230 行：优先级测试（同时命中多个归因时，取最严重的）

---

## 🛠️ 第六步：改点东西（动手）

**目标**：通过改代码加深理解。

### 任务 1：加一个新工具

在 `builtin_tools.py` 里加一个"查物流"工具：

```python
class QueryLogisticsArgs(BaseModel):
    tracking_number: str

def query_logistics(args: QueryLogisticsArgs) -> Dict[str, Any]:
    return {
        "tracking_number": args.tracking_number,
        "status": "运输中",
        "location": "北京分拨中心"
    }

# 注册
ToolSpec(
    name="query_logistics",
    description="查询物流信息",
    args_model=QueryLogisticsArgs,
    handler=query_logistics,
    allowed_roles={"agent_staff", "agent_reader"},  # 谁能用
    timeout_seconds=3.0,
    idempotent=True,  # 只读操作
    side_effect=False
)
```

然后在 `playground.py` 里调用它，看是否 work。

---

### 任务 2：改一个约束参数

把 `max_steps` 改成 3，看死循环任务会不会更快停下。

---

### 任务 3：制造一个"机制失效"

**挑战**：能不能让幂等检查失效，导致同一个调用被真实执行两次？

提示：可以试试注释掉 `gateway.py` 里的幂等检查代码，看归因会不会变成 `IDEMPOTENCY_FAILURE`。

---

## 📖 第七步：看完整文档（可选）

如果你想了解"为什么这么设计"、"踩了哪些坑"，再看 [README.md](README.md)。

重点看这几节：
- **典型场景**（第 12 行）：一笔 3000 元退款是怎么走的
- **工程问题档案**（第 279 行）：11 个真实踩坑记录
- **效果验证：A/B 对照实验**（第 54 行）：量化"没有网关会发生什么"

---

## 🎯 总结：你现在应该知道的

1. **核心流程**：THINKING → ACTING（过网关 8 步）→ OBSERVING → 循环或结束
2. **三道防线**：
   - 工具网关：权限 + 幂等 + 审批 + 参数校验
   - 运行时约束：步数上限 + token 预算 + 循环检测
   - 归因系统：告诉你哪一环出了问题
3. **关键设计**：
   - 幂等键带 task_id（区分重试 vs 新业务）
   - 优雅收尾（达到上限不直接报错，而是给阶段性结论）
   - 确定性回放（从轨迹能原样重跑）

---

## 🚀 下一步

- 想**接真实 LLM**：看我的改进建议第 2 条
- 想**生产化部署**：看改进建议第 1 条（Redis + PostgreSQL）
- 想**加更多工具**：看 `builtin_tools.py`，照着写就行

有问题随时问！
