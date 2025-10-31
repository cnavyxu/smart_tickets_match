# 票据配票优化系统

基于多目标组合优化的智能票据配票算法，用于解决票据池中选择最优票据组合以匹配付款单金额的问题。

## 项目简介

本项目实现了一个复杂的票据配票系统，能够根据多维度偏好（金额、期限、承兑人、组织）从大规模票据池（7k~10k张票据）中智能选择最优票据组合，支持拆票、库存平衡等高级功能。

### 核心特性

- ✅ **多维度优化**：支持金额、期限、承兑人、组织四个维度的权重配置
- ✅ **多种金额策略**：大额优先、小额优先、接近目标金额、库存优化等
- ✅ **智能拆票**：当无法精确匹配时自动拆分票据
- ✅ **库存平衡**：考虑库存结构，优化剩余票据分布
- ✅ **约束保障**：支持张数限制、金额上下限、极小票限制等多种约束
- ✅ **高效求解**：三阶段混合策略，秒级完成优化
- ✅ **精确计算**：使用 Decimal 类型保证金额精度

## 算法设计

算法采用 **"预筛选 ➜ 贪心构建 ➜ 局部优化"** 的三阶段分层混合策略：

1. **快速预筛选**：根据硬约束筛选候选票据，并计算各维度评分
2. **改进贪心构建**：分轮次逐步挑选高分票据，逐步逼近目标金额
3. **局部优化**：
   - 拆票优化：处理金额差异
   - 交换优化：替换低分票据
   - 库存平衡：微调库存结构

详细的算法设计请参考：
- [建模设计文档.md](./建模设计文档.md) - 数学建模、变量定义、约束体系
- [算法流程说明.md](./算法流程说明.md) - 算法流程与实现细节
- [智能配票算法逻辑.md](./智能配票算法逻辑.md) - 业务逻辑与需求说明

## 快速开始

### 基础用法

```python
from decimal import Decimal
from ticket_matching import (
    OptimizedTicketMatcher,
    PaymentOrder,
    TargetWeights,
    AmountStrategy,
    TermStrategy,
    AcceptorStrategy,
    OrganizationStrategy,
    SplitRule,
    SplitStrategy,
    TailDiffType,
    Constraints,
    UserPreference,
    generate_test_data,
)

# 生成测试数据
tickets, inventory_info = generate_test_data(ticket_count=500, seed=42)

# 创建付款单
payment_order = PaymentOrder(
    id="PO001",
    amount=Decimal("1000000"),  # 100万元
    organization="A"
)

# 配置目标权重（大额优先）
target_weights = TargetWeights(
    w1=0.5,  # 金额权重
    w2=0.2,  # 期限权重
    w3=0.2,  # 承兑人权重
    w4=0.1,  # 组织权重
    amount_strategy=AmountStrategy.BIG_FIRST,
    term_strategy=TermStrategy.FAR_FIRST,
    acceptor_strategy=AcceptorStrategy.GOOD_FIRST,
    organization_strategy=OrganizationStrategy.SAME_ORG_FIRST,
)

# 配置拆票规则
split_rule = SplitRule(
    tail_diff_type=TailDiffType.AMOUNT,
    tail_diff_value=50_000,  # 尾差阈值5万
    split_strategy=SplitStrategy.BY_AMOUNT,
)

# 配置约束条件
constraints = Constraints(
    max_tickets=8,
    min_amount=Decimal("30000"),
    max_amount=Decimal("600000"),
    remain_after_split=Decimal("10000"),
    max_mini_amount_tickets=2,
)

# 用户偏好
user_preference = UserPreference(
    prefer_exact=False,
    allow_split=True,
    allow_inventory_balance=False,
)

# 创建匹配器并执行
matcher = OptimizedTicketMatcher(
    tickets=tickets,
    payment_order=payment_order,
    target_weights=target_weights,
    constraints=constraints,
    split_rule=split_rule,
    user_preference=user_preference,
    inventory_info=inventory_info,
    random_seed=42,
)

solution = matcher.optimize()

# 查看结果
print(f"选中票据: {len(solution.selected_tickets)} 张")
print(f"总金额: {solution.total_amount}")
print(f"综合得分: {solution.total_score:.4f}")
print(f"执行时间: {solution.execution_time:.4f}秒")
```

### 运行示例

```bash
# 运行完整示例
python example_usage.py
```

示例包含：
1. 基础用法 - 大额优先策略
2. 小额优先策略
3. 库存平衡优化

## 核心组件

### 数据结构

#### Ticket（票据）
```python
@dataclass
class Ticket:
    id: str                              # 票据ID
    amount: Decimal                      # 票面金额
    days_to_expire: int                  # 到期天数
    acceptor_score: float                # 承兑人评分 [1-8]
    organization: str                    # 所属组织
    category: Optional[TicketCategory]   # 票据分类（大/中/小）
```

#### PaymentOrder（付款单）
```python
@dataclass
class PaymentOrder:
    id: str           # 付款单ID
    amount: Decimal   # 付款单金额
    organization: str # 所属组织
```

#### TargetWeights（目标权重）
```python
@dataclass
class TargetWeights:
    w1: float                                # 金额权重
    w2: float                                # 期限权重
    w3: float                                # 承兑人权重
    w4: float                                # 组织权重
    amount_strategy: AmountStrategy          # 金额策略
    term_strategy: TermStrategy              # 期限策略
    acceptor_strategy: AcceptorStrategy      # 承兑人策略
    organization_strategy: OrganizationStrategy  # 组织策略
```

### 策略类型

#### AmountStrategy（金额策略）
- `BIG_FIRST` - 大额优先
- `SMALL_FIRST` - 小额优先
- `RANDOM` - 随机选择
- `CLOSE_BELOW` - 接近但小于目标金额
- `CLOSE_ABOVE` - 接近但大于目标金额
- `OPTIMIZE_INVENTORY` - 优化库存占比

#### TermStrategy（期限策略）
- `FAR_FIRST` - 优先远期
- `NEAR_FIRST` - 优先近期

#### AcceptorStrategy（承兑人策略）
- `GOOD_FIRST` - 优先好承兑人
- `BAD_FIRST` - 优先差承兑人

#### OrganizationStrategy（组织策略）
- `SAME_ORG_FIRST` - 优先同组织
- `DIFF_ORG_FIRST` - 优先跨组织

## 配置说明

### 约束条件

```python
Constraints(
    max_tickets=10,                    # 单笔交易最多可用票据张数
    min_amount=Decimal("30000"),       # 单票最小金额
    max_amount=Decimal("600000"),      # 单票最大金额
    remain_after_split=Decimal("10000"), # 拆分后留存金额阈值
    max_mini_amount_tickets=2,         # 极小金额票据数量上限
)
```

### 拆票规则

```python
SplitRule(
    tail_diff_type=TailDiffType.AMOUNT,  # 尾差类型（AMOUNT/PERCENTAGE/UNLIMITED）
    tail_diff_value=50_000,              # 尾差值
    split_strategy=SplitStrategy.BY_AMOUNT, # 拆票策略（BY_AMOUNT/BY_TERM/BY_ACCEPTOR）
)
```

### 用户偏好

```python
UserPreference(
    prefer_exact=False,                  # 是否优先匹配等额票
    allow_split=True,                    # 是否允许拆票
    allow_inventory_balance=True,        # 是否考虑库存平衡
    remain_dist={                        # 期望剩余库存占比
        "大额": 0.4,
        "中额": 0.4,
        "小额": 0.2,
    },
)
```

## 输出说明

### Solution（解决方案）

```python
solution = matcher.optimize()

# 基本信息
solution.selected_tickets          # 选中的票据列表
solution.total_amount              # 总金额
solution.wire_transfer_amount      # 电汇尾差
solution.split_amount              # 拆票金额

# 评分信息
solution.total_score               # 综合得分
solution.amount_score              # 金额得分
solution.term_score                # 期限得分
solution.acceptor_score            # 承兑人得分
solution.organization_score        # 组织得分
solution.inventory_balance_score   # 库存平衡得分

# 结构信息
solution.big_ticket_count          # 大票数量
solution.middle_ticket_count       # 中票数量
solution.small_ticket_count        # 小票数量
solution.remaining_inventory       # 余票库存分布

# 执行信息
solution.execution_time            # 执行时间（秒）

# 转换为字典格式
result_dict = solution.to_dict()
```

## 性能

- **票据池规模**：支持 7k~10k 张票据
- **执行时间**：通常在 1 秒内完成
- **内存占用**：约 100MB（取决于票据池大小）
- **精度保证**：使用 Decimal 类型，金额精确到分（0.01元）

## 项目结构

```
.
├── README.md                               # 本文件
├── 建模设计文档.md                          # 数学建模文档
├── 算法流程说明.md                          # 算法流程说明
├── 智能配票算法逻辑.md                      # 业务逻辑文档
├── example_usage.py                        # 使用示例
└── ticket_matching/                        # 核心代码包
    ├── __init__.py                         # 包初始化
    ├── data_structures.py                  # 数据结构定义
    └── optimized_ticket_matcher_v2.py      # 核心算法实现
```

## 技术栈

- Python 3.7+
- 标准库：dataclasses, decimal, random, time

## 扩展建议

基于文档提出的后续扩展方向：

1. **库存微调增强**：在库存平衡阶段加入基于偏差的替换策略
2. **分布式预筛**：支持并行处理超大规模票据池
3. **拆票策略多样化**：支持混合评分的拆票选择
4. **混合整数规划**：在极端场景下引入 MIP 求解器作为备选方案

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

---

**注意**：本系统设计用于处理金融票据配票场景，在生产环境使用前请充分测试并根据实际业务需求调整参数。
