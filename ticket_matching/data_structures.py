"""
数据结构定义
定义票据配票系统所需的所有数据类
"""
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class AmountStrategy(Enum):
    """金额策略"""
    BIG_FIRST = "big_first"  # 大额优先
    SMALL_FIRST = "small_first"  # 小额优先
    RANDOM = "random"  # 金额随机
    CLOSE_BELOW = "close_below"  # 接近但小于交易单金额
    CLOSE_ABOVE = "close_above"  # 接近但大于交易单金额
    OPTIMIZE_INVENTORY = "optimize_inventory"  # 优化期望库存票据占比


class AmountSubStrategy(Enum):
    """金额子策略"""
    RANDOM = "random"  # 随机
    BIG_TO_SMALL = "big_to_small"  # 从大到小
    SMALL_TO_BIG = "small_to_big"  # 从小到大
    CLOSE_ABOVE_BIAS = "close_above_bias"  # 接近但大于差额


class TermStrategy(Enum):
    """期限策略"""
    FAR_FIRST = "far_first"  # 优先远期
    NEAR_FIRST = "near_first"  # 优先近期


class AcceptorStrategy(Enum):
    """承兑人策略"""
    GOOD_FIRST = "good_first"  # 优先好
    BAD_FIRST = "bad_first"  # 优先差


class OrganizationStrategy(Enum):
    """组织策略"""
    SAME_ORG_FIRST = "same_org_first"  # 优先使用当前付款单所属组织的票
    DIFF_ORG_FIRST = "diff_org_first"  # 优先使用其他组织的票


class SplitStrategy(Enum):
    """拆票策略"""
    BY_TERM = "by_term"  # 按期限
    BY_ACCEPTOR = "by_acceptor"  # 按承兑人
    BY_AMOUNT = "by_amount"  # 按金额


class TailDiffType(Enum):
    """尾差类型"""
    PERCENTAGE = "percentage"  # 尾差占付款单金额占比
    UNLIMITED = "unlimited"  # 无限制
    AMOUNT = "amount"  # 尾差金额


class SplitConditionType(Enum):
    """拆票条件类型"""
    UNLIMITED = "unlimited"  # 无限制
    WITHIN_TAIL_DIFF = "within_tail_diff"  # 差额范围内无需拆分（走电汇尾差补齐）


class TicketCategory(Enum):
    """票据分类"""
    BIG = "big"  # 大票
    MIDDLE = "middle"  # 中票
    SMALL = "small"  # 小票


@dataclass
class Ticket:
    """票据"""
    id: str  # 票据ID
    amount: Decimal  # 票面金额
    days_to_expire: int  # 到期天数
    acceptor_score: float  # 承兑人评分 [1-8]，数值越小越好
    organization: str  # 所属组织
    category: Optional[TicketCategory] = None  # 票据分类


@dataclass
class PaymentOrder:
    """付款单"""
    id: str  # 付款单ID
    amount: Decimal  # 付款单金额
    organization: str  # 所属组织


@dataclass
class TargetWeights:
    """目标权重配置"""
    w1: float  # 金额权重
    w2: float  # 期限权重
    w3: float  # 承兑人权重
    w4: float  # 组织权重
    amount_strategy: AmountStrategy  # 金额策略
    amount_sub_strategy: Optional[AmountSubStrategy] = None  # 金额子策略
    term_strategy: TermStrategy = TermStrategy.FAR_FIRST  # 期限策略
    term_threshold: Optional[int] = None  # 期限阈值（优先远/优先近时生效）
    acceptor_strategy: AcceptorStrategy = AcceptorStrategy.GOOD_FIRST  # 承兑人策略
    organization_strategy: OrganizationStrategy = OrganizationStrategy.SAME_ORG_FIRST  # 组织策略

    def __post_init__(self):
        """验证权重和为1"""
        total = self.w1 + self.w2 + self.w3 + self.w4
        if not (0.99 <= total <= 1.01):  # 允许浮点误差
            raise ValueError(f"权重和必须为1，当前为{total}")
        if self.term_threshold is not None and self.term_threshold < 0:
            raise ValueError("term_threshold 不能为负数")


@dataclass
class SplitRule:
    """拆票规则"""
    tail_diff_type: TailDiffType  # 尾差类型
    tail_diff_value: float  # 尾差值
    split_strategy: SplitStrategy  # 拆票策略
    split_sub_strategy: Optional[AmountSubStrategy] = None  # 拆票子策略（仅当split_strategy为BY_AMOUNT时使用）
    split_condition_type: SplitConditionType = SplitConditionType.UNLIMITED  # 拆票条件类型（无限制/差额范围内无需拆分）
    split_min_used_amount: Decimal = Decimal("50000")  # 拆分使用金额（差额）最小值（默认5万）
    split_min_ratio: Decimal = Decimal("0.3")  # 拆分金额占整张票据比例最小值（默认30%）

    def __post_init__(self) -> None:
        if not isinstance(self.split_min_used_amount, Decimal):
            self.split_min_used_amount = Decimal(str(self.split_min_used_amount))
        if not isinstance(self.split_min_ratio, Decimal):
            self.split_min_ratio = Decimal(str(self.split_min_ratio))
        if self.split_min_ratio <= 0 or self.split_min_ratio > 1:
            raise ValueError("split_min_ratio 必须位于(0, 1]区间")


@dataclass
class Constraints:
    """约束条件"""
    max_tickets: int  # 单张交易单最多可用票据张数
    min_amount: Decimal  # 最小可用票据金额
    max_amount: Decimal  # 最大可用票据金额
    remain_after_split: Decimal  # 拆分后留存金额阈值
    max_mini_amount_tickets: int = 2  # 极小金额票据数量上限（默认2张）


@dataclass
class InventoryInfo:
    """库存信息"""
    total_big: int  # 大票总数
    total_middle: int  # 中票总数
    total_small: int  # 小票总数
    target_big_ratio: float  # 期望大票占比
    target_middle_ratio: float  # 期望中票占比
    target_small_ratio: float  # 期望小票占比

    @property
    def total_count(self) -> int:
        """总票据数"""
        return self.total_big + self.total_middle + self.total_small

    @property
    def current_big_ratio(self) -> float:
        """当前大票占比"""
        return self.total_big / self.total_count if self.total_count > 0 else 0

    @property
    def current_middle_ratio(self) -> float:
        """当前中票占比"""
        return self.total_middle / self.total_count if self.total_count > 0 else 0

    @property
    def current_small_ratio(self) -> float:
        """当前小票占比"""
        return self.total_small / self.total_count if self.total_count > 0 else 0


@dataclass
class UserPreference:
    """用户配票偏好"""
    prefer_exact: bool = False  # 是否优先匹配等额票
    allow_split: bool = True  # 是否允许拆分票据
    allow_inventory_balance: bool = False  # 是否考虑库存平衡
    force_top_weight_dimension: bool = False  # 最高权重维度是否强制穿透
    amount_dist: Optional[Dict[str, List[Decimal]]] = None  # 金额区间划分 {"大额": [min, max], "小额": [min, max]}
    remain_dist: Optional[Dict[str, float]] = None  # 期望剩余票据占比 {"大额": 0.5, "中额": 0.3, "小额": 0.2}


@dataclass
class TicketUsageDetail:
    """票据使用明细"""
    ticket_id: str  # 票据ID
    original_amount: Decimal  # 原始金额
    used_amount: Decimal  # 使用金额
    remain_amount: Decimal  # 留存金额
    is_split: bool  # 是否拆分


@dataclass
class Solution:
    """配票解决方案"""
    selected_tickets: List[TicketUsageDetail] = field(default_factory=list)  # 选中的票据列表
    total_amount: Decimal = Decimal("0")  # 总金额
    wire_transfer_amount: Decimal = Decimal("0")  # 电汇尾差
    split_amount: Decimal = Decimal("0")  # 拆票金额
    total_score: float = 0.0  # 综合得分
    amount_score: float = 0.0  # 金额得分
    term_score: float = 0.0  # 期限得分
    acceptor_score: float = 0.0  # 承兑人得分
    organization_score: float = 0.0  # 组织得分
    inventory_balance_score: float = 0.0  # 库存平衡得分
    big_ticket_count: int = 0  # 大票数量
    middle_ticket_count: int = 0  # 中票数量
    small_ticket_count: int = 0  # 小票数量
    big_ticket_ratio: float = 0.0  # 大票占比
    middle_ticket_ratio: float = 0.0  # 中票占比
    small_ticket_ratio: float = 0.0  # 小票占比
    remaining_inventory: Optional[Dict[str, Any]] = None  # 余票库存分布
    execution_time: float = 0.0  # 执行时间（秒）

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "选中票据": [
                {
                    "票据ID": detail.ticket_id,
                    "原始金额": float(detail.original_amount),
                    "使用金额": float(detail.used_amount),
                    "留存金额": float(detail.remain_amount),
                    "是否拆分": detail.is_split
                }
                for detail in self.selected_tickets
            ],
            "票据张数": len(self.selected_tickets),
            "总金额": float(self.total_amount),
            "电汇尾差": float(self.wire_transfer_amount),
            "拆票金额": float(self.split_amount),
            "综合得分": self.total_score,
            "金额得分": self.amount_score,
            "期限得分": self.term_score,
            "承兑人得分": self.acceptor_score,
            "组织得分": self.organization_score,
            "库存平衡得分": self.inventory_balance_score,
            "选票结构": {
                "大票数量": self.big_ticket_count,
                "中票数量": self.middle_ticket_count,
                "小票数量": self.small_ticket_count,
                "大票占比": self.big_ticket_ratio,
                "中票占比": self.middle_ticket_ratio,
                "小票占比": self.small_ticket_ratio
            },
            "余票库存分布": self.remaining_inventory,
            "执行时间": f"{self.execution_time:.4f}秒"
        }
