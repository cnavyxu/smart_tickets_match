"""
票据配票优化系统
基于多目标组合优化的票据智能配票算法
"""

from .data_structures import (
    AcceptorStrategy,
    AmountStrategy,
    AmountSubStrategy,
    Constraints,
    InventoryInfo,
    OrganizationStrategy,
    PaymentOrder,
    Solution,
    SplitRule,
    SplitStrategy,
    TailDiffType,
    TargetWeights,
    TermStrategy,
    Ticket,
    TicketCategory,
    TicketUsageDetail,
    UserPreference,
)
from .optimized_ticket_matcher_v2 import OptimizedTicketMatcher, generate_test_data, run_example

__all__ = [
    "AcceptorStrategy",
    "AmountStrategy",
    "AmountSubStrategy",
    "Constraints",
    "InventoryInfo",
    "OptimizedTicketMatcher",
    "OrganizationStrategy",
    "PaymentOrder",
    "Solution",
    "SplitRule",
    "SplitStrategy",
    "TailDiffType",
    "TargetWeights",
    "TermStrategy",
    "Ticket",
    "TicketCategory",
    "TicketUsageDetail",
    "UserPreference",
    "generate_test_data",
    "run_example",
]
