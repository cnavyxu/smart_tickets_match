"""
优化票据配票算法实现
基于《建模设计文档.md》和《算法流程说明.md》实现预筛选 ➜ 改进贪心构建 ➜ 局部优化的三阶段混合策略
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
import math
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

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

# 为 Decimal 运算设定默认精度
getcontext().prec = 12

# ============================
# 数据结构定义
# ============================


@dataclass
class CandidateTicket:
    """候选票据，包含预计算的得分信息"""

    ticket: Ticket
    amount_score: float
    term_score: float
    acceptor_score: float
    organization_score: float
    total_score: float
    is_mini_amount: bool


@dataclass
class SelectedTicket:
    """内部使用的选票结构"""

    ticket: Ticket
    used_amount: Decimal
    remain_amount: Decimal
    is_split: bool
    scores: Tuple[float, float, float, float, float]

    @property
    def total_amount(self) -> Decimal:
        return self.used_amount


# ============================
# 辅助函数
# ============================


def quantize_decimal(value: Decimal) -> Decimal:
    """统一金额精度，保留两位小数"""

    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def safe_divide(numerator: float, denominator: float) -> float:
    """安全除法，避免被0除"""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize(values: Iterable[float]) -> List[float]:
    """线性归一化，映射到[0,1]"""

    values = list(values)
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return [0.5 for _ in values]
    return [safe_divide(v - v_min, v_max - v_min) for v in values]


def normalize_desc(values: Iterable[float]) -> List[float]:
    """逆序归一化，最大值映射为1"""

    normalized = normalize(values)
    return [1 - v for v in normalized]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """限制值在给定范围内"""

    return max(lower, min(upper, value))


# ============================
# 核心算法实现
# ============================


class OptimizedTicketMatcher:
    """票据配票优化算法入口"""

    def __init__(
        self,
        tickets: List[Ticket],
        payment_order: PaymentOrder,
        target_weights: TargetWeights,
        constraints: Constraints,
        split_rule: SplitRule,
        user_preference: UserPreference,
        inventory_info: Optional[InventoryInfo] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.tickets = tickets
        self.payment_order = payment_order
        self.target_weights = target_weights
        self.constraints = constraints
        self.split_rule = split_rule
        self.user_preference = user_preference
        self.inventory_info = inventory_info
        self.random = random.Random(random_seed)

        # 缓存变量
        self.tail_threshold = self._calculate_tail_threshold()
        self._candidate_map: Dict[str, CandidateTicket] = {}
        self._unused_candidates: List[CandidateTicket] = []
        self._selected: List[SelectedTicket] = []

    # ---------- 核心公开方法 ----------

    def optimize(self) -> Solution:
        """执行完整配票流程"""

        start_time = time.perf_counter()

        candidates = self._build_candidates()
        filtered_candidates = self._fast_prefilter(candidates)

        # 如果优先匹配等额票，先检查是否有等额票
        if self.user_preference.prefer_exact:
            exact_match = self._find_exact_match_ticket(filtered_candidates)
            if exact_match:
                # 找到等额票，直接使用
                self._selected = [exact_match]
                selected_ids = {exact_match.ticket.id}
                self._unused_candidates = [
                    candidate for candidate in filtered_candidates if candidate.ticket.id not in selected_ids
                ]
                solution = self._create_solution()
                solution.execution_time = time.perf_counter() - start_time
                return solution

        self._selected = self._improved_greedy_construct(filtered_candidates)
        selected_ids = {selected.ticket.id for selected in self._selected}
        self._unused_candidates = [
            candidate for candidate in filtered_candidates if candidate.ticket.id not in selected_ids
        ]

        self._optimize_split()
        self._swap_optimization()
        self._balance_inventory()

        solution = self._create_solution()
        solution.execution_time = time.perf_counter() - start_time
        return solution

    # ---------- 阶段一：数据准备与预筛选 ----------

    def _build_candidates(self) -> List[CandidateTicket]:
        """构建候选票据并计算各维度得分"""

        amount_scores = self._compute_amount_scores(self.tickets)
        term_scores = self._compute_term_scores(self.tickets)
        acceptor_scores = self._compute_acceptor_scores(self.tickets)
        organization_scores = self._compute_organization_scores(self.tickets)

        candidates: List[CandidateTicket] = []
        for ticket in self.tickets:
            amount_score = amount_scores.get(ticket.id, 0.0)
            term_score = term_scores.get(ticket.id, 0.0)
            acceptor_score = acceptor_scores.get(ticket.id, 0.0)
            org_score = organization_scores.get(ticket.id, 0.0)
            total_score = (
                self.target_weights.w1 * amount_score
                + self.target_weights.w2 * term_score
                + self.target_weights.w3 * acceptor_score
                + self.target_weights.w4 * org_score
            )
            candidate = CandidateTicket(
                ticket=ticket,
                amount_score=amount_score,
                term_score=term_score,
                acceptor_score=acceptor_score,
                organization_score=org_score,
                total_score=total_score,
                is_mini_amount=ticket.amount < self.constraints.min_amount * 2,
            )
            candidates.append(candidate)
            self._candidate_map[ticket.id] = candidate

        # 按综合评分从高到低排序
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        return candidates

    def _fast_prefilter(self, candidates: List[CandidateTicket]) -> List[CandidateTicket]:
        """快速预筛选，保留满足硬约束的候选票据"""

        filtered: List[CandidateTicket] = []
        for candidate in candidates:
            ticket = candidate.ticket
            if ticket.amount < self.constraints.min_amount:
                continue
            if ticket.amount > self.constraints.max_amount:
                continue
            filtered.append(candidate)
        return filtered

    def _find_exact_match_ticket(self, candidates: List[CandidateTicket]) -> Optional[SelectedTicket]:
        """在候选列表中找到金额等于付款单金额的最佳票据"""

        target_amount = quantize_decimal(self.payment_order.amount)
        exact_candidates = [
            candidate
            for candidate in candidates
            if quantize_decimal(candidate.ticket.amount) == target_amount
        ]
        if not exact_candidates:
            return None
        best_candidate = max(exact_candidates, key=lambda c: c.total_score)
        return SelectedTicket(
            ticket=best_candidate.ticket,
            used_amount=quantize_decimal(best_candidate.ticket.amount),
            remain_amount=Decimal("0"),
            is_split=False,
            scores=(
                best_candidate.amount_score,
                best_candidate.term_score,
                best_candidate.acceptor_score,
                best_candidate.organization_score,
                best_candidate.total_score,
            ),
        )

    # ---------- 阶段二：改进贪心构建 ----------

    def _improved_greedy_construct(self, candidates: List[CandidateTicket]) -> List[SelectedTicket]:
        """使用三轮策略构建初始组合"""

        rounds = [
            (Decimal("1.00"), "strict"),
            (Decimal("1.10"), "moderate"),
            (Decimal("1.20"), "relaxed"),
        ]

        selected: List[SelectedTicket] = []
        selected_ids = set()
        total_amount = Decimal("0")
        mini_count = 0

        target_amount = self.payment_order.amount

        for limit_factor, _ in rounds:
            limit_amount = target_amount * limit_factor
            for candidate in candidates:
                if candidate.ticket.id in selected_ids:
                    continue
                if len(selected) >= self.constraints.max_tickets:
                    break
                if candidate.is_mini_amount and mini_count >= self.constraints.max_mini_amount_tickets:
                    continue

                new_total = total_amount + candidate.ticket.amount
                if new_total > limit_amount and limit_factor == Decimal("1.00"):
                    continue

                selected.append(
                    SelectedTicket(
                        ticket=candidate.ticket,
                        used_amount=quantize_decimal(candidate.ticket.amount),
                        remain_amount=Decimal("0"),
                        is_split=False,
                        scores=(
                            candidate.amount_score,
                            candidate.term_score,
                            candidate.acceptor_score,
                            candidate.organization_score,
                            candidate.total_score,
                        ),
                    )
                )
                selected_ids.add(candidate.ticket.id)
                total_amount = quantize_decimal(new_total)
                if candidate.is_mini_amount:
                    mini_count += 1

                if total_amount >= target_amount:
                    break
            if total_amount >= target_amount:
                break

        return selected

    # ---------- 阶段三：局部优化 ----------

    def _optimize_split(self) -> None:
        """根据尾差优化拆票，调整金额与目标金额的差距"""

        if not self.user_preference.allow_split:
            return

        bias_amount = self.payment_order.amount - self._current_total_amount()
        if abs(bias_amount) <= self.tail_threshold:
            return

        if bias_amount > 0:
            self._split_from_pool(bias_amount)
        else:
            self._split_from_current(-bias_amount)

    def _split_from_pool(self, bias_amount: Decimal) -> None:
        """从未选票据中拆分出差额"""

        sorted_candidates = sorted(
            [c for c in self._unused_candidates if c.ticket.amount > bias_amount],
            key=lambda c: (c.ticket.amount - bias_amount, -c.total_score),
        )

        for candidate in sorted_candidates:
            if len(self._selected) >= self.constraints.max_tickets:
                break
            ticket = candidate.ticket
            remain = ticket.amount - bias_amount
            if remain < self.constraints.remain_after_split:
                continue

            used_amount = quantize_decimal(bias_amount)
            remain_amount = quantize_decimal(remain)
            self._selected.append(
                SelectedTicket(
                    ticket=ticket,
                    used_amount=used_amount,
                    remain_amount=remain_amount,
                    is_split=True,
                    scores=(
                        candidate.amount_score,
                        candidate.term_score,
                        candidate.acceptor_score,
                        candidate.organization_score,
                        candidate.total_score,
                    ),
                )
            )
            self._unused_candidates.remove(candidate)
            return

    def _split_from_current(self, excess_amount: Decimal) -> None:
        """对当前选票进行拆分，减少超额金额"""

        sorted_selected = sorted(
            [s for s in self._selected if s.used_amount > excess_amount],
            key=lambda s: (s.used_amount - excess_amount, -s.scores[-1]),
        )

        for selected in sorted_selected:
            remain = selected.used_amount - excess_amount
            if remain < self.constraints.remain_after_split:
                continue

            selected.used_amount = quantize_decimal(remain)
            selected.remain_amount = quantize_decimal(
                selected.ticket.amount - selected.used_amount
            )
            selected.is_split = True
            return

    def _swap_optimization(self) -> None:
        """尝试通过交换提升得分或减少尾差"""

        target_amount = self.payment_order.amount
        current_total = self._current_total_amount()
        current_bias = abs(target_amount - current_total)

        if not self._unused_candidates or not self._selected:
            return

        improvement_threshold = 1.001  # 至少提升0.1%
        for _ in range(10):  # 限制迭代次数，避免过长
            swap_made = False
            for candidate in sorted(self._unused_candidates, key=lambda c: c.total_score, reverse=True):
                for selected in sorted(self._selected, key=lambda s: s.scores[-1]):
                    new_total = current_total - selected.used_amount + candidate.ticket.amount
                    new_bias = abs(target_amount - new_total)
                    if new_bias > current_bias * Decimal("1.5"):
                        continue
                    if candidate.total_score <= selected.scores[-1] * improvement_threshold:
                        continue
                    if candidate.is_mini_amount and self._count_mini_amount() >= self.constraints.max_mini_amount_tickets:
                        continue

                    # 执行交换
                    self._selected.remove(selected)
                    self._selected.append(
                        SelectedTicket(
                            ticket=candidate.ticket,
                            used_amount=quantize_decimal(candidate.ticket.amount),
                            remain_amount=Decimal("0"),
                            is_split=False,
                            scores=(
                                candidate.amount_score,
                                candidate.term_score,
                                candidate.acceptor_score,
                                candidate.organization_score,
                                candidate.total_score,
                            ),
                        )
                    )
                    self._unused_candidates.remove(candidate)
                    self._unused_candidates.append(
                        CandidateTicket(
                            ticket=selected.ticket,
                            amount_score=selected.scores[0],
                            term_score=selected.scores[1],
                            acceptor_score=selected.scores[2],
                            organization_score=selected.scores[3],
                            total_score=selected.scores[4],
                            is_mini_amount=selected.ticket.amount < self.constraints.min_amount * 2,
                        )
                    )
                    current_total = new_total
                    current_bias = abs(target_amount - current_total)
                    swap_made = True
                    break
                if swap_made:
                    break
            if not swap_made:
                break

    def _balance_inventory(self) -> None:
        """库存平衡微调"""

        if not self.user_preference.allow_inventory_balance or not self.inventory_info:
            return
        # 当前实现仅计算库存偏差，不做替换（留待扩展）

    # ---------- 评分计算 ----------

    def _compute_amount_scores(self, tickets: List[Ticket]) -> Dict[str, float]:
        amounts = [float(ticket.amount) for ticket in tickets]
        normalized_amounts = normalize(amounts)
        amount_map = {ticket.id: score for ticket, score in zip(tickets, normalized_amounts)}

        strategy = self.target_weights.amount_strategy
        target_amount = float(self.payment_order.amount)

        if strategy == AmountStrategy.BIG_FIRST:
            scores = normalize(amounts)
            return {ticket.id: score for ticket, score in zip(tickets, scores)}
        elif strategy == AmountStrategy.SMALL_FIRST:
            scores = normalize_desc(amounts)
            return {ticket.id: score for ticket, score in zip(tickets, scores)}
        elif strategy == AmountStrategy.CLOSE_BELOW:
            scores = []
            for ticket in tickets:
                amount = float(ticket.amount)
                if amount > target_amount:
                    scores.append(0.0)
                else:
                    scores.append(1 - (target_amount - amount) / target_amount)
            return {ticket.id: clamp(score) for ticket, score in zip(tickets, scores)}
        elif strategy == AmountStrategy.CLOSE_ABOVE:
            scores = []
            for ticket in tickets:
                amount = float(ticket.amount)
                if amount < target_amount:
                    scores.append(0.0)
                else:
                    scores.append(1 - (amount - target_amount) / target_amount)
            return {ticket.id: clamp(score) for ticket, score in zip(tickets, scores)}
        elif strategy == AmountStrategy.OPTIMIZE_INVENTORY:
            if not self.inventory_info:
                return amount_map
            weights = self._inventory_priority_weights()
            scores = []
            for ticket in tickets:
                weight = weights.get(ticket.category or TicketCategory.MIDDLE, 0.5)
                scores.append(weight)
            return {ticket.id: score for ticket, score in zip(tickets, scores)}
        elif strategy == AmountStrategy.RANDOM:
            scores = [self.random.random() for _ in tickets]
            return {ticket.id: score for ticket, score in zip(tickets, scores)}
        return amount_map

    def _inventory_priority_weights(self) -> Dict[TicketCategory, float]:
        if not self.inventory_info or not self.user_preference.remain_dist:
            return {}

        current_ratios = {
            TicketCategory.BIG: self.inventory_info.current_big_ratio,
            TicketCategory.MIDDLE: self.inventory_info.current_middle_ratio,
            TicketCategory.SMALL: self.inventory_info.current_small_ratio,
        }
        target_ratios = {
            TicketCategory.BIG: self.user_preference.remain_dist.get("大额", self.inventory_info.target_big_ratio),
            TicketCategory.MIDDLE: self.user_preference.remain_dist.get("中额", self.inventory_info.target_middle_ratio),
            TicketCategory.SMALL: self.user_preference.remain_dist.get("小额", self.inventory_info.target_small_ratio),
        }
        priority = {}
        for category, current_ratio in current_ratios.items():
            target_ratio = target_ratios.get(category, current_ratio)
            gap = max(target_ratio - current_ratio, 0.0)
            priority[category] = gap
        total = sum(priority.values())
        if total == 0:
            return {category: 1 / 3 for category in TicketCategory}
        return {category: value / total for category, value in priority.items()}

    def _compute_term_scores(self, tickets: List[Ticket]) -> Dict[str, float]:
        days = [ticket.days_to_expire for ticket in tickets]
        normalized_days = normalize(days)
        if self.target_weights.term_strategy == TermStrategy.FAR_FIRST:
            return {ticket.id: score for ticket, score in zip(tickets, normalized_days)}
        else:
            return {ticket.id: score for ticket, score in zip(tickets, normalize_desc(days))}

    def _compute_acceptor_scores(self, tickets: List[Ticket]) -> Dict[str, float]:
        scores = [ticket.acceptor_score for ticket in tickets]
        normalized_scores = normalize(scores)
        if self.target_weights.acceptor_strategy == AcceptorStrategy.GOOD_FIRST:
            return {ticket.id: 1 - score for ticket, score in zip(tickets, normalized_scores)}
        else:
            return {ticket.id: score for ticket, score in zip(tickets, normalized_scores)}

    def _compute_organization_scores(self, tickets: List[Ticket]) -> Dict[str, float]:
        same_org_scores = []
        for ticket in tickets:
            if ticket.organization == self.payment_order.organization:
                same_org_scores.append(1.0)
            else:
                same_org_scores.append(0.0)
        if self.target_weights.organization_strategy == OrganizationStrategy.SAME_ORG_FIRST:
            return {ticket.id: score for ticket, score in zip(tickets, same_org_scores)}
        else:
            return {ticket.id: 1 - score for ticket, score in zip(tickets, same_org_scores)}

    # ---------- Solution 构建 ----------

    def _create_solution(self) -> Solution:
        total_amount = quantize_decimal(self._current_total_amount())
        target_amount = quantize_decimal(self.payment_order.amount)
        wire_transfer_amount = quantize_decimal(target_amount - total_amount)
        split_amount = sum(
            (ticket.ticket.amount - ticket.used_amount)
            for ticket in self._selected
            if ticket.is_split
        ) or Decimal("0")

        amount_score, term_score, acceptor_score, organization_score = 0.0, 0.0, 0.0, 0.0
        if self._selected:
            amount_score = sum(ticket.scores[0] for ticket in self._selected) / len(self._selected)
            term_score = sum(ticket.scores[1] for ticket in self._selected) / len(self._selected)
            acceptor_score = sum(ticket.scores[2] for ticket in self._selected) / len(self._selected)
            organization_score = sum(ticket.scores[3] for ticket in self._selected) / len(self._selected)
        inventory_balance_score = self._calc_inventory_balance_score_for_solution()
        total_score = (
            self.target_weights.w1 * amount_score
            + self.target_weights.w2 * term_score
            + self.target_weights.w3 * acceptor_score
            + self.target_weights.w4 * organization_score
        )
        if self.user_preference.allow_inventory_balance:
            total_score = total_score * 0.8 + inventory_balance_score * 0.2

        big_count, middle_count, small_count = self._calc_ticket_structure()
        total_count = max(len(self._selected), 1)

        solution = Solution(
            selected_tickets=[
                TicketUsageDetail(
                    ticket_id=ticket.ticket.id,
                    original_amount=quantize_decimal(ticket.ticket.amount),
                    used_amount=quantize_decimal(ticket.used_amount),
                    remain_amount=quantize_decimal(ticket.remain_amount),
                    is_split=ticket.is_split,
                )
                for ticket in self._selected
            ],
            total_amount=total_amount,
            wire_transfer_amount=wire_transfer_amount,
            split_amount=quantize_decimal(split_amount),
            total_score=total_score,
            amount_score=amount_score,
            term_score=term_score,
            acceptor_score=acceptor_score,
            organization_score=organization_score,
            inventory_balance_score=inventory_balance_score,
            big_ticket_count=big_count,
            middle_ticket_count=middle_count,
            small_ticket_count=small_count,
            big_ticket_ratio=big_count / total_count,
            middle_ticket_ratio=middle_count / total_count,
            small_ticket_ratio=small_count / total_count,
            remaining_inventory=self._calc_remaining_inventory(),
        )
        return solution

    # ---------- 辅助计算 ----------

    def _calculate_tail_threshold(self) -> Decimal:
        if self.split_rule.tail_diff_type == TailDiffType.UNLIMITED:
            return Decimal("Infinity")
        elif self.split_rule.tail_diff_type == TailDiffType.PERCENTAGE:
            return quantize_decimal(self.payment_order.amount * Decimal(str(self.split_rule.tail_diff_value)))
        else:
            return quantize_decimal(Decimal(str(self.split_rule.tail_diff_value)))

    def _current_total_amount(self) -> Decimal:
        return sum(ticket.used_amount for ticket in self._selected)

    def _count_mini_amount(self) -> int:
        return sum(
            1 for ticket in self._selected if ticket.ticket.amount < self.constraints.min_amount * 2
        )

    def _calc_ticket_structure(self) -> Tuple[int, int, int]:
        big_count = sum(1 for ticket in self._selected if ticket.ticket.category == TicketCategory.BIG)
        middle_count = sum(
            1 for ticket in self._selected if ticket.ticket.category == TicketCategory.MIDDLE
        )
        small_count = sum(
            1 for ticket in self._selected if ticket.ticket.category == TicketCategory.SMALL
        )
        return big_count, middle_count, small_count

    def _calc_inventory_balance_score_for_solution(self) -> float:
        if not self.user_preference.allow_inventory_balance or not self.inventory_info:
            return 0.0
        if not self.user_preference.remain_dist:
            return 0.0

        target_ratios = self.user_preference.remain_dist
        remaining = self._calc_remaining_inventory()
        if not remaining:
            return 0.0
        remaining_ratios = remaining.get("占比", {})
        differences = []
        for key, target_ratio in target_ratios.items():
            remaining_ratio = remaining_ratios.get(key, target_ratio)
            differences.append(1 - abs(target_ratio - remaining_ratio))
        if not differences:
            return 0.0
        return sum(differences) / len(differences)

    def _calc_remaining_inventory(self) -> Optional[Dict[str, Dict[str, float]]]:
        if not self.inventory_info:
            return None

        used_big, used_middle, used_small = self._calc_ticket_structure()
        remaining_big = max(self.inventory_info.total_big - used_big, 0)
        remaining_middle = max(self.inventory_info.total_middle - used_middle, 0)
        remaining_small = max(self.inventory_info.total_small - used_small, 0)
        total_remaining = remaining_big + remaining_middle + remaining_small
        if total_remaining == 0:
            return {
                "数量": {
                    "大额": 0,
                    "中额": 0,
                    "小额": 0,
                },
                "占比": {
                    "大额": 0.0,
                    "中额": 0.0,
                    "小额": 0.0,
                },
            }
        return {
            "数量": {
                "大额": remaining_big,
                "中额": remaining_middle,
                "小额": remaining_small,
            },
            "占比": {
                "大额": remaining_big / total_remaining,
                "中额": remaining_middle / total_remaining,
                "小额": remaining_small / total_remaining,
            },
        }


# ============================
# 示例数据与执行
# ============================


def generate_test_data(
    ticket_count: int = 500,
    seed: Optional[int] = None,
) -> Tuple[List[Ticket], InventoryInfo]:
    random_gen = random.Random(seed)
    tickets: List[Ticket] = []
    big, middle, small = 0, 0, 0
    for i in range(ticket_count):
        amount = Decimal(str(random_gen.choice([50_000, 100_000, 200_000, 500_000])))
        days_to_expire = random_gen.randint(10, 360)
        acceptor_score = random_gen.randint(1, 8)
        organization = random_gen.choice(["A", "B", "C"])
        category = random_gen.choice(list(TicketCategory))
        if category == TicketCategory.BIG:
            amount = Decimal(str(random_gen.randint(300_000, 600_000)))
            big += 1
        elif category == TicketCategory.MIDDLE:
            amount = Decimal(str(random_gen.randint(100_000, 300_000)))
            middle += 1
        else:
            amount = Decimal(str(random_gen.randint(30_000, 100_000)))
            small += 1
        tickets.append(
            Ticket(
                id=f"T{i:04d}",
                amount=quantize_decimal(amount),
                days_to_expire=days_to_expire,
                acceptor_score=acceptor_score,
                organization=organization,
                category=category,
            )
        )
    inventory_info = InventoryInfo(
        total_big=big,
        total_middle=middle,
        total_small=small,
        target_big_ratio=0.3,
        target_middle_ratio=0.4,
        target_small_ratio=0.3,
    )
    return tickets, inventory_info


def run_example() -> Dict:
    tickets, inventory_info = generate_test_data(ticket_count=300, seed=42)
    payment_order = PaymentOrder(id="PO001", amount=Decimal("800000"), organization="A")
    target_weights = TargetWeights(
        w1=0.4,
        w2=0.2,
        w3=0.2,
        w4=0.2,
        amount_strategy=AmountStrategy.BIG_FIRST,
        term_strategy=TermStrategy.FAR_FIRST,
        acceptor_strategy=AcceptorStrategy.GOOD_FIRST,
        organization_strategy=OrganizationStrategy.SAME_ORG_FIRST,
    )
    split_rule = SplitRule(
        tail_diff_type=TailDiffType.AMOUNT,
        tail_diff_value=50_000,
        split_strategy=SplitStrategy.BY_AMOUNT,
    )
    constraints = Constraints(
        max_tickets=5,
        min_amount=Decimal("30000"),
        max_amount=Decimal("600000"),
        remain_after_split=Decimal("10000"),
        max_mini_amount_tickets=2,
    )
    user_preference = UserPreference(
        prefer_exact=False,
        allow_split=True,
        allow_inventory_balance=True,
        remain_dist={"大额": 0.4, "中额": 0.4, "小额": 0.2},
    )

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
    return solution.to_dict()


if __name__ == "__main__":
    result = run_example()
    for key, value in result.items():
        print(key, ":", value)
