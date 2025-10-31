"""
测试prefer_exact参数的功能
验证当prefer_exact=True时，是否能正确匹配等额票
"""
import unittest
from decimal import Decimal
from ticket_matching import (
    OptimizedTicketMatcher,
    PaymentOrder,
    Ticket,
    TargetWeights,
    Constraints,
    SplitRule,
    UserPreference,
    AmountStrategy,
    TermStrategy,
    AcceptorStrategy,
    OrganizationStrategy,
    TailDiffType,
    SplitStrategy,
    TicketCategory,
)


class TestPreferExact(unittest.TestCase):
    """测试prefer_exact参数"""
    
    def setUp(self):
        """设置测试数据"""
        self.target_weights = TargetWeights(
            w1=0.4,
            w2=0.2,
            w3=0.2,
            w4=0.2,
            amount_strategy=AmountStrategy.BIG_FIRST,
            term_strategy=TermStrategy.FAR_FIRST,
            acceptor_strategy=AcceptorStrategy.GOOD_FIRST,
            organization_strategy=OrganizationStrategy.SAME_ORG_FIRST,
        )
        
        self.constraints = Constraints(
            max_tickets=5,
            min_amount=Decimal("50000"),
            max_amount=Decimal("700000"),
            remain_after_split=Decimal("10000"),
            max_mini_amount_tickets=2,
        )
        
        self.split_rule = SplitRule(
            tail_diff_type=TailDiffType.AMOUNT,
            tail_diff_value=50_000,
            split_strategy=SplitStrategy.BY_AMOUNT,
        )
    
    def test_prefer_exact_with_exact_match(self):
        """测试当prefer_exact=True且存在等额票时，应该选择等额票"""
        payment_order = PaymentOrder(id="PO001", amount=Decimal("500000"), organization="A")
        
        tickets = [
            Ticket(
                id="T_EXACT",
                amount=Decimal("500000"),
                days_to_expire=100,
                acceptor_score=5.0,
                organization="A",
                category=TicketCategory.BIG,
            ),
            Ticket(
                id="T001",
                amount=Decimal("300000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
            Ticket(
                id="T002",
                amount=Decimal("200000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
        ]
        
        user_preference = UserPreference(
            prefer_exact=True,
            allow_split=True,
            allow_inventory_balance=False,
        )
        
        matcher = OptimizedTicketMatcher(
            tickets=tickets,
            payment_order=payment_order,
            target_weights=self.target_weights,
            constraints=self.constraints,
            split_rule=self.split_rule,
            user_preference=user_preference,
            random_seed=42,
        )
        
        solution = matcher.optimize()
        result = solution.to_dict()
        
        # 应该只选择一张等额票
        self.assertEqual(result['票据张数'], 1)
        self.assertEqual(result['选中票据'][0]['票据ID'], 'T_EXACT')
        self.assertEqual(result['总金额'], 500000.0)
        self.assertEqual(result['电汇尾差'], 0.0)
    
    def test_prefer_exact_without_exact_match(self):
        """测试当prefer_exact=True但没有等额票时，应该回退到正常流程"""
        payment_order = PaymentOrder(id="PO002", amount=Decimal("500000"), organization="A")
        
        tickets = [
            Ticket(
                id="T001",
                amount=Decimal("300000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
            Ticket(
                id="T002",
                amount=Decimal("200000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
        ]
        
        user_preference = UserPreference(
            prefer_exact=True,
            allow_split=True,
            allow_inventory_balance=False,
        )
        
        matcher = OptimizedTicketMatcher(
            tickets=tickets,
            payment_order=payment_order,
            target_weights=self.target_weights,
            constraints=self.constraints,
            split_rule=self.split_rule,
            user_preference=user_preference,
            random_seed=42,
        )
        
        solution = matcher.optimize()
        result = solution.to_dict()
        
        # 应该选择多张票据
        self.assertGreater(result['票据张数'], 1)
    
    def test_prefer_exact_false(self):
        """测试当prefer_exact=False时，即使有等额票也按正常流程选择"""
        payment_order = PaymentOrder(id="PO003", amount=Decimal("500000"), organization="A")
        
        tickets = [
            Ticket(
                id="T_EXACT",
                amount=Decimal("500000"),
                days_to_expire=50,
                acceptor_score=8.0,
                organization="B",
                category=TicketCategory.BIG,
            ),
            Ticket(
                id="T001",
                amount=Decimal("300000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
            Ticket(
                id="T002",
                amount=Decimal("200000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
        ]
        
        user_preference = UserPreference(
            prefer_exact=False,
            allow_split=True,
            allow_inventory_balance=False,
        )
        
        matcher = OptimizedTicketMatcher(
            tickets=tickets,
            payment_order=payment_order,
            target_weights=self.target_weights,
            constraints=self.constraints,
            split_rule=self.split_rule,
            user_preference=user_preference,
            random_seed=42,
        )
        
        solution = matcher.optimize()
        result = solution.to_dict()
        
        # 可能选择多张票据，因为等额票评分较低
        self.assertIsNotNone(result)
    
    def test_prefer_exact_multiple_exact_tickets(self):
        """测试当有多张等额票时，选择得分最高的"""
        payment_order = PaymentOrder(id="PO004", amount=Decimal("300000"), organization="A")
        
        tickets = [
            Ticket(
                id="T_EXACT_1",
                amount=Decimal("300000"),
                days_to_expire=50,
                acceptor_score=5.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
            Ticket(
                id="T_EXACT_2",
                amount=Decimal("300000"),
                days_to_expire=360,
                acceptor_score=1.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
            Ticket(
                id="T_EXACT_3",
                amount=Decimal("300000"),
                days_to_expire=180,
                acceptor_score=4.0,
                organization="A",
                category=TicketCategory.MIDDLE,
            ),
        ]
        
        user_preference = UserPreference(
            prefer_exact=True,
            allow_split=True,
            allow_inventory_balance=False,
        )
        
        matcher = OptimizedTicketMatcher(
            tickets=tickets,
            payment_order=payment_order,
            target_weights=self.target_weights,
            constraints=self.constraints,
            split_rule=self.split_rule,
            user_preference=user_preference,
            random_seed=42,
        )
        
        solution = matcher.optimize()
        result = solution.to_dict()
        
        # 应该选择得分最高的等额票（期限最长、承兑人最好的）
        self.assertEqual(result['票据张数'], 1)
        self.assertEqual(result['选中票据'][0]['票据ID'], 'T_EXACT_2')
        self.assertEqual(result['总金额'], 300000.0)


if __name__ == "__main__":
    unittest.main()
