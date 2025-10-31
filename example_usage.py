"""
票据配票系统使用示例
展示如何使用 OptimizedTicketMatcher 进行配票优化
"""
from decimal import Decimal
from ticket_matching import (
    AcceptorStrategy,
    AmountStrategy,
    Constraints,
    InventoryInfo,
    OptimizedTicketMatcher,
    OrganizationStrategy,
    PaymentOrder,
    SplitRule,
    SplitStrategy,
    TailDiffType,
    TargetWeights,
    TermStrategy,
    Ticket,
    TicketCategory,
    UserPreference,
    generate_test_data,
)


def example_1_basic_usage():
    """示例1：基础用法 - 大额优先策略"""
    print("=" * 80)
    print("示例1：基础用法 - 大额优先策略")
    print("=" * 80)

    # 生成测试数据
    tickets, inventory_info = generate_test_data(ticket_count=500, seed=42)
    
    # 创建付款单
    payment_order = PaymentOrder(
        id="PO001",
        amount=Decimal("1000000"),  # 100万
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
        max_tickets=8,  # 最多8张票
        min_amount=Decimal("30000"),  # 最小金额3万
        max_amount=Decimal("600000"),  # 最大金额60万
        remain_after_split=Decimal("10000"),  # 拆分后留存至少1万
        max_mini_amount_tickets=2,  # 最多2张小额票
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
    print_solution(solution)


def example_2_small_first():
    """示例2：小额优先策略"""
    print("\n" + "=" * 80)
    print("示例2：小额优先策略")
    print("=" * 80)

    tickets, inventory_info = generate_test_data(ticket_count=500, seed=123)
    
    payment_order = PaymentOrder(
        id="PO002",
        amount=Decimal("500000"),  # 50万
        organization="B"
    )
    
    # 小额优先策略
    target_weights = TargetWeights(
        w1=0.6,
        w2=0.2,
        w3=0.1,
        w4=0.1,
        amount_strategy=AmountStrategy.SMALL_FIRST,
        term_strategy=TermStrategy.NEAR_FIRST,  # 优先近期到期
        acceptor_strategy=AcceptorStrategy.BAD_FIRST,  # 优先差承兑人
        organization_strategy=OrganizationStrategy.DIFF_ORG_FIRST,  # 优先跨组织
    )
    
    split_rule = SplitRule(
        tail_diff_type=TailDiffType.PERCENTAGE,
        tail_diff_value=0.05,  # 尾差占比5%
        split_strategy=SplitStrategy.BY_TERM,
    )
    
    constraints = Constraints(
        max_tickets=10,
        min_amount=Decimal("20000"),
        max_amount=Decimal("500000"),
        remain_after_split=Decimal("5000"),
        max_mini_amount_tickets=3,
    )
    
    user_preference = UserPreference(
        prefer_exact=True,
        allow_split=True,
        allow_inventory_balance=False,
    )
    
    matcher = OptimizedTicketMatcher(
        tickets=tickets,
        payment_order=payment_order,
        target_weights=target_weights,
        constraints=constraints,
        split_rule=split_rule,
        user_preference=user_preference,
        inventory_info=inventory_info,
        random_seed=123,
    )
    
    solution = matcher.optimize()
    print_solution(solution)


def example_3_inventory_balance():
    """示例3：库存平衡优化"""
    print("\n" + "=" * 80)
    print("示例3：库存平衡优化")
    print("=" * 80)

    tickets, inventory_info = generate_test_data(ticket_count=800, seed=456)
    
    payment_order = PaymentOrder(
        id="PO003",
        amount=Decimal("1500000"),  # 150万
        organization="C"
    )
    
    # 库存优化策略
    target_weights = TargetWeights(
        w1=0.4,
        w2=0.2,
        w3=0.2,
        w4=0.2,
        amount_strategy=AmountStrategy.OPTIMIZE_INVENTORY,
        term_strategy=TermStrategy.FAR_FIRST,
        acceptor_strategy=AcceptorStrategy.GOOD_FIRST,
        organization_strategy=OrganizationStrategy.SAME_ORG_FIRST,
    )
    
    split_rule = SplitRule(
        tail_diff_type=TailDiffType.AMOUNT,
        tail_diff_value=100_000,
        split_strategy=SplitStrategy.BY_ACCEPTOR,
    )
    
    constraints = Constraints(
        max_tickets=6,
        min_amount=Decimal("50000"),
        max_amount=Decimal("600000"),
        remain_after_split=Decimal("20000"),
        max_mini_amount_tickets=1,
    )
    
    # 启用库存平衡，设置期望剩余库存占比
    user_preference = UserPreference(
        prefer_exact=False,
        allow_split=True,
        allow_inventory_balance=True,
        remain_dist={
            "大额": 0.4,
            "中额": 0.4,
            "小额": 0.2,
        },
    )
    
    matcher = OptimizedTicketMatcher(
        tickets=tickets,
        payment_order=payment_order,
        target_weights=target_weights,
        constraints=constraints,
        split_rule=split_rule,
        user_preference=user_preference,
        inventory_info=inventory_info,
        random_seed=456,
    )
    
    solution = matcher.optimize()
    print_solution(solution)


def print_solution(solution):
    """打印解决方案详情"""
    result = solution.to_dict()
    
    print(f"\n【配票结果】")
    print(f"选中票据数量: {result['票据张数']}")
    print(f"总金额: {result['总金额']:,.2f} 元")
    print(f"电汇尾差: {result['电汇尾差']:,.2f} 元")
    print(f"拆票金额: {result['拆票金额']:,.2f} 元")
    print(f"执行时间: {result['执行时间']}")
    
    print(f"\n【评分详情】")
    print(f"综合得分: {result['综合得分']:.4f}")
    print(f"  - 金额得分: {result['金额得分']:.4f}")
    print(f"  - 期限得分: {result['期限得分']:.4f}")
    print(f"  - 承兑人得分: {result['承兑人得分']:.4f}")
    print(f"  - 组织得分: {result['组织得分']:.4f}")
    print(f"  - 库存平衡得分: {result['库存平衡得分']:.4f}")
    
    print(f"\n【票据结构】")
    structure = result['选票结构']
    print(f"大票: {structure['大票数量']}张 ({structure['大票占比']:.1%})")
    print(f"中票: {structure['中票数量']}张 ({structure['中票占比']:.1%})")
    print(f"小票: {structure['小票数量']}张 ({structure['小票占比']:.1%})")
    
    print(f"\n【选中票据明细】")
    for i, ticket_detail in enumerate(result['选中票据'][:5], 1):  # 只显示前5张
        print(f"{i}. 票据ID: {ticket_detail['票据ID']}, "
              f"原始金额: {ticket_detail['原始金额']:,.2f}, "
              f"使用金额: {ticket_detail['使用金额']:,.2f}, "
              f"拆分: {'是' if ticket_detail['是否拆分'] else '否'}")
    if len(result['选中票据']) > 5:
        print(f"... 共{len(result['选中票据'])}张票据")
    
    if result['余票库存分布']:
        print(f"\n【余票库存分布】")
        remaining = result['余票库存分布']
        if '数量' in remaining and '占比' in remaining:
            print(f"大票余量: {remaining['数量']['大额']}张 ({remaining['占比']['大额']:.1%})")
            print(f"中票余量: {remaining['数量']['中额']}张 ({remaining['占比']['中额']:.1%})")
            print(f"小票余量: {remaining['数量']['小额']}张 ({remaining['占比']['小额']:.1%})")


if __name__ == "__main__":
    # 运行所有示例
    example_1_basic_usage()
    example_2_small_first()
    example_3_inventory_balance()
    
    print("\n" + "=" * 80)
    print("所有示例执行完毕！")
    print("=" * 80)
