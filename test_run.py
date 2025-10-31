#!/usr/bin/env python
"""快速测试脚本"""

from ticket_matching import run_example

if __name__ == "__main__":
    print("=" * 80)
    print("运行票据配票算法测试")
    print("=" * 80)
    print()
    
    result = run_example()
    
    print()
    print("=" * 80)
    print("【配票结果详情】")
    print("=" * 80)
    
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"\n{key}: (显示前3项)")
            for i, item in enumerate(value[:3], 1):
                print(f"  {i}. {item}")
            if len(value) > 3:
                print(f"  ... 共 {len(value)} 项")
        else:
            print(f"{key}: {value}")
