"""Alternative connected owners and complete-dependency budget planning.

This package does not change the prediction model or the frozen training code.
"""
from .planner import Budget, EmpiricalMemoryCost, PartitionBudgetError, plan_partition, plan_for_model

__all__ = ['Budget', 'EmpiricalMemoryCost', 'PartitionBudgetError', 'plan_partition', 'plan_for_model']
