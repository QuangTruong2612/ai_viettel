"""Verify inference state isolation."""
import sys, inspect
sys.path.insert(0, r'f:\AI_VIETTEL')

# Inspect process_record to verify reset+log wired
from src import inference
src = inspect.getsource(inference.process_record)
print('=== process_record signature ===')
print(src[:1200])
print()
print('=== Reset + token log functions ===')
print('- _reset_per_record_state:', inspect.getsource(inference._reset_per_record_state))
print()
print('- _log_token_budget:', inspect.getsource(inference._log_token_budget)[:600])