import sys
import traceback
sys.stderr = sys.stdout

print("Test 1: torch")
try:
    import torch
    print(f"  OK: {torch.__version__}")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 2: transformers")
try:
    import transformers
    print(f"  OK: {transformers.__version__}")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 3: tokenizers")
try:
    import tokenizers
    print(f"  OK: {tokenizers.__version__}")
except Exception as e:
    print(f"  FAIL: {e}")

print("Test 4: sentence_transformers (just the package)")
try:
    import sentence_transformers
    print(f"  OK: {sentence_transformers.__version__}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

print("Done")
