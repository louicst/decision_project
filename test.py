import numpy as np

# 1. Build a mini 3D Reward Matrix
# Dimensions: 2 States, 2 Actions, 3 Criteria
# Shape: (2, 2, 3)
R = np.array([
    # STATE 0
    [
        [10,  50,   5],  # Action 0 -> [Profit, Carbon, Biodiversity]
        [20,  10,   0]   # Action 1 -> [Profit, Carbon, Biodiversity]
    ],
    # STATE 1
    [
        [100, 80,  20],  # Action 0 -> [Profit, Carbon, Biodiversity]
        [150,  0, -10]   # Action 1 -> [Profit, Carbon, Biodiversity]
    ]
])

print("--- Original Matrix R ---")
print(f"Shape: {R.shape} -> (States, Actions, Criteria)")
print("\n")

# ---------------------------------------------------------
# Test 1: The one you need for your project
# ---------------------------------------------------------
print("--- Test 1: R.min(axis=(0, 1)) ---")
print("Instruction: Squash States (axis 0) and Actions (axis 1). Keep Criteria separate.")
print("Goal: Find the worst possible Profit, worst Carbon, and worst Bio across the whole game.")
print("Result:")
print(R.min(axis=(0, 1))) 
# Expected: [10, 0, -10]
print("\n")

# ---------------------------------------------------------
# Test 2: Squashing only one axis
# ---------------------------------------------------------
print("--- Test 2: R.min(axis=0) ---")
print("Instruction: Squash ONLY States (axis 0). Keep Actions and Criteria separate.")
print("Goal: If I take Action 0, what is the worst outcome regardless of the state I am in?")
print(f"Result Shape: {R.min(axis=0).shape}")
print("Result:")
print(R.min(axis=0))
print("\n")

# ---------------------------------------------------------
# Test 3: Squashing the criteria
# ---------------------------------------------------------
print("--- Test 3: R.min(axis=2) ---")
print("Instruction: Squash ONLY Criteria (axis 2). Keep States and Actions separate.")
print("Goal: For every state-action combo, what is the single lowest number inside its reward vector?")
print(f"Result Shape: {R.min(axis=2).shape}")
print("Result:")
print(R.min(axis=2))