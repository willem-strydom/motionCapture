import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def optimize_mab_simplex(p_values, w_original, alpha, sum_constraint=2.0):
    """
    Optimize win probabilities for a MAB problem using simplex method.
    
    Args:
        p_values: Array of probabilities that player plays each machine
        w_original: Original win probabilities for each machine
        alpha: Maximum adjustment allowed for each win probability
        sum_constraint: Sum of win probabilities (default: 2.0)
    
    Returns:
        Optimized win probabilities and expected return
    """
    n = len(p_values)
    if not (np.sum(p_values)==1):
        p_values = np.exp(p_values)/(np.sum(np.exp(p_values)))
    
    # For linprog, we need to minimize c^T @ x
    # Since we want to maximize sum(p_i * w_i), we minimize -sum(p_i * w_i)
    c = p_values  # Negative because linprog minimizes
    
    # Bounds for each win probability
    # Each w must be between max(0, w_original - alpha) and min(1, w_original + alpha)
    bounds = [(max(0, w_original[i] - alpha), min(1, w_original[i] + alpha)) for i in range(n)]
    
    # Constraint: sum of w = sum_constraint
    # This is expressed as A_eq @ x = b_eq
    A_eq = np.ones((1, n))  # Coefficient matrix for equality constraint
    b_eq = np.array([sum_constraint])  # Right-hand side for equality constraint
    
    # Solve the linear programming problem
    result = linprog(
        c,          # Coefficients of the objective function
        A_eq=A_eq,  # Coefficient matrix for equality constraints
        b_eq=b_eq,  # Right-hand side for equality constraints
        bounds=bounds,
        method='simplex'  # Use the simplex method
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    # Calculate the expected return
    expected_return = np.sum(p_values * result.x)
    
    return result.x, expected_return

def example_usage():
    # Example inputs
    p_values = np.array([-0.5, 0.5, -0.9, 0.1])  # Player's probability of playing each machine
    w_original = np.array([0.9, 0.1, 0.7, 0.3])  # Original win probabilities
    alpha = 0.2  # Maximum adjustment allowed
    
    # Run optimization
    w_optimized, expected_return = optimize_mab_simplex(p_values, w_original, alpha)
    
    print(f"Original win probabilities: {w_original}")
    print(f"Optimized win probabilities: {w_optimized}")
    print(f"Sum of optimized win probabilities: {np.sum(w_optimized):.6f}")
    print(f"Expected return: {expected_return:.6f}")
    
    # For comparison, calculate expected return with original probabilities
    original_return = np.sum(p_values * w_original)
    print(f"Original expected return: {original_return:.6f}")
    print(f"Improvement: {(expected_return - original_return) / original_return * 100:.2f}%")
    
    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bandits = [f"Bandit {i+1}" for i in range(len(p_values))]
    x = np.arange(len(bandits))
    width = 0.35
    
    ax.bar(x - width/2, w_original, width, label='Original')
    ax.bar(x + width/2, w_optimized, width, label='Optimized')
    
    # Add text annotations for play probabilities
    for i, p in enumerate(p_values):
        ax.annotate(f"p={p}", xy=(i, 0.05), ha='center')
    
    ax.set_ylabel('Win Probability')
    ax.set_title('Original vs Optimized Win Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(bandits)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_usage()