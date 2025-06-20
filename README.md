# Optimal Strategy Solver for the "Pick-A-Number" Game

**A computational game theory approach to finding the minimax optimal strategy for a number-picking game using linear programming.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [The Game](#the-game)
- [Technical Deep Dive](#technical-deep-dive)
- [How to Run](#how-to-run)
- [Key Takeaways & Challenges](#key-takeaways--challenges)
- [Dependencies](#dependencies)

## Project Overview

This project explores the classic "Pick-A-Number" game from a computational game theory perspective. Instead of using simple heuristics (like binary search), this script formulates the game as a zero-sum game and solves for the guesser's optimal minimax strategy. The solution ensures the highest possible guaranteed payoff, regardless of which number the "hider" chooses.

## The Game

The Pick A Number game comes from the first example from the book 'The Art of Strategy' by Dixit and Nalebuff. The game is as follows.

- **Objective:** A "guesser" has `T` turns to guess a secret number chosen by a "hider" from a range of `0` to `N-1`.
- **Feedback:** After each incorrect guess, the hider reveals if the true number is "higher" or "lower".
- **Scoring:** The guesser receives a payoff `V[t]` if they guess correctly on turn `t`, with earlier guesses yielding higher scores.

## Technical Deep Dive

The core of this project is the `solve_using_LP` function, which models the game as a **Linear Program (LP)**.

1.  **Information Sets:** The state of the game is captured in "Information Sets" (`Infosets`), which represent what the guesser knows at any point (the current turn and the possible range of the secret number).
2.  **Sequence-Form Formulation:** The entire game tree is translated into an LP. Each possible sequence of moves becomes a variable representing its probability.
3.  **Minimax via Duality:** The LP is designed to find a minimax equilibrium. It maximizes a variable `L`, which represents the guesser's minimum guaranteed payoff, subject to:
    - **Flow Constraints:** Ensuring the probabilities are valid and sum to 1.
    - **Optimality Constraints:** Forcing `L` to be less than or equal to the expected payoff for every possible secret number the hider could choose.

The solution to this LP is a complete, optimal strategy dictating the exact guess (or mix of guesses) to make in any situation to maximize the worst-case outcome.

## How to Run

1.  **Clone the repository:**
    *(Remember to use your new repository name here)*
    ```bash
    git clone [https://github.com/ziliangstanley/Pick-A-Number.git](https://github.com/ziliangstanley/Pick-A-Number.git)
    cd Pick-A-Number
    ```
2.  **Install dependencies:** Make sure you have the required Python libraries installed.
    ```bash
    pip install numpy cvxpy gurobipy
    ```
    *(Note: Gurobi is a commercial solver but provides a free, restricted license for academic and non-production use, which is automatically installed with the pip package).*

3.  **Run the script:**
    ```bash
    python pick-a-number.py
    ```
    *(You may also want to rename your `.py` file to `pick-a-number.py` for consistency).*

## Key Takeaways & Challenges

- **Optimal vs. Naive Strategy:** The script compares the optimal strategy's payoff against simpler, naive strategies (like deterministic binary search), demonstrating a clear performance improvement.
- **Scalability Challenge:** The sequence-form LP formulation is powerful but suffers from poor scalability. The number of variables grows polynomially with the size of the game (`N`) and the number of turns (`T`). This leads to significant memory consumption, highlighting a common trade-off in computational modeling between theoretical exactness and practical feasibility.

## Dependencies

- Python 3.x
- [CVXPY](https://www.cvxpy.org/)
- [Gurobi](https://www.gurobi.com/)
- NumPy
