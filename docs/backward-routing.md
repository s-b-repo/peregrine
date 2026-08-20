# Backward Routing

## Problem

Normally inference asks: Given hidden state x, which experts should run?

## Inverted question

Given the desired final residual, which expert outputs could possibly matter?

## Method

Approximate the Jacobian: J_e = ∂R/∂E_e and determine sensitivity.

An expert with low sensitivity could be deprioritized even if its router
probability is significant.

## Key insight

Two different concepts:
  router importance ≠ output sensitivity

Peregrine could potentially exploit both.

## Implementation

1. Computes Jacobian-vector products: J_e · δ for each expert (cheap, no expert load)
2. Estimates sensitivity: |∂R/∂E_e| for each expert
3. Deprioritizes experts with low sensitivity even if router probability is high
4. Combines router probability × sensitivity for final expert selection
