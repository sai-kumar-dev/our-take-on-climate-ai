# Target Redesign Roadmap

## Current State

The current labels are district-time crop-share distributions derived from area or production history.

## Gold-Standard Direction

Move toward targets built from one or more of:
- realized yield
- net farmer return
- crop failure or loss probability
- agronomist-reviewed suitability labels
- resilience under stress scenarios

## Recommended Modeling Split

1. Prior model
   Learns historical adoption patterns and crop availability by district and season.
2. Outcome model
   Learns yield, failure, or resilience outcomes from agronomic evidence.
3. Ranker
   Combines prior, outcome, and policy constraints into the final advisory shortlist.

## Why This Matters

Without this redesign, a historically common crop can stay highly ranked even when we would rather optimize for resilience or field-level success.
