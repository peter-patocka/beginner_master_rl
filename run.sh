#!/usr/bin/env bash

case $1 in
  'start') # start jupyter notebook
    conda env create --file environment.yml
    conda activate bmrl
    jupyter notebook
    ;;
  'execute_random')
    python examples/maze_random.py
    ;;
  'execute_iteration')
    python examples/policy_iteration.py
    ;;
  'execute_montecarlo')
    python examples/montecarlo_on_policy_control_optimized.py
    ;;
  'execute_sarsa')
    python examples/sarsa_n_step.py
    ;;
  *)
    echo "Invalid command"
    ;;
esac
