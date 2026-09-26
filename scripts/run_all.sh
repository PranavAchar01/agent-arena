#!/bin/zsh
# Runs the three demo tasks in order (one heavy job at a time), low priority.
cd "$(dirname $0)/.."
for spec in "push|I want to train a robot to push a block onto a target" "place|I want to train a robot to put a block in a bowl" "stack|I want to train a robot to stack a block on top of another block"; do
  name=${spec%%|*}; text=${spec#*|}
  echo "=== $name $(date +%T)"
  nice -n 19 taskpolicy -b .venv/bin/python scripts/run_task.py "$name" "$text" > "runs/$name.log" 2>&1
  echo "exit $? $(date +%T)"
done
