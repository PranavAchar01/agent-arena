#!/bin/zsh
# A -> B -> C block chain, fully automatic (agent plans, agent picks, VLM verifies). Low priority, one at a time.
cd "$(dirname $0)/.."
for spec in "unjar|I want to train a robot to take a block out of a jar" "stack|I want to train a robot to stack a block on another block" "tower|I want to train a robot to stack one more block to make a tower"; do
  name=${spec%%|*}; text=${spec#*|}
  echo "=== $name $(date +%T)"
  nice -n 19 taskpolicy -b .venv/bin/python scripts/run_task.py "blocks-$name" "$text" "$name" > "runs/blocks-$name.log" 2>&1
  echo "exit $? $(date +%T)"
done
