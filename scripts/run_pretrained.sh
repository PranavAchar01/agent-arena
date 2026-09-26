#!/bin/zsh
cd "$(dirname $0)/.."
for m in scratch frozen finetune; do
  it=$([ $m = finetune ] && echo 1500 || echo 3000)
  nice -n 19 taskpolicy -b .venv/bin/python scripts/pretrained_policy.py $m $it > runs/pretrained-$m.log 2>&1
  echo "$m exit $? $(date +%T)"; grep RESULT runs/pretrained-$m.log | cut -c1-400
done
