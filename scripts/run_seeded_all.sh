#!/bin/zsh
cd "$(dirname $0)/.."
for t in stack place push; do
  echo "=== $t $(date +%T)"; nice -n 19 taskpolicy -b .venv/bin/python scripts/run_seeded.py $t > runs/$t-seeded.log 2>&1; echo "exit $? $(date +%T)"
done
