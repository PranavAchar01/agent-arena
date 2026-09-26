#!/bin/zsh
# Three fully automatic runs, one per box: bowl (people 1,2), stack (people 5,6), tower (people 3,4).
cd "$(dirname $0)/.."
for spec in "place|subject_1,subject_2" "stack|subject_5,subject_6" "tower|subject_3,subject_4"; do
  t=${spec%%|*}; s=${spec#*|}
  nice -n 19 taskpolicy -b .venv/bin/python scripts/run_hocap.py $t $s 2 24 box-$t > runs/box-$t.log 2>&1
  echo "$t exit $? $(date +%T)"
done
