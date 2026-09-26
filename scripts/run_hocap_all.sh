#!/bin/zsh
cd "$(dirname $0)/.."
nice -n 19 taskpolicy -b .venv/bin/python scripts/run_hocap.py place subject_1,subject_2 2 > runs/hocap-place.log 2>&1; echo "place exit $? $(date +%T)"
