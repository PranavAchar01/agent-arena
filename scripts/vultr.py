"""Vultr helpers for the event. Needs VULTR_API_KEY in the environment (never in a file).

scripts/vultr.py check   account, region and plan availability, and (with VULTR_INFERENCE_KEY) the inference models
scripts/vultr.py smoke   one throwaway VM end to end: create, boot, build the sandbox, run an empty job, delete
scripts/vultr.py sweep   delete arena-sbx VMs older than 90 min, plus unused firewall groups and keys (after a crash)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.sandbox import runner as R


def api() -> httpx.Client:
    return httpx.Client(
        base_url=R.VULTR_API,
        headers={"Authorization": f"Bearer {os.environ['VULTR_API_KEY']}"},
        timeout=30,
    )


def check() -> None:
    c = api()
    acct = c.get("/account").raise_for_status().json()["account"]
    print(
        f"account ok · balance {acct.get('balance')} · pending {acct.get('pending_charges')}"
    )
    region, plan = (
        os.environ.get("VULTR_REGION", "sjc"),
        os.environ.get("VULTR_PLAN", "vc2-1c-2gb"),
    )
    avail = (
        c.get(f"/regions/{region}/availability")
        .raise_for_status()
        .json()
        .get("available_plans", [])
    )
    print(
        f"region {region}: plan {plan} {'available' if plan in avail else 'NOT available'} ({len(avail)} plans)"
    )
    key = os.environ.get("VULTR_INFERENCE_KEY")
    if key:
        base = os.environ.get(
            "VULTR_INFERENCE_URL", "https://api.vultrinference.com/v1"
        )
        models = httpx.get(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30
        )
        models.raise_for_status()
        for m in models.json().get("data", []):
            print("inference model:", m.get("id"))
    else:
        print("VULTR_INFERENCE_KEY not set: skipping the inference model list")


def smoke() -> None:
    os.environ["SANDBOX_BACKEND"] = "vultr"
    out = Path(tempfile.mkdtemp(prefix="vultr-smoke-"))
    with R.box("smoke") as b:
        m = b.runner.run(
            {"queries": [], "fetch": [], "max_clips": 0, "time_budget_s": 30},
            out,
            print,
        )
    print("manifest:", m)


def sweep() -> None:
    """Only VMs older than SWEEP_MINUTES (default 90): live boxes and warm-pool VMs are always younger."""
    import datetime as dt

    c = api()
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(
        minutes=int(os.environ.get("SWEEP_MINUTES", "90"))
    )
    for i in (
        c.get("/instances", params={"tag": "arena-sbx", "per_page": 500})
        .raise_for_status()
        .json()["instances"]
    ):
        if dt.datetime.fromisoformat(i["date_created"]) > cutoff:
            continue
        c.delete(f"/instances/{i['id']}").raise_for_status()
        print("deleted instance", i["id"], i["label"])
    for f in (
        c.get("/firewalls", params={"per_page": 500})
        .raise_for_status()
        .json()["firewall_groups"]
    ):
        if f["description"].startswith("arena-sbx-") and not f.get("instance_count"):
            c.delete(f"/firewalls/{f['id']}")
            print("deleted firewall group", f["id"])
    for k in (
        c.get("/ssh-keys", params={"per_page": 500})
        .raise_for_status()
        .json()["ssh_keys"]
    ):
        if k["name"].startswith("arena-sbx-"):
            c.delete(f"/ssh-keys/{k['id']}")
            print("deleted ssh key", k["id"])


if __name__ == "__main__":
    {"check": check, "smoke": smoke, "sweep": sweep}[
        sys.argv[1] if len(sys.argv) > 1 else "check"
    ]()
