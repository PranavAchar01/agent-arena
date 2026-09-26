"""Sandbox runner tests that need no Vultr account.

The Vultr lifecycle runs against a fake API (httpx.MockTransport). The SSH path of DockerRunner runs against local
Docker with `sh -c` standing in for the ssh prefix, which exercises the same remote mkdir, docker run and tar fetch.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile

import httpx
import pytest

from arena.sandbox import runner as R


def fake_vultr(fail_on: str | None = None):
    calls: list[tuple[str, str]] = []

    def handle(req: httpx.Request) -> httpx.Response:
        path = req.url.path.removeprefix("/v2")
        calls.append((req.method, path))
        if fail_on and path.startswith(fail_on):
            return httpx.Response(500, json={"error": "boom"})
        if req.method == "DELETE":
            return httpx.Response(204)
        if path == "/ssh-keys":
            return httpx.Response(201, json={"ssh_key": {"id": "key-1"}})
        if path == "/firewalls":
            return httpx.Response(201, json={"firewall_group": {"id": "fw-1"}})
        if path.endswith("/rules"):
            return httpx.Response(201, json={"firewall_rule": {"id": 1}})
        if path == "/os":
            return httpx.Response(
                200,
                json={
                    "os": [
                        {
                            "id": 1743,
                            "name": "Ubuntu 22.04 LTS x64",
                            "family": "ubuntu",
                            "arch": "x64",
                        },
                        {
                            "id": 2284,
                            "name": "Ubuntu 24.04 LTS x64",
                            "family": "ubuntu",
                            "arch": "x64",
                        },
                    ]
                },
            )
        if path == "/plans":
            return httpx.Response(
                200, json={"plans": [{"id": "vc2-1c-2gb", "monthly_cost": 10}]}
            )
        if path == "/instances" and req.method == "POST":
            body = json.loads(req.content)
            assert body["firewall_group_id"] == "fw-1" and body["sshkey_id"] == [
                "key-1"
            ]
            assert body["os_id"] == 2284 and "arena-sbx" in body["tags"]
            return httpx.Response(
                202, json={"instance": {"id": "inst-123456789", "status": "pending"}}
            )
        if path.startswith("/instances/"):
            return httpx.Response(
                200, json={"instance": {"status": "pending", "main_ip": "0.0.0.0"}}
            )
        return httpx.Response(404)

    return calls, httpx.MockTransport(handle)


@pytest.fixture
def vultr(monkeypatch):
    monkeypatch.setenv("VULTR_API_KEY", "test-not-a-key")
    monkeypatch.setenv("VULTR_ALLOW_CIDR", "203.0.113.7/32")
    monkeypatch.setenv("SANDBOX_BACKEND", "vultr")

    def make(fail_on=None):
        calls, transport = fake_vultr(fail_on)
        v = R.VultrRunner()
        v.api = httpx.Client(base_url=R.VULTR_API, transport=transport)
        return v, calls

    return make


def test_killed_while_booting_deletes_vm_firewall_and_key(vultr, monkeypatch):
    v, calls = vultr()
    events = []
    with pytest.raises(R.BoxKilled), R.box("t-kill") as b:
        b.runner = v

        def boot_then_kill():
            b.killed = True
            R.check_killed()

        monkeypatch.setattr(v, "_wait_active", boot_then_kill)
        v.run({}, R.Path("/tmp/unused"), events.append)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "vm_create" and "vm_destroyed" in kinds
    assert ("DELETE", "/instances/inst-123456789") in calls
    assert ("DELETE", "/firewalls/fw-1") in calls
    assert ("DELETE", "/ssh-keys/key-1") in calls
    create = next(e for e in events if e["type"] == "vm_create")
    assert create["firewall"] == "inbound 22/tcp from 203.0.113.7/32 only"
    assert create["hourly_usd"] == round(10 / 672, 4)


def test_api_failure_after_create_still_cleans_up(vultr):
    v, calls = vultr(fail_on="/instances/inst")  # the status poll fails with a 500
    with pytest.raises(RuntimeError), R.box("t-fail") as b:
        b.runner = v
        v.run({}, R.Path("/tmp/unused"), lambda e: None)
    assert (
        calls.count(("DELETE", "/instances/inst-123456789")) == 1
    )  # DELETE also 500s here, and is tried once
    assert ("DELETE", "/ssh-keys/key-1") in calls


def test_close_is_idempotent(vultr):
    v, calls = vultr()
    v.close()
    v.close()
    assert not [
        c for c in calls if c[0] == "DELETE"
    ]  # nothing was created, nothing to delete


def test_fetch_tar_rejects_escapes_and_links(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (("manifest.json", b"{}"), ("../escape.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("clips/evil.mp4")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    src = tmp_path / "out.tar"
    src.write_bytes(buf.getvalue())
    dest = tmp_path / "dest"
    dest.mkdir()
    R._fetch_tar(["cat", str(src)], dest)
    assert (dest / "manifest.json").read_text() == "{}"
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "clips" / "evil.mp4").exists()


@pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(
        ["docker", "image", "inspect", R.IMAGE], capture_output=True, check=False
    ).returncode,
    reason="needs local Docker and the sandbox image",
)
def test_remote_path_runs_the_same_container(tmp_path):
    d = R.DockerRunner(ssh=["sh", "-c"], host="stand-in")
    events = []
    job = {
        "queries": [],
        "fetch": [],
        "max_clips": 0,
        "time_budget_s": 30,
        "sandbox_seconds": 60,
    }
    manifest = d.run(job, tmp_path, events.append)
    start = next(e for e in events if e["type"] == "sandbox_start")
    end = next(e for e in events if e["type"] == "sandbox_destroyed")
    assert start["backend"] == "vultr" and start["host"] == "stand-in"
    assert end["exit"] == 0, end
    assert manifest == {"clips": []}
    assert not R.Path(
        f"/tmp/{start['container']}"
    ).exists()  # the remote scratch dir is removed after the fetch
