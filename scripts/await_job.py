#!/usr/bin/env python3
"""Wait on a cluster job, and fail fast when it is queued rather than running.

`kubectl wait --for=condition=complete` cannot distinguish a job that is doing
work from one the scheduler has never placed. A pod that no node can satisfy
stays Pending until the wait expires, so a scheduling problem is indistinguishable
from a slow run for as long as the timeout allows - which converts a fixable
misconfiguration into hours of silence.

This separates the two. A pod that has not been assigned a node within the
scheduling guard is reported with the scheduler's own reason, the pod's resource
requests, and an explicit verdict, then exits non-zero. Only a pod that has
actually started is given the long run timeout.

Exit codes are the interface:

    0  the job completed
    2  the job never scheduled within the guard - the reason is printed
    3  the job ran and failed - the container's last output is printed
    4  the wait itself could not proceed, for example the job does not exist
    5  --once only: the job is still queued or still running

`--once` reports the current state and returns immediately. It exists because a
saturated cluster makes queueing the dominant cost, and a job that is meant to be
left alone until an accelerator frees should be checked rather than waited on.

Nothing here mutates cluster state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

SCHEDULE_GUARD_SECONDS = 420
RUN_TIMEOUT_SECONDS = 5400
POLL_SECONDS = 15
HEARTBEAT_SECONDS = 60


def _kubectl(namespace: str, *args: str) -> dict | None:
    result = subprocess.run(
        ["kubectl", "-n", namespace, *args, "-o", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _pods(namespace: str, job: str) -> list[dict]:
    payload = _kubectl(namespace, "get", "pods", "-l", f"job-name={job}")
    return payload.get("items", []) if payload else []


def _scheduling_reason(pod: dict) -> str:
    """The scheduler's own message, from the pod condition rather than the event.

    The event object truncates the per-node reason list; the PodScheduled
    condition carries the same message and is read first because it is more often
    complete.
    """
    for condition in pod.get("status", {}).get("conditions", []):
        if condition.get("type") == "PodScheduled" and condition.get("status") != "True":
            return condition.get("message") or condition.get("reason") or "unknown"
    return "no PodScheduled condition recorded"


def _report_unscheduled(namespace: str, job: str, pods: list[dict], waited: int) -> None:
    print(f"\nNOT SCHEDULED: {job} has been Pending for {waited}s without a node.")
    print("This will not resolve by waiting longer at the current request shape.\n")
    for pod in pods:
        meta = pod["metadata"]["name"]
        spec = pod["spec"]["containers"][0].get("resources", {}).get("requests", {})
        print(f"pod {meta}")
        print(f"  requests: {json.dumps(spec)}")
        selectors = (
            pod["spec"]
            .get("affinity", {})
            .get("nodeAffinity", {})
            .get("requiredDuringSchedulingIgnoredDuringExecution", {})
            .get("nodeSelectorTerms", [])
        )
        for term in selectors:
            for expression in term.get("matchExpressions", []):
                print(f"  requires {expression.get('key')} in {expression.get('values')}")
        print(f"  scheduler: {_scheduling_reason(pod)}\n")

    events = _kubectl(namespace, "get", "events", "--field-selector", "reason=FailedScheduling")
    if events:
        messages = [
            item.get("message", "")
            for item in events.get("items", [])
            if job in item.get("involvedObject", {}).get("name", "")
        ]
        if messages:
            print("most recent scheduler event:")
            print(f"  {messages[-1]}\n")


def _report_failure(namespace: str, job: str, pods: list[dict]) -> None:
    print(f"\nFAILED: {job} ran and did not complete.\n")
    for pod in pods:
        name = pod["metadata"]["name"]
        for status in pod.get("status", {}).get("containerStatuses", []) or []:
            terminated = (status.get("state") or {}).get("terminated") or {}
            if terminated:
                print(
                    f"pod {name}: exit {terminated.get('exitCode')} "
                    f"({terminated.get('reason')})"
                )
        logs = subprocess.run(
            ["kubectl", "-n", namespace, "logs", name, "--tail", "60"],
            capture_output=True,
            text=True,
        )
        if logs.stdout.strip():
            print(logs.stdout)


def await_job(
    namespace: str,
    job: str,
    schedule_guard: int,
    run_timeout: int,
) -> int:
    started = time.monotonic()
    scheduled_at: float | None = None
    last_heartbeat = 0.0

    while True:
        payload = _kubectl(namespace, "get", "job", job)
        if payload is None:
            print(f"cannot read job {job} in namespace {namespace}", file=sys.stderr)
            return 4

        status = payload.get("status", {})
        if status.get("succeeded"):
            elapsed = int(time.monotonic() - started)
            print(f"\nCOMPLETE: {job} finished after {elapsed}s.")
            return 0
        if status.get("failed"):
            _report_failure(namespace, job, _pods(namespace, job))
            return 3

        pods = _pods(namespace, job)
        placed = [pod for pod in pods if pod.get("spec", {}).get("nodeName")]
        now = time.monotonic()

        if placed and scheduled_at is None:
            scheduled_at = now
            node = placed[0]["spec"]["nodeName"]
            print(f"scheduled on {node} after {int(now - started)}s; running")

        if scheduled_at is None:
            if now - started > schedule_guard:
                _report_unscheduled(namespace, job, pods, int(now - started))
                return 2
        elif now - scheduled_at > run_timeout:
            print(f"\nTIMEOUT: {job} has been running for more than {run_timeout}s.")
            _report_failure(namespace, job, pods)
            return 3

        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            phase = "running" if scheduled_at else "waiting for a node"
            print(f"[{int(now - started):5d}s] {job}: {phase}", flush=True)
            last_heartbeat = now

        time.sleep(POLL_SECONDS)


def check_once(namespace: str, job: str) -> int:
    """Report the job's current state and return, without waiting."""
    payload = _kubectl(namespace, "get", "job", job)
    if payload is None:
        print(f"cannot read job {job} in namespace {namespace}", file=sys.stderr)
        return 4

    status = payload.get("status", {})
    pods = _pods(namespace, job)
    if status.get("succeeded"):
        print(f"COMPLETE: {job}")
        return 0
    if status.get("failed"):
        _report_failure(namespace, job, pods)
        return 3

    placed = [pod for pod in pods if pod.get("spec", {}).get("nodeName")]
    if placed:
        print(f"RUNNING: {job} on {placed[0]['spec']['nodeName']}")
    else:
        print(f"QUEUED: {job} has no node yet")
        for pod in pods:
            print(f"  {_scheduling_reason(pod)}")
    return 5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--schedule-guard", type=int, default=SCHEDULE_GUARD_SECONDS)
    parser.add_argument("--run-timeout", type=int, default=RUN_TIMEOUT_SECONDS)
    parser.add_argument(
        "--once",
        action="store_true",
        help="report the current state and exit rather than waiting",
    )
    args = parser.parse_args(argv)
    if args.once:
        return check_once(args.namespace, args.job)
    return await_job(args.namespace, args.job, args.schedule_guard, args.run_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
