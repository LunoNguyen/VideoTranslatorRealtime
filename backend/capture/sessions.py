"""Enumerate audio-playing apps and track per-app capture permission."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import psutil


@dataclass
class AppSession:
    exe: str  # lower-case executable name
    pid: int
    display_name: str
    active: bool


def resolve_root_pids(exe: str, procs: Iterable[psutil.Process] | None = None) -> list[int]:
    """PIDs of `exe` processes whose parent is not the same exe."""
    exe = exe.lower()
    plist = list(psutil.process_iter() if procs is None else procs)
    names: dict[int, str] = {}
    for p in plist:
        try:
            names[p.pid] = p.name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def parent_name(ppid: int) -> str | None:
        if ppid in names:
            return names[ppid]
        try:
            return psutil.Process(ppid).name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    roots: list[int] = []
    for p in plist:
        if names.get(p.pid) != exe:
            continue
        try:
            parent = parent_name(p.ppid())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent = None
        if parent != exe:
            roots.append(p.pid)
    return sorted(roots)


def list_sessions() -> list[AppSession]:
    """List audio sessions, one per (exe, process-tree root) (Windows only, uses pycaw)."""
    from pycaw.constants import AudioSessionState
    from pycaw.pycaw import AudioUtilities

    raw: list[tuple[int, str, str, bool]] = []
    for s in AudioUtilities.GetAllSessions():
        proc = s.Process
        if proc is None:
            continue
        try:
            exe = proc.name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        raw.append((proc.pid, exe, (s.DisplayName or "").strip(), s.State == AudioSessionState.Active))
    return group_sessions(raw, live_proc_table() if raw else {})


def same_exe_root(pid: int, exe: str, table: "ProcTable") -> int:
    """Walk up from `pid` while the parent has the same exe name."""
    cur, seen = pid, {pid}
    while cur in table:
        ppid = table[cur][1]
        if ppid in seen or ppid not in table or table[ppid][0] != exe:
            break
        cur = ppid
        seen.add(cur)
    return cur


def group_sessions(raw: Iterable[tuple[int, str, str, bool]], table: "ProcTable") -> list[AppSession]:
    """Group raw (pid, exe, display name, active) sessions by (exe, own tree root)."""
    groups: dict[tuple[str, int], AppSession] = {}
    for pid, exe, display, active in raw:
        key = (exe, same_exe_root(pid, exe, table))
        if key in groups:
            groups[key].active = groups[key].active or active
            groups[key].display_name = groups[key].display_name or display
            continue
        groups[key] = AppSession(exe, key[1], display, active)
    for g in groups.values():
        g.display_name = g.display_name or g.exe.removesuffix(".exe")
    return sorted(groups.values(), key=lambda g: (g.exe, g.pid))


ProcTable = dict[int, tuple[str, int]]  # pid -> (lower-case name, parent pid)


def live_proc_table() -> ProcTable:
    table: ProcTable = {}
    for p in psutil.process_iter(["name", "ppid"]):
        table[p.pid] = ((p.info["name"] or "").lower(), p.info["ppid"] or 0)
    return table


def is_in_tree(pid: int, root: int, table: ProcTable) -> bool:
    """True if `pid` is `root` or one of its descendants."""
    seen: set[int] = set()
    while pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        if pid not in table:
            return False
        pid = table[pid][1]
    return False


class PermissionTracker:
    """Tracks which root PIDs should be captured given the allow-list."""

    def __init__(
        self, allowed: set[str], own_pid: int, proc_table: Callable[[], ProcTable] | None = None
    ) -> None:
        self._proc_table = proc_table or live_proc_table
        self._allowed = {e.lower() for e in allowed}
        self._own_pid = own_pid
        self._running: set[int] = set()
        self.self_tree_exes: set[str] = set()  # allowed exes skipped because they are our ancestors

    @property
    def allowed(self) -> set[str]:
        return set(self._allowed)

    def set_allowed(self, exe: str, allowed: bool) -> None:
        (self._allowed.add if allowed else self._allowed.discard)(exe.lower())

    def forget(self, pid: int) -> None:
        """Drop a pid from the running set so the next update() reports it to start again."""
        self._running.discard(pid)

    def update(
        self, sessions: list[AppSession], table: ProcTable | None = None
    ) -> tuple[set[int], set[int]]:
        """Return (pids to start, pids to stop) relative to the previous call.

        `table` is an optional process snapshot (pid -> (name, ppid)); if omitted the
        constructor's provider is called.
        """
        cand = {s.pid: s.exe.lower() for s in sessions if s.exe.lower() in self._allowed}
        if table is None:
            table = self._proc_table() if any(p != self._own_pid for p in cand) else {}
        # never capture ourselves: own pid, any process below it, or any ancestor (its tree holds us)
        ancestors = {p for p in cand if p != self._own_pid and is_in_tree(self._own_pid, p, table)}
        self.self_tree_exes = {cand[p] for p in ancestors}
        wanted = {p for p in cand if p not in ancestors and not is_in_tree(p, self._own_pid, table)}
        start, stop = wanted - self._running, self._running - wanted
        self._running = wanted
        return start, stop
