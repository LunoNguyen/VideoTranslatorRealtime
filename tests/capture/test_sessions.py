from types import SimpleNamespace

from backend.capture.sessions import AppSession, PermissionTracker, group_sessions, resolve_root_pids


def proc(pid, name, ppid):
    return SimpleNamespace(pid=pid, name=lambda: name, ppid=lambda: ppid)


def sess(exe, pid, active=True):
    return AppSession(exe=exe, pid=pid, display_name=exe, active=active)


def test_resolve_root_pids_picks_top_level_parent():
    procs = [
        proc(1, "explorer.exe", 0),
        proc(100, "chrome.exe", 1),
        proc(101, "chrome.exe", 100),
        proc(102, "Chrome.exe", 100),
    ]
    assert resolve_root_pids("chrome.exe", procs) == [100]


def test_unallowed_app_not_started():
    t = PermissionTracker(set(), own_pid=1)
    assert t.update([sess("vlc.exe", 10)]) == (set(), set())


def test_reattach_when_allowed_app_restarts():
    t = PermissionTracker({"vlc.exe"}, own_pid=1)
    assert t.update([sess("vlc.exe", 10)]) == ({10}, set())
    assert t.update([]) == (set(), {10})
    assert t.update([sess("vlc.exe", 20)]) == ({20}, set())


def test_own_process_never_allowed():
    t = PermissionTracker({"python.exe"}, own_pid=5)
    assert t.update([sess("python.exe", 5)]) == (set(), set())


def test_set_allowed_takes_effect_on_next_update():
    t = PermissionTracker({"vlc.exe"}, own_pid=1)
    s = [sess("vlc.exe", 10)]
    t.update(s)
    t.set_allowed("VLC.exe", False)
    assert t.allowed == set()
    assert t.update(s) == (set(), {10})
    t.set_allowed("vlc.exe", True)
    assert t.update(s) == ({10}, set())


def test_own_process_tree_excluded_even_when_other_root_has_same_exe():
    # two python.exe roots: 10 (ours, with real interpreter child 11) and 20 (someone else's)
    table = {1: ("explorer.exe", 0), 10: ("python.exe", 1), 11: ("python.exe", 10), 20: ("python.exe", 1)}
    t = PermissionTracker({"python.exe"}, own_pid=10, proc_table=lambda: table)
    assert t.update([sess("python.exe", 10)]) == (set(), set())
    assert t.update([sess("python.exe", 11)]) == (set(), set())
    assert t.update([sess("python.exe", 20)]) == ({20}, set())


def test_forget_makes_pid_startable_again():
    t = PermissionTracker({"vlc.exe"}, own_pid=1, proc_table=lambda: {})
    s = [sess("vlc.exe", 10)]
    assert t.update(s) == ({10}, set())
    t.forget(10)
    assert t.update(s) == ({10}, set())


def test_update_accepts_table_snapshot_without_calling_ctor_provider():
    def boom():
        raise AssertionError("provider must not be called")

    table = {1: ("explorer.exe", 0), 10: ("python.exe", 1), 11: ("python.exe", 10)}
    t = PermissionTracker({"python.exe"}, own_pid=10, proc_table=boom)
    assert t.update([sess("python.exe", 11)], table) == (set(), set())


def test_ancestor_of_own_process_excluded_and_reported():
    # explorer(1) -> WindowsTerminal(5) -> python(10, us) ; another terminal 6 is unrelated
    table = {1: ("explorer.exe", 0), 5: ("windowsterminal.exe", 1), 6: ("windowsterminal.exe", 1),
             10: ("python.exe", 5)}
    t = PermissionTracker({"windowsterminal.exe", "explorer.exe"}, own_pid=10, proc_table=lambda: table)
    s = [sess("windowsterminal.exe", 5), sess("windowsterminal.exe", 6), sess("explorer.exe", 1)]
    assert t.update(s) == ({6}, set())
    assert t.self_tree_exes == {"windowsterminal.exe", "explorer.exe"}
    assert t.update([sess("windowsterminal.exe", 6)]) == (set(), set())
    assert t.self_tree_exes == set()


# two independent msedgewebview2 trees (e.g. Teams and Widgets), each with renderer children
WEBVIEW_TABLE = {
    1: ("explorer.exe", 0),
    50: ("ms-teams.exe", 1), 100: ("msedgewebview2.exe", 50), 101: ("msedgewebview2.exe", 100),
    60: ("widgets.exe", 1), 200: ("msedgewebview2.exe", 60), 201: ("msedgewebview2.exe", 200),
}


def test_group_sessions_one_root_per_tree():
    raw = [(101, "msedgewebview2.exe", "", True), (201, "msedgewebview2.exe", "", False),
           (200, "msedgewebview2.exe", "", False)]
    got = group_sessions(raw, WEBVIEW_TABLE)
    assert sorted((s.exe, s.pid, s.active) for s in got) == [
        ("msedgewebview2.exe", 100, True), ("msedgewebview2.exe", 200, False)]


def test_group_sessions_root_stable_across_ticks():
    a = [(101, "msedgewebview2.exe", "", True), (201, "msedgewebview2.exe", "", True)]
    t = PermissionTracker({"msedgewebview2.exe"}, own_pid=9999)
    assert t.update(group_sessions(a, WEBVIEW_TABLE), WEBVIEW_TABLE) == ({100, 200}, set())
    # next tick: pycaw order changed and another child of tree 200 plays -> no stop/start flip
    b = [(200, "msedgewebview2.exe", "", True), (201, "msedgewebview2.exe", "", True), (101, "msedgewebview2.exe", "", False)]
    assert t.update(group_sessions(b, WEBVIEW_TABLE), WEBVIEW_TABLE) == (set(), set())
