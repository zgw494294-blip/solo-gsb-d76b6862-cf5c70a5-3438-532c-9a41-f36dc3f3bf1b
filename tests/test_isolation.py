"""停管隔离作业的端到端测试（pytest）。

覆盖：创建（含初始欠压/不收敛不创建）、allow_unserved 校验、切换管段的
失供/欠压/不收敛规则、仅剩水源供水量为零、重连恢复原需水量、版本冲突
（含多线程并发同一版本最多成功一次）、读取快照一致性、404/400/422 分支，
以及计算期间的 HTTP 并发顺序（可暂停求解门：健康检查/读取/其他作业不阻塞、
后来的切换先提交、迟到者 409、失败后同版本重试）。
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.isolation as iso
from app.isolation import (
    VersionConflictError,
    create_job,
    get_snapshot,
    toggle_pipe,
)
from app.main import app
from app.solver import SolverNotConverged

client = TestClient(app)


def _tree_payload():
    """树状网：S(50) ─P1→ A(需2) ─P2→ B(需3)；P1、P2 都是桥。

    基线：q_P1=5、q_P2=3，H_A=25、H_B=16，供水量 5。
    """
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 50.0},
            {"id": "A", "type": "demand", "demand": 2.0},
            {"id": "B", "type": "demand", "demand": 3.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
        ],
        "min_head": 0.0,
    }


def _loop_payload():
    """环网：S(30)，A/B 各需 1；P3(S-B) 阻力 100。

    基线各点水头约 26+（合格）；关掉 P1 或 P2 后全部流量压上 P3，
    B 点水头必跌破 0（欠压）；只有关 P3 是合法操作（H_A=26、H_B=25）。
    """
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": 30.0},
            {"id": "A", "type": "demand", "demand": 1.0},
            {"id": "B", "type": "demand", "demand": 1.0},
        ],
        "pipes": [
            {"id": "P1", "start": "S", "end": "A", "r": 1.0},
            {"id": "P2", "start": "A", "end": "B", "r": 1.0},
            {"id": "P3", "start": "S", "end": "B", "r": 100.0},
        ],
        "min_head": 0.0,
    }


def _create(payload) -> dict:
    resp = client.post("/api/isolation-jobs", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    return resp.json()["job"]


def _toggle(job_id: str, version: int, pipe_id: str):
    return client.post(f"/api/isolation-jobs/{job_id}/toggle",
                       json={"version": version, "pipe_id": pipe_id})


def _get(job_id: str) -> dict:
    resp = client.get(f"/api/isolation-jobs/{job_id}")
    assert resp.status_code == 200
    return resp.json()["job"]


# ---------------------------- 创建作业 ---------------------------- #
def test_create_job_initial_state_and_read():
    job = _create(_tree_payload())
    assert job["job_id"]
    assert job["version"] == 0
    assert job["source_node"] == "S"
    assert job["min_head"] == pytest.approx(0.0)
    assert job["allowed_unserved"] == []  # 省略时为空集合
    assert job["closed_pipes"] == [] and job["open_pipes"] == ["P1", "P2"]
    assert job["pipe_states"] == [
        {"id": "P1", "start": "S", "end": "A", "closed": False},
        {"id": "P2", "start": "A", "end": "B", "closed": False},
    ]
    assert job["unserved_nodes"] == [] and job["unserved_demand"] == 0.0
    assert job["component_nodes"] == ["S", "A", "B"]
    assert job["component_pipes"] == ["P1", "P2"]
    assert job["source_only"] is False
    assert job["source_supply"] == pytest.approx(5.0)
    assert job["total_demand"] == pytest.approx(5.0)
    assert job["solution"]["converged"] is True
    assert job["solution"]["heads"]["B"] == pytest.approx(16.0, abs=1e-9)
    # 读取同一作业：快照与创建响应一致（同一版本的整体数据）
    assert _get(job["job_id"]) == job


def test_create_job_allow_unserved_normalized_to_entry_order():
    p = _tree_payload()
    p["allow_unserved"] = ["B", "A"]  # 故意乱序：一律按节点录入顺序归并
    job = _create(p)
    assert job["allowed_unserved"] == ["A", "B"]


def test_create_job_initial_under_pressure_rejected():
    p = _tree_payload()
    p["min_head"] = 40.0  # 基线 H_A=25、H_B=16，全部欠压
    resp = client.post("/api/isolation-jobs", json=p)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert "不创建停管隔离作业" in err
    assert "“A”" in err and "“B”" in err
    assert "40" in err


def test_create_job_initial_not_converged_rejected(monkeypatch):
    def boom(net):
        raise SolverNotConverged("模拟初始求解不收敛")

    monkeypatch.setattr(iso, "solve_network", boom)
    resp = client.post("/api/isolation-jobs", json=_tree_payload())
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert "未能收敛" in err and "不创建停管隔离作业" in err
    assert "模拟初始求解不收敛" in err


def test_create_job_reuses_network_validation():
    bad = _tree_payload()
    bad["pipes"][0]["r"] = 0
    resp = client.post("/api/isolation-jobs", json=bad)
    assert resp.status_code == 422
    assert "阻力" in resp.json()["error"]

    disconnected = _tree_payload()
    disconnected["nodes"].append({"id": "Z", "type": "demand", "demand": 1.0})
    resp = client.post("/api/isolation-jobs", json=disconnected)
    assert resp.status_code == 422
    assert "连通" in resp.json()["error"]


def test_create_job_allow_unserved_validation():
    def expect_error(mutate, fragment):
        p = _tree_payload()
        mutate(p)
        resp = client.post("/api/isolation-jobs", json=p)
        assert resp.status_code == 422
        assert fragment in resp.json()["error"]

    expect_error(lambda p: p.update(allow_unserved="A"), "数组")
    expect_error(lambda p: p.update(allow_unserved=["S"]), "用水节点")
    expect_error(lambda p: p.update(allow_unserved=["ZZ"]), "用水节点")
    expect_error(lambda p: p.update(allow_unserved=["A", "A"]), "互异")
    expect_error(lambda p: p.update(allow_unserved=[{"id": "A"}]), "非空字符串")


def test_create_job_empty_body_and_nan():
    resp = client.post("/api/isolation-jobs", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    body = ('{"nodes":[{"id":"S","type":"source","head":NaN},'
            '{"id":"A","type":"demand","demand":1}],'
            '"pipes":[{"id":"P1","start":"S","end":"A","r":1}],"min_head":0}')
    resp = client.post("/api/isolation-jobs", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "JSON" in resp.json()["error"]


# ---------------------------- 切换管段：规则 ---------------------------- #
def test_toggle_close_pipe_with_allowed_unserved():
    p = _tree_payload()
    p["allow_unserved"] = ["B"]
    job = _create(p)
    resp = _toggle(job["job_id"], 0, "P2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["toggled_pipe"] == "P2" and data["toggled_to"] == "closed"
    job1 = data["job"]
    assert job1["version"] == 1
    assert job1["closed_pipes"] == ["P2"] and job1["open_pipes"] == ["P1"]
    assert job1["pipe_states"][1]["closed"] is True
    # B 失供但在允许集合内；连通区仅剩 S、A，按原需水量求解
    assert job1["unserved_nodes"] == ["B"]
    assert job1["unserved_demand"] == pytest.approx(3.0)
    assert job1["component_nodes"] == ["S", "A"]
    assert job1["component_pipes"] == ["P1"]
    assert job1["source_only"] is False
    assert job1["source_supply"] == pytest.approx(2.0)
    assert job1["total_demand"] == pytest.approx(2.0)
    assert job1["solution"]["heads"] == {"S": 50.0, "A": 46.0}
    assert set(job1["solution"]["flows"]) == {"P1"}
    # 读取快照与切换响应完全一致（不混合不同版本）
    assert _get(job["job_id"]) == job1


def test_toggle_reopen_restores_original_demand():
    """失供节点保留原需水量：重连后恢复参与求解，供水量回到基线。"""
    p = _tree_payload()
    p["allow_unserved"] = ["B"]
    job = _create(p)
    _toggle(job["job_id"], 0, "P2")
    resp = _toggle(job["job_id"], 1, "P2")  # 再次切换 = 重开
    assert resp.status_code == 200
    assert resp.json()["toggled_to"] == "open"
    job2 = resp.json()["job"]
    assert job2["version"] == 2
    assert job2["closed_pipes"] == [] and job2["open_pipes"] == ["P1", "P2"]
    assert job2["unserved_nodes"] == []
    assert job2["source_supply"] == pytest.approx(5.0)
    assert job2["total_demand"] == pytest.approx(5.0)
    assert job2["solution"]["heads"]["B"] == pytest.approx(16.0, abs=1e-9)


def test_toggle_unserved_outside_allowed_rejected_and_unchanged():
    job = _create(_tree_payload())  # 允许集合为空
    resp = _toggle(job["job_id"], 0, "P2")
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert "“B”" in err and "失供" in err and "不在允许暂时失供" in err
    assert "保持原状" in err
    # 关闭集合、版本、稳态结果全部保持原状
    got = _get(job["job_id"])
    assert got["version"] == 0
    assert got["closed_pipes"] == []
    assert got["unserved_nodes"] == []
    assert got["source_supply"] == pytest.approx(5.0)


def test_toggle_under_pressure_rejected_then_legal_toggle_still_works():
    """欠压拒绝不消耗版本：同一版本号的合法切换随后仍可成功。"""
    job = _create(_loop_payload())
    resp = _toggle(job["job_id"], 0, "P1")  # 关 P1 后 B 点水头 -370，欠压
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert "低于最低水头限值" in err and "“B”" in err and "保持原状" in err
    got = _get(job["job_id"])
    assert got["version"] == 0 and got["closed_pipes"] == []
    # 版本未被消耗：用版本 0 关 P3（合法）成功，版本 +1
    resp = _toggle(job["job_id"], 0, "P3")
    assert resp.status_code == 200
    job1 = resp.json()["job"]
    assert job1["version"] == 1
    assert job1["closed_pipes"] == ["P3"]
    assert job1["unserved_nodes"] == []
    assert job1["component_nodes"] == ["S", "A", "B"]
    assert job1["solution"]["heads"]["A"] == pytest.approx(26.0, abs=1e-9)
    assert job1["solution"]["heads"]["B"] == pytest.approx(25.0, abs=1e-9)


def test_toggle_source_only_component_supply_zero():
    """连通区仅剩水源：供水量为零，不再调用求解器，失供节点保留原需水量。"""
    p = _tree_payload()
    p["allow_unserved"] = ["A", "B"]
    job = _create(p)
    resp = _toggle(job["job_id"], 0, "P1")
    assert resp.status_code == 200
    job1 = resp.json()["job"]
    assert job1["version"] == 1
    assert job1["source_only"] is True
    assert job1["component_nodes"] == ["S"]
    assert job1["component_pipes"] == []
    assert job1["unserved_nodes"] == ["A", "B"]
    assert job1["unserved_demand"] == pytest.approx(5.0)
    assert job1["source_supply"] == pytest.approx(0.0)
    assert job1["total_demand"] == pytest.approx(0.0)
    assert job1["solution"]["heads"] == {"S": 50.0}
    assert job1["solution"]["flows"] == {}
    assert job1["solution"]["source_supply"] == pytest.approx(0.0)
    # 重开后全部恢复
    resp = _toggle(job["job_id"], 1, "P1")
    assert resp.status_code == 200
    job2 = resp.json()["job"]
    assert job2["version"] == 2
    assert job2["source_only"] is False
    assert job2["source_supply"] == pytest.approx(5.0)
    assert job2["unserved_nodes"] == []


def test_toggle_not_converged_keeps_state(monkeypatch):
    """切换后连通区求解不收敛：全部保持原状并说明原因。"""
    real_solve = iso.solve_network
    calls = {"n": 0}

    def flaky(net):
        calls["n"] += 1
        if calls["n"] >= 2:  # 第 1 次为创建时的初始求解
            raise SolverNotConverged("模拟切换后不收敛")
        return real_solve(net)

    monkeypatch.setattr(iso, "solve_network", flaky)
    job = _create(_loop_payload())
    resp = _toggle(job["job_id"], 0, "P3")
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert "未能收敛" in err and "保持原状" in err and "模拟切换后不收敛" in err
    got = _get(job["job_id"])
    assert got["version"] == 0
    assert got["closed_pipes"] == []
    assert got["source_supply"] == pytest.approx(2.0)


# ---------------------------- 版本与并发 ---------------------------- #
def test_toggle_stale_version_conflict_409():
    job = _create(_loop_payload())
    assert _toggle(job["job_id"], 0, "P3").status_code == 200  # v0 -> v1
    resp = _toggle(job["job_id"], 0, "P3")  # 旧版本重试
    assert resp.status_code == 409
    data = resp.json()
    assert data["ok"] is False
    assert "版本冲突" in data["error"]
    assert data["current_version"] == 1
    resp = _toggle(job["job_id"], 5, "P3")  # 不存在的未来版本
    assert resp.status_code == 409
    # 冲突不影响状态：仍是版本 1、P3 关闭
    got = _get(job["job_id"])
    assert got["version"] == 1 and got["closed_pipes"] == ["P3"]
    # 用当前版本重开 P3 成功
    resp = _toggle(job["job_id"], 1, "P3")
    assert resp.status_code == 200
    assert resp.json()["job"]["version"] == 2
    assert resp.json()["job"]["closed_pipes"] == []


def test_concurrent_same_version_at_most_one_success():
    """同一版本并发切换：8 个线程中恰好 1 个成功，其余全部版本冲突。"""
    snap = create_job(_loop_payload())
    job_id = snap["job_id"]
    outcomes = []
    outcomes_lock = threading.Lock()

    def worker():
        try:
            toggle_pipe(job_id, {"version": 0, "pipe_id": "P3"})
            outcome = "ok"
        except VersionConflictError:
            outcome = "conflict"
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 1
    assert outcomes.count("conflict") == 7
    assert get_snapshot(job_id)["version"] == 1
    assert get_snapshot(job_id)["closed_pipes"] == ["P3"]


def test_jobs_are_independent():
    job_a = _create(_loop_payload())
    job_b = _create(_tree_payload())
    assert job_a["job_id"] != job_b["job_id"]
    assert _toggle(job_a["job_id"], 0, "P3").status_code == 200
    # 另一个作业不受任何影响
    got_b = _get(job_b["job_id"])
    assert got_b["version"] == 0 and got_b["closed_pipes"] == []
    assert got_b["source_supply"] == pytest.approx(5.0)


# ---------------------------- 非法操作与 404 ---------------------------- #
def test_toggle_unknown_pipe_rejected():
    job = _create(_loop_payload())
    resp = _toggle(job["job_id"], 0, "PX")
    assert resp.status_code == 422
    assert "不是该作业管网中已定义的管段" in resp.json()["error"]
    assert _get(job["job_id"])["version"] == 0


def test_toggle_bad_payload_rejected():
    job = _create(_loop_payload())
    jid = job["job_id"]
    # 缺 version / 缺 pipe_id / 版本类型错误 / 版本为负 / pipe_id 为空
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"pipe_id": "P3"}).status_code == 422
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"version": 0}).status_code == 422
    resp = client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"version": "0", "pipe_id": "P3"})
    assert resp.status_code == 422 and "非负整数" in resp.json()["error"]
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"version": 0.0, "pipe_id": "P3"}).status_code == 422
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"version": -1, "pipe_id": "P3"}).status_code == 422
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json={"version": 0, "pipe_id": "  "}).status_code == 422
    # 请求体不是对象 / 空请求体
    assert client.post(f"/api/isolation-jobs/{jid}/toggle",
                       json=[1, 2]).status_code == 422
    resp = client.post(f"/api/isolation-jobs/{jid}/toggle", content=b"",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    # 全部失败后状态不变
    assert _get(jid)["version"] == 0


def test_unknown_job_returns_404():
    resp = client.get("/api/isolation-jobs/does-not-exist")
    assert resp.status_code == 404
    assert "不存在" in resp.json()["error"]
    assert "重启" in resp.json()["error"]
    resp = client.post("/api/isolation-jobs/does-not-exist/toggle",
                       json={"version": 0, "pipe_id": "P1"})
    assert resp.status_code == 404


def test_read_snapshot_never_mixes_versions():
    """连续多次成功切换后，读取到的快照内部字段始终同属一个版本。"""
    p = _tree_payload()
    p["allow_unserved"] = ["A", "B"]
    job = _create(p)
    jid = job["job_id"]
    expected_closed: list[str] = []
    for step, pipe in enumerate(["P1", "P2", "P1"], start=1):
        resp = _toggle(jid, step - 1, pipe)
        assert resp.status_code == 200
        expected_closed = resp.json()["job"]["closed_pipes"]
        got = _get(jid)
        # 版本号、关闭集合、逐管开闭、稳态结果必须互相对应
        assert got["version"] == step
        assert got["closed_pipes"] == expected_closed
        assert [ps["id"] for ps in got["pipe_states"] if ps["closed"]] \
            == expected_closed
        assert set(got["solution"]["flows"]) == set(got["component_pipes"])
        assert got["source_supply"] == pytest.approx(
            got["solution"]["source_supply"])
    # 最终状态：P1 重开、P2 仍关，B 失供
    final = _get(jid)
    assert final["version"] == 3
    assert final["closed_pipes"] == ["P2"]
    assert final["unserved_nodes"] == ["B"]
    assert final["unserved_demand"] == pytest.approx(3.0)


# ------------------ 计算期间的 HTTP 并发（可暂停求解门） ------------------ #
class _PausableCompute:
    """测试用“求解门”：命中条件的那次重算在真正求解前被暂停，可显式放行。

    不靠大管网或 sleep 制造耗时：替换 iso._COMPUTE_GATE 后，目标切换的重算
    线程会阻塞在 gate 内（此时不持有任何作业锁），测试线程可借这个确定的
    暂停点核对并发顺序，再 release() 放行让其继续提交。仅首个命中的调用被
    暂停（原子认领，其余调用直接放行），天然适配“失败后用同版本重试”。
    """

    def __init__(self, *, job_id, stage, closed=frozenset(), hold_seconds=5.0):
        self._want_job = job_id  # None 表示不挑作业（用于创建时 ID 尚未生成）
        self._want_stage = stage
        self._want_closed = frozenset(closed)
        self._hold = hold_seconds
        self._claimed = threading.Lock()
        self.arrived = threading.Event()  # 命中的重算已到达并被暂停
        self.released = threading.Event()  # 放行开关

    def __call__(self, gate_job_id, stage, *, version, closed):
        hit = ((self._want_job is None or gate_job_id == self._want_job)
               and stage == self._want_stage
               and frozenset(closed) == self._want_closed)
        if not hit:
            return
        if not self._claimed.acquire(blocking=False):
            return  # 仅首个命中者被暂停；之后（如重试）的调用直接通过
        self.arrived.set()
        if not self.released.wait(timeout=self._hold):
            raise AssertionError("求解门在超时内未被放行，测试疑似死锁")



# --------------------------------------------------------------------------- #
# 计算期间的 HTTP 并发：对真实 Uvicorn 进程（守护线程内、随机端口）发请求
# --------------------------------------------------------------------------- #
import socket  # noqa: E402

import httpx  # noqa: E402
import uvicorn  # noqa: E402


@pytest.fixture(scope="module")
def server():
    """在守护线程内启动单进程 Uvicorn（与 Compose 同一启动方式的进程模型）。

    所有并发断言都打在这个真实服务器上，而非每请求独立门户的 TestClient，
    才能证明求解期间唯一的事件循环仍可处理健康检查与读取。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="warning", lifespan="off")
    server_inst = uvicorn.Server(config)
    thread = threading.Thread(target=server_inst.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.02)
    else:
        raise RuntimeError("测试用 Uvicorn 未能在超时内就绪")
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as http:
        yield http
    server_inst.should_exit = True
    thread.join(timeout=5)


def _create_on(http, payload) -> dict:
    resp = http.post("/api/isolation-jobs", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["job"]


def _toggle_on(http, job_id, version, pipe_id):
    return http.post(f"/api/isolation-jobs/{job_id}/toggle",
                     json={"version": version, "pipe_id": pipe_id})


def _post_async_on(http, path, body):
    """守护线程内发起真实 HTTP 请求，返回 (线程, result 容器)。"""
    result: dict[str, object] = {}

    def worker():
        try:
            result["response"] = http.post(path, json=body)
        except Exception as exc:  # noqa: BLE001 - 让 join 方看到异常
            result["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, result


def test_http_concurrency_during_toggle_later_toggle_commits_first(server, monkeypatch):
    """同一版本两个切换并发：后来的切换先算完先提交，先发起的迟到提交收 409。

    暂停点内同时核对：健康检查立即 200（唯一事件循环未被占住）、读取只给
    最近已提交快照（v0）、其他作业的合法切换立即完成；迟到者收到 409 与
    current_version，按最新版本重试后成功，最终状态不被覆盖。
    """
    p = _tree_payload()
    p["allow_unserved"] = ["A", "B"]
    job = _create_on(server, p)
    jid = job["job_id"]
    other = _create_on(server, _loop_payload())  # 预先存在的另一作业

    gate = _PausableCompute(job_id=jid, stage="toggle", closed=frozenset({"P1"}))
    monkeypatch.setattr(iso, "_COMPUTE_GATE", gate)

    # A：v0 关 P1（连通区仅剩水源，合法）——重算被暂停
    t_a, res_a = _post_async_on(
        server, f"/api/isolation-jobs/{jid}/toggle",
        {"version": 0, "pipe_id": "P1"})
    assert gate.arrived.wait(timeout=5)

    # 暂停期间：健康检查立即返回（证明单一事件循环未被求解占住）
    assert server.get("/api/health").status_code == 200
    # 读取只展示最近已提交的完整快照：仍是 v0，不泄露 A 的候选结果
    got = server.get(f"/api/isolation-jobs/{jid}").json()["job"]
    assert got["version"] == 0 and got["closed_pipes"] == []
    # 其他作业的合法切换照常算完并提交
    other_resp = _toggle_on(server, other["job_id"], 0, "P3")
    assert other_resp.status_code == 200
    assert other_resp.json()["job"]["version"] == 1
    assert other_resp.json()["job"]["unserved_nodes"] == []
    # 暂停期间重复读取，永远拿到 v0 整体快照（绝不含半成品）
    assert server.get(f"/api/isolation-jobs/{jid}").json()["job"]["version"] == 0

    # B：同一作业、同一 v0，关 P2（B 失供但允许，合法）——不命中暂停门，
    # 在 A 仍被暂停时先算完并先提交：v0 -> v1
    resp_b = _toggle_on(server, jid, 0, "P2")
    assert resp_b.status_code == 200
    job1 = resp_b.json()["job"]
    assert job1["version"] == 1
    assert job1["closed_pipes"] == ["P2"]
    assert job1["unserved_nodes"] == ["B"]

    # 放行 A：其候选基于已过期的 v0 算成，提交时版本已变 -> 409，
    # current_version 为 1，且绝不覆盖 B 刚提交的状态
    gate.released.set()
    t_a.join(timeout=5)
    assert not t_a.is_alive()
    assert "error" not in res_a, res_a.get("error")
    resp_a = res_a["response"]
    assert resp_a.status_code == 409
    assert resp_a.json()["current_version"] == 1
    assert "版本冲突" in resp_a.json()["error"]
    got = server.get(f"/api/isolation-jobs/{jid}").json()["job"]
    assert got["version"] == 1 and got["closed_pipes"] == ["P2"]

    # 按 409 携带的 current_version 重试 A 的意图（在 v1 上再关 P1）：
    # A、B 同时失供但均在允许集合内，成功提交 v2
    resp_retry = _toggle_on(server, jid, 1, "P1")
    assert resp_retry.status_code == 200
    job2 = resp_retry.json()["job"]
    assert job2["version"] == 2
    assert job2["closed_pipes"] == ["P1", "P2"]
    assert job2["unserved_nodes"] == ["A", "B"]
    assert job2["source_only"] is True
    assert server.get(f"/api/isolation-jobs/{jid}").json()["job"] == job2


def test_http_concurrency_failing_toggle_unserved_keeps_state_then_retry(
        server, monkeypatch):
    """被暂停的切换最终因失供规则失败：耗时期间服务可交互，失败不推进版本。

    放行返回 422 后，用同一版本 0 改发另一合法切换仍然成功。
    """
    p = _tree_payload()
    p["allow_unserved"] = ["B"]  # 关 P1 会让 A 失供（不允许）；关 P2 合法
    job = _create_on(server, p)
    jid = job["job_id"]
    gate = _PausableCompute(job_id=jid, stage="toggle", closed=frozenset({"P1"}))
    monkeypatch.setattr(iso, "_COMPUTE_GATE", gate)

    t_fail, res_fail = _post_async_on(
        server, f"/api/isolation-jobs/{jid}/toggle",
        {"version": 0, "pipe_id": "P1"})
    assert gate.arrived.wait(timeout=5)

    # 慢失败计算期间：健康检查与读取都立即返回，快照仍为 v0
    assert server.get("/api/health").status_code == 200
    assert server.get(f"/api/isolation-jobs/{jid}").json()["job"]["version"] == 0

    gate.released.set()
    t_fail.join(timeout=5)
    resp_fail = res_fail["response"]
    assert resp_fail.status_code == 422
    err = resp_fail.json()["error"]
    assert "“A”" in err and "失供" in err and "保持原状" in err
    # 失败不推进版本、不改变关闭集合与稳态结果
    got = server.get(f"/api/isolation-jobs/{jid}").json()["job"]
    assert got["version"] == 0 and got["closed_pipes"] == []
    assert got["source_supply"] == pytest.approx(5.0)

    # 同一版本 0 改发合法切换（关 P2，B 在允许集合内）仍可成功
    resp = _toggle_on(server, jid, 0, "P2")
    assert resp.status_code == 200
    job1 = resp.json()["job"]
    assert job1["version"] == 1 and job1["closed_pipes"] == ["P2"]
    assert job1["unserved_nodes"] == ["B"]


def test_http_concurrency_failing_toggle_under_pressure_keeps_version(
        server, monkeypatch):
    """被暂停的切换最终因欠压失败：失败后同一版本的合法切换照常成功。"""
    job = _create_on(server, _loop_payload())
    jid = job["job_id"]
    gate = _PausableCompute(job_id=jid, stage="toggle", closed=frozenset({"P1"}))
    monkeypatch.setattr(iso, "_COMPUTE_GATE", gate)

    t_bad, res_bad = _post_async_on(
        server, f"/api/isolation-jobs/{jid}/toggle",
        {"version": 0, "pipe_id": "P1"})  # 关 P1 后 B 欠压
    assert gate.arrived.wait(timeout=5)
    assert server.get("/api/health").status_code == 200  # 计算期间仍可交互

    gate.released.set()
    t_bad.join(timeout=5)
    resp_bad = res_bad["response"]
    assert resp_bad.status_code == 422
    assert "低于最低水头限值" in resp_bad.json()["error"]
    assert server.get(f"/api/isolation-jobs/{jid}").json()["job"]["version"] == 0
    # 欠压失败也不消耗版本：仍用版本 0 关 P3（合法）成功
    resp = _toggle_on(server, jid, 0, "P3")
    assert resp.status_code == 200
    assert resp.json()["job"]["version"] == 1
    assert resp.json()["job"]["closed_pipes"] == ["P3"]


def test_http_concurrency_during_create_other_requests_still_served(
        server, monkeypatch):
    """创建作业本身在求解时，健康检查、已有作业读取和其他作业切换也能返回。"""
    existing = _create_on(server, _loop_payload())
    # 创建作业的 job_id 随机生成：job_id=None 表示不挑作业，只匹配创建重算
    gate = _PausableCompute(job_id=None, stage="create")
    monkeypatch.setattr(iso, "_COMPUTE_GATE", gate)

    t_new, res_new = _post_async_on(server, "/api/isolation-jobs", _tree_payload())
    assert gate.arrived.wait(timeout=5)

    # 创建的初始求解期间：健康检查、已存在作业读取均立即返回
    assert server.get("/api/health").status_code == 200
    got = server.get(f"/api/isolation-jobs/{existing['job_id']}").json()["job"]
    assert got["version"] == 0 and got["closed_pipes"] == []
    # 其他作业的合法切换也能在创建结束前算完提交
    resp = _toggle_on(server, existing["job_id"], 0, "P3")
    assert resp.status_code == 200
    assert resp.json()["job"]["version"] == 1

    # 放行创建：返回版本 0 快照；作业 ID 与已有作业不同
    gate.released.set()
    t_new.join(timeout=5)
    created = res_new["response"]
    assert created.status_code == 200, created.text
    new_job = created.json()["job"]
    assert new_job["version"] == 0
    assert new_job["job_id"] != existing["job_id"]
    assert new_job["source_supply"] == pytest.approx(5.0)
    # 放行期间被切换的已有作业状态不受创建影响
    final = server.get(
        f"/api/isolation-jobs/{existing['job_id']}").json()["job"]
    assert final["version"] == 1 and final["closed_pipes"] == ["P3"]
    # 创建放行后读到的仍是它自己版本 0 的完整快照
    assert server.get(
        f"/api/isolation-jobs/{new_job['job_id']}").json()["job"] == new_job


def test_http_concurrent_same_version_two_legal_toggles_one_wins(
        server, monkeypatch):
    """两个同版本合法切换同时计算：先算完者提交成功，后到提交点的收 409。

    先发起者先完成计算但提交被门挡住，另一切换走完计算并先提交 v1；
    随后放行先发起者，验证无论到达提交点先后，同一版本最多一次成功提交，
    且 409 带当前版本、不覆盖新状态。
    """
    p = _tree_payload()
    p["allow_unserved"] = ["A", "B"]
    job = _create_on(server, p)
    jid = job["job_id"]
    gate = _PausableCompute(job_id=jid, stage="toggle", closed=frozenset({"P2"}))
    monkeypatch.setattr(iso, "_COMPUTE_GATE", gate)

    t_p2, res_p2 = _post_async_on(
        server, f"/api/isolation-jobs/{jid}/toggle",
        {"version": 0, "pipe_id": "P2"})
    assert gate.arrived.wait(timeout=5)

    # P2 已算好（提交被暂停）时，P1 从计算到提交一次走完 -> v1
    resp_p1 = _toggle_on(server, jid, 0, "P1")
    assert resp_p1.status_code == 200
    assert resp_p1.json()["job"]["version"] == 1
    assert resp_p1.json()["job"]["closed_pipes"] == ["P1"]

    # 放行先发起的 P2：候选基于 v0，提交时版本已是 1 -> 409，不覆盖
    gate.released.set()
    t_p2.join(timeout=5)
    resp_p2 = res_p2["response"]
    assert resp_p2.status_code == 409
    assert resp_p2.json()["current_version"] == 1
    got = server.get(f"/api/isolation-jobs/{jid}").json()["job"]
    assert got["version"] == 1 and got["closed_pipes"] == ["P1"]
