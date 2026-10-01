"""停管隔离作业的端到端测试（pytest）。

覆盖：创建（含初始欠压/不收敛不创建）、allow_unserved 校验、切换管段的
失供/欠压/不收敛规则、仅剩水源供水量为零、重连恢复原需水量、版本冲突
（含多线程并发同一版本最多成功一次）、读取快照一致性、404/400/422 分支，
以及用“可暂停并释放”的求解器做真实 HTTP 并发：验证耗时计算期间健康检查、
已存在作业读取与其他合法切换不被阻塞，后来的同作业切换可先算完并提交、
先到者提交时返回 409 且不覆盖新状态，以及失败（422）后同版本重试。
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


# --------------------------------------------------------------------------- #
# 真实 HTTP 并发：可暂停并释放的求解器（不依赖大管网或长时间等待）
# --------------------------------------------------------------------------- #
class _PauseGate:
    """测试用“闸门”：按求解子网管段集合作为键拦截 solve_network 调用。

    arm(key) 后，对该子网络的求解会阻塞在闸门处（记录到达、计数 +1），
    直到 release(key) 放行（可指定放行时抛出不收敛）；到达后可用 wait_arrived
    同步。拦截为一次性的（释放时摘除键值），故重试不会再次被同一键拦住。
    不放大管网、不做固定睡眠：未被 arm 的键直接正常求解。
    """

    def __init__(self) -> None:
        self._armed: dict[str, dict[str, object]] = {}
        self._cv = threading.Condition()
        self.hits: dict[str, int] = {}
        self.total_calls = 0

    def arm(self, key: str, *, fail: bool = False) -> None:
        with self._cv:
            self._armed[key] = {"released": False, "fail": fail, "arrived": 0}
            self.hits[key] = 0

    def _pipe_key(self, net) -> str:
        return ",".join(p["id"] for p in net.pipes)

    def __call__(self, net):
        self.total_calls += 1
        key = self._pipe_key(net)
        with self._cv:
            slot = self._armed.get(key)
            if slot is None:
                return self._real(net)
            slot["arrived"] = int(slot["arrived"]) + 1
            self.hits[key] = self.hits.get(key, 0) + 1
            self._cv.notify_all()
            while not slot["released"]:
                self._cv.wait()
            fail = bool(slot["fail"])
            # 一次性拦截：释放后摘除，后续相同子网络（如重试）直接正常求解
            self._armed.pop(key, None)
        if fail:
            raise SolverNotConverged("闸门注入：本次求解不收敛")
        return self._real(net)

    def wait_arrived(self, key: str, *, count: int = 1, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cv:
            while int(self._armed.get(key, {}).get("arrived", 0)) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cv.wait(remaining):
                    raise AssertionError(f"等待求解到达闸门 {key!r} 超时")

    def release(self, key: str) -> None:
        with self._cv:
            slot = self._armed.get(key)
            if slot is not None:
                slot["released"] = True
            self._cv.notify_all()

    def install(self, monkeypatch) -> None:
        self._real = iso.solve_network
        monkeypatch.setattr(iso, "solve_network", self)


class _Background:
    """在后台线程发起 TestClient 请求并保留结果（每个线程使用独立 client）。"""

    def __init__(self, method: str, path: str, **kwargs) -> None:
        def run():
            local = TestClient(app)
            try:
                self.response = getattr(local, method)(path, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 主线程断言时统一报告
                self.error = exc

        self.error: Exception | None = None
        self.response = None
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self, timeout: float = 10.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "后台 HTTP 请求超时未返回"
        if self.error is not None:
            raise self.error
        return self.response


def _chain_payload(*, source_head=50.0):
    """链状网 S─QX→X─QY→Y：QX 关闭则 X、Y 都失供（源端平凡解，无需求解）。"""
    return {
        "nodes": [
            {"id": "S", "type": "source", "head": source_head},
            {"id": "X", "type": "demand", "demand": 2.0},
            {"id": "Y", "type": "demand", "demand": 3.0},
        ],
        "pipes": [
            {"id": "QX", "start": "S", "end": "X", "r": 1.0},
            {"id": "QY", "start": "X", "end": "Y", "r": 1.0},
        ],
        "min_head": 0.0,
        "allow_unserved": ["X", "Y"],
    }


def test_http_concurrency_during_create_and_toggle(monkeypatch):
    """创建（慢求解）进行期间：健康检查、已有作业读取、其他作业的合法切换立即返回。"""
    gate = _PauseGate()
    gate.install(monkeypatch)
    # 创建新作业时子网络为全部开启管段，键即两根管
    gate.arm("QX,QY")

    pending_create = _Background(
        "post", "/api/isolation-jobs", json=_chain_payload())
    gate.wait_arrived("QX,QY")

    # 1) 健康检查不被耗时计算阻塞
    t0 = time.monotonic()
    resp = TestClient(app).get("/api/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}
    assert time.monotonic() - t0 < 2.0

    # 2) 已存在作业的读取立即返回最近已提交的完整快照（版本 0），不等待计算
    existing = _create(_tree_payload())
    t0 = time.monotonic()
    got = _get(existing["job_id"])
    assert time.monotonic() - t0 < 2.0
    assert got == existing and got["version"] == 0

    # 3) 另一个作业的合法切换（关桥 P2 → B 失供但允许）：连通区剩 S、A，
    #    子网管段集合 "P1"，键不同，立即算完提交，不等闸门
    other = _create({**_tree_payload(), "allow_unserved": ["B"]})
    t0 = time.monotonic()
    resp = _toggle(other["job_id"], 0, "P2")
    assert resp.status_code == 200
    assert time.monotonic() - t0 < 2.0
    assert resp.json()["job"]["version"] == 1
    assert resp.json()["job"]["closed_pipes"] == ["P2"]
    # 慢创建尚未提交：作业注册表中仍读不到它（读到的是创建前的世界）
    # 而 other 作业的读取展示的就是刚刚提交的版本 1
    assert _get(other["job_id"])["version"] == 1

    # 4) 放行创建：返回版本 0 快照
    gate.release("QX,QY")
    created = pending_create.join().json()["job"]
    assert created["version"] == 0
    assert _get(created["job_id"])["closed_pipes"] == []


def test_http_same_job_later_toggle_may_commit_first(monkeypatch):
    """同一作业：后发的切换先算完并提交；先来者提交时版本已变 → 409，不覆盖。

    旧实现里先来的切换会在整个求解期间占住作业锁，后发切换必然排队等待；
    新实现求解在锁外进行，后发的关 QX（仅余水源、平凡解）立即提交 0→1，
    先来者放行后在提交阶段复核版本，得到 409 与 current_version，其结果
    （关闭集合 {QY}）被丢弃，绝不覆盖已提交的 {QX}。
    """
    gate = _PauseGate()
    gate.install(monkeypatch)
    job = _create(_chain_payload())
    jid = job["job_id"]

    gate.arm("QX")  # 先来者：v0 关 QY → 连通区 S,X 需解 "QX"，被闸门拦住
    slow_a = _Background(
        "post", f"/api/isolation-jobs/{jid}/toggle",
        json={"version": 0, "pipe_id": "QY"})
    gate.wait_arrived("QX")

    # 计算期间：读取只展示最近已提交的完整快照（仍为版本 0），健康检查正常
    assert _get(jid)["version"] == 0
    assert _get(jid)["closed_pipes"] == []
    assert TestClient(app).get("/api/health").status_code == 200

    # 后来的切换（同版本 0、关 QX）：连通区仅余水源，平凡解不调求解器，
    # 不被闸门阻塞，立即算完并先提交：版本 0 → 1
    t0 = time.monotonic()
    resp_b = _toggle(jid, 0, "QX")
    assert resp_b.status_code == 200
    assert time.monotonic() - t0 < 2.0
    snap_b = resp_b.json()["job"]
    assert snap_b["version"] == 1 and snap_b["closed_pipes"] == ["QX"]
    assert snap_b["source_only"] is True
    # A 仍在计算：读取已能看到 B 提交的完整版本 1 快照
    assert _get(jid)["version"] == 1
    assert _get(jid)["closed_pipes"] == ["QX"]

    # 放行先来者 A：提交时版本已被推进到 1，返回 409 + current_version=1，
    # A 的结果（关闭集合 {QY}、快照版本 1）被丢弃，不覆盖 B 的新状态
    gate.release("QX")
    resp_a = slow_a.join()
    assert resp_a.status_code == 409
    body_a = resp_a.json()
    assert body_a["ok"] is False and body_a["current_version"] == 1
    assert "版本冲突" in body_a["error"]

    final = _get(jid)
    assert final["version"] == 1
    assert final["closed_pipes"] == ["QX"]
    assert final["unserved_nodes"] == ["X", "Y"]

    # 客户端按 current_version 重读后携带新版本重试：v1 再关 QY（仍仅余
    # 水源，平凡解）成功，版本 1 → 2，且同版本再来一次立即 409
    resp = _toggle(jid, 1, "QY")
    assert resp.status_code == 200
    assert resp.json()["job"]["version"] == 2
    assert resp.json()["job"]["closed_pipes"] == ["QX", "QY"]
    dup = _toggle(jid, 1, "QY")
    assert dup.status_code == 409 and dup.json()["current_version"] == 2


def test_http_failed_toggle_unblocks_and_retry_same_version(monkeypatch):
    """慢切换在闸门处失败（422，不推进版本）后：同版本合法重试成功。"""
    gate = _PauseGate()
    gate.install(monkeypatch)
    job = _create(_chain_payload())
    jid = job["job_id"]

    # v0 关 QY：连通区 S,X 需解 "QX"；放行时让其“不收敛”→ 422
    gate.arm("QX", fail=True)
    slow = _Background(
        "post", f"/api/isolation-jobs/{jid}/toggle",
        json={"version": 0, "pipe_id": "QY"})
    gate.wait_arrived("QX")

    # 计算失败期间其他操作照常：健康检查、读取（仍版本 0）
    assert TestClient(app).get("/api/health").status_code == 200
    assert _get(jid)["version"] == 0

    gate.release("QX")
    resp = slow.join()
    assert resp.status_code == 422
    assert "未能收敛" in resp.json()["error"] and "保持原状" in resp.json()["error"]

    # 失败不推进版本：仍是版本 0、无关闭管段
    got = _get(jid)
    assert got["version"] == 0 and got["closed_pipes"] == []

    # 同版本改发另一个合法操作（关 QX：仅余水源，平凡解不调求解器）成功
    resp = _toggle(jid, 0, "QX")
    assert resp.status_code == 200
    snap = resp.json()["job"]
    assert snap["version"] == 1
    assert snap["closed_pipes"] == ["QX"]
    assert snap["source_only"] is True and snap["unserved_nodes"] == ["X", "Y"]


def test_http_transient_failure_then_retry_recomputes(monkeypatch):
    """慢切换第一次放行时不收敛（422），拦截为一次性：同版本重试正常求解并成功。"""
    gate = _PauseGate()
    gate.install(monkeypatch)
    p = _chain_payload()
    job = _create(p)
    jid = job["job_id"]

    gate.arm("QX", fail=True)
    slow = _Background(
        "post", f"/api/isolation-jobs/{jid}/toggle",
        json={"version": 0, "pipe_id": "QY"})
    gate.wait_arrived("QX")
    gate.release("QX")
    assert slow.join().status_code == 422
    total_before = gate.total_calls
    assert gate.hits["QX"] == 1

    # 闸门键已摘除：同版本同管段重试走真实求解（S、X 连通区），成功提交
    t0 = time.monotonic()
    resp = _toggle(jid, 0, "QY")
    assert resp.status_code == 200
    assert time.monotonic() - t0 < 2.0
    snap = resp.json()["job"]
    assert snap["version"] == 1 and snap["closed_pipes"] == ["QY"]
    assert snap["component_nodes"] == ["S", "X"]
    assert snap["unserved_nodes"] == ["Y"]
    # 重试重新触发了一次求解（没有被旧闸门吞掉，也没复用失败结果）
    assert gate.total_calls == total_before + 1
    assert gate.hits["QX"] == 1  # 第二次为未拦截的直通


def test_http_concurrent_fail_then_other_version_commits(monkeypatch):
    """慢切换失败计算进行中，另一同版本合法切换已提交：失败结果不得覆盖新状态。"""
    gate = _PauseGate()
    gate.install(monkeypatch)
    job = _create(_chain_payload())
    jid = job["job_id"]

    gate.arm("QX", fail=True)  # 慢：v0 关 QY，放行时不收敛
    slow = _Background(
        "post", f"/api/isolation-jobs/{jid}/toggle",
        json={"version": 0, "pipe_id": "QY"})
    gate.wait_arrived("QX")

    # 同版本另一合法切换（关 QX：仅余水源，平凡解）立即成功：0 → 1
    resp = _toggle(jid, 0, "QX")
    assert resp.status_code == 200
    assert resp.json()["job"]["version"] == 1
    assert _get(jid)["closed_pipes"] == ["QX"]

    # 放行失败的慢切换：规则失败返回 422，绝不能覆盖已提交的版本 1
    gate.release("QX")
    resp = slow.join()
    assert resp.status_code == 422
    got = _get(jid)
    assert got["version"] == 1 and got["closed_pipes"] == ["QX"]
