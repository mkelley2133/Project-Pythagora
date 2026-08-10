from workers.tasks import healthcheck


def test_worker_healthcheck_task() -> None:
    result = healthcheck()
    assert result == {"status": "ok", "service": "workers"}
