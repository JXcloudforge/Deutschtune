import pytest

from api.server import validate_loopback_host


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_server_accepts_loopback_hosts(host):
    assert validate_loopback_host(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.10", "example.invalid", ""])
def test_server_rejects_non_loopback_hosts(host):
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host(host)
