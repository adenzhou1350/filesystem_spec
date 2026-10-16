import json
import pickle
import shlex
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

import fsspec

requests = pytest.importorskip("requests")

from fsspec.implementations.webhdfs import WebHDFS  # noqa: E402


@pytest.fixture
def webhdfs_read_server():
    content = b"0123456789abcdefghijklmnopqrstuvwxyz"
    state = SimpleNamespace(status=200, redirect=True, proxy=False, requests=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append(self.path)
            target = urlsplit(self.path)
            query = parse_qs(target.query)
            status, headers = 200, {}
            if query.get("op") == ["GETFILESTATUS"]:
                body = json.dumps(
                    {"FileStatus": {"type": "FILE", "length": len(content)}}
                ).encode()
            elif target.path == "/webhdfs/v1/file" and state.redirect:
                status, body = 307, b""
                host = "datanode.invalid" if state.proxy else "127.0.0.1"
                headers["Location"] = (
                    f"http://{host}:{self.server.server_port}/data?{target.query}"
                )
            else:
                status = state.status
                start = int(query.get("offset", [0])[0])
                length = int(query.get("length", [len(content)])[0])
                body = content[start : start + length] if status == 200 else b"HTTP error"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        try:
            yield SimpleNamespace(port=httpd.server_port, state=state, content=content)
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("status", [403, 404, 500, 503])
def test_read_rejects_datanode_http_error(webhdfs_read_server, status):
    server = webhdfs_read_server
    server.state.status = status
    fs = WebHDFS(
        "127.0.0.1", port=server.port, user="testuser", skip_instance_cache=True
    )
    try:
        with fs.open("/file", "rb", block_size=4) as stream:
            with pytest.raises(requests.HTTPError) as error:
                stream.read(5)
            assert error.value.response.status_code == status
            assert urlsplit(error.value.response.url).path == "/data"
            assert stream.tell() == 0
    finally:
        fs.session.close()


@pytest.mark.parametrize("redirect,proxy", [(False, False), (True, False), (True, True)])
def test_read_success_with_and_without_redirect(webhdfs_read_server, redirect, proxy):
    server = webhdfs_read_server
    server.state.redirect, server.state.proxy = redirect, proxy
    fs = WebHDFS(
        "127.0.0.1",
        port=server.port,
        user="testuser",
        skip_instance_cache=True,
        data_proxy={"datanode.invalid": "127.0.0.1"} if proxy else None,
    )
    try:
        with fs.open("/file", "rb", block_size=4) as stream:
            assert stream.read(5) == server.content[:5]
            stream.seek(7)
            assert stream.read(8) == server.content[7:15]
            before = len(server.state.requests)
            stream.seek(len(server.content))
            assert stream.read(5) == b""
            assert len(server.state.requests) == before
        assert any(path.startswith("/data?") for path in server.state.requests) == redirect
    finally:
        fs.session.close()


def test_read_preserves_direct_http_error(webhdfs_read_server):
    server = webhdfs_read_server
    server.state.redirect, server.state.status = False, 503
    fs = WebHDFS(
        "127.0.0.1", port=server.port, user="testuser", skip_instance_cache=True
    )
    try:
        with fs.open("/file", "rb", block_size=4) as stream:
            with pytest.raises(requests.HTTPError) as error:
                stream.read(5)
            assert error.value.response.status_code == 503
            assert urlsplit(error.value.response.url).path == "/webhdfs/v1/file"
    finally:
        fs.session.close()


@pytest.fixture(scope="module")
def hdfs_cluster():
    cmd0 = shlex.split("htcluster shutdown")
    try:
        subprocess.check_output(cmd0, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        pytest.skip("htcluster not found")
    except subprocess.CalledProcessError as ex:
        pytest.skip(f"htcluster failed: {ex.output.decode()}")
    cmd1 = shlex.split("htcluster startup --image base")
    subprocess.check_output(cmd1)
    try:
        while True:
            t = 90
            try:
                requests.get("http://localhost:50070/webhdfs/v1/?op=LISTSTATUS")
            except:  # noqa: E722
                t -= 1
                assert t > 0, "Timeout waiting for HDFS"
                time.sleep(1)
                continue
            break
        time.sleep(7)
        yield "localhost"
    finally:
        subprocess.check_output(cmd0)


def test_pickle(hdfs_cluster):
    w = WebHDFS(hdfs_cluster, user="testuser")
    w2 = pickle.loads(pickle.dumps(w))
    assert w == w2


def test_simple(hdfs_cluster):
    w = WebHDFS(hdfs_cluster, user="testuser")
    home = w.home_directory()
    assert home == "/user/testuser"
    with pytest.raises(PermissionError):
        w.mkdir("/root")


def test_url(hdfs_cluster):
    url = "webhdfs://testuser@localhost:50070/user/testuser/myfile"
    fo = fsspec.open(url, "wb", data_proxy={"worker.example.com": "localhost"})
    with fo as f:
        f.write(b"hello")
    fo = fsspec.open(url, "rb", data_proxy={"worker.example.com": "localhost"})
    with fo as f:
        assert f.read() == b"hello"


def test_workflow(hdfs_cluster):
    w = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )
    fn = "/user/testuser/testrun/afile"
    w.mkdir("/user/testuser/testrun")
    with w.open(fn, "wb") as f:
        f.write(b"hello")
    assert w.exists(fn)
    info = w.info(fn)
    assert info["size"] == 5
    assert w.isfile(fn)
    assert w.cat(fn) == b"hello"
    w.rm("/user/testuser/testrun", recursive=True)
    assert not w.exists(fn)


def test_with_gzip(hdfs_cluster):
    from gzip import GzipFile

    w = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )
    fn = "/user/testuser/gzfile"
    with w.open(fn, "wb") as f:
        gf = GzipFile(fileobj=f, mode="w")
        gf.write(b"hello")
        gf.close()
    with w.open(fn, "rb") as f:
        gf = GzipFile(fileobj=f, mode="r")
        assert gf.read() == b"hello"


def test_workflow_transaction(hdfs_cluster):
    w = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )
    fn = "/user/testuser/testrun/afile"
    w.mkdirs("/user/testuser/testrun")
    with w.transaction:
        with w.open(fn, "wb") as f:
            f.write(b"hello")
        assert not w.exists(fn)
    assert w.exists(fn)
    assert w.ukey(fn)
    files = w.ls("/user/testuser/testrun", True)
    summ = w.content_summary("/user/testuser/testrun")
    assert summ["length"] == files[0]["size"]
    assert summ["fileCount"] == 1

    w.rm("/user/testuser/testrun", recursive=True)
    assert not w.exists(fn)


def test_webhdfs_cp_file(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )

    src, dst = "/user/testuser/testrun/f1", "/user/testuser/testrun/f2"

    fs.mkdir("/user/testuser/testrun")

    with fs.open(src, "wb") as f:
        f.write(b"hello")

    fs.cp_file(src, dst)

    assert fs.exists(src)
    assert fs.exists(dst)
    assert fs.cat(src) == fs.cat(dst)


def test_path_with_equals(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )
    path_with_equals = "/user/testuser/some_table/datestamp=2023-11-11"

    fs.mkdir(path_with_equals)

    result = fs.ls(path_with_equals)
    assert result is not None
    assert fs.exists(path_with_equals)


def test_error_handling_with_equals_in_path(hdfs_cluster):
    fs = WebHDFS(hdfs_cluster, user="testuser")
    invalid_path_with_equals = (
        "/user/testuser/some_table/invalid_path=datestamp=2023-11-11"
    )

    with pytest.raises(FileNotFoundError):
        fs.ls(invalid_path_with_equals)


def test_create_and_touch_file_with_equals(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    base_path = "/user/testuser/some_table/datestamp=2023-11-11"
    file_path = f"{base_path}/testfile.txt"

    fs.mkdir(base_path)
    fs.touch(file_path, "wb")
    assert fs.exists(file_path)


def test_write_read_verify_file_with_equals(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    base_path = "/user/testuser/some_table/datestamp=2023-11-11"
    file_path = f"{base_path}/testfile.txt"
    content = b"This is some content!"

    fs.mkdir(base_path)
    with fs.open(file_path, "wb") as f:
        f.write(content)

    with fs.open(file_path, "rb") as f:
        assert f.read() == content

    file_info = fs.ls(base_path, detail=True)
    assert len(file_info) == 1
    assert file_info[0]["name"] == file_path
    assert file_info[0]["size"] == len(content)


def test_protocol_prefixed_path(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster, user="testuser", data_proxy={"worker.example.com": "localhost"}
    )
    protocol_prefixed_path = "webhdfs://localhost:50070/user/testuser/test_dir"

    fs.mkdir(protocol_prefixed_path)
    assert fs.exists(protocol_prefixed_path)

    file_info = fs.ls(protocol_prefixed_path, detail=True)
    assert len(file_info) == 0


def test_modified_nonexistent_path(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    nonexistent_path = "/user/testuser/nonexistent_file.txt"

    with pytest.raises(FileNotFoundError):
        fs.modified(nonexistent_path)


def test_modified_time(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    dir_path = "/user/testuser/"
    file_path = f"{dir_path}/testfile.txt"

    fs.mkdir(dir_path)

    # Check first modified time for directories
    modified_dir_date: datetime = fs.modified(dir_path)

    # I think it is the only thing we can assume, but I'm not sure if the server has a different time
    assert modified_dir_date <= datetime.now()

    # Create a file and check modified time again
    with fs.open(file_path, "wb") as f:
        f.write(b"test content")

    modified_file_date: datetime = fs.modified(file_path)
    assert modified_file_date >= modified_dir_date
    assert modified_file_date <= datetime.now()


# NOTE: These following two tests are a copy of the modified ones, as
# WebHDFS does not have a created time API, we are using modified as a proxy.


def test_created_nonexistent_path(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    nonexistent_path = "/user/testuser/nonexistent_file.txt"

    with pytest.raises(FileNotFoundError):
        fs.created(nonexistent_path)


def test_created_time(hdfs_cluster):
    fs = WebHDFS(
        hdfs_cluster,
        user="testuser",
        data_proxy={"worker.example.com": "localhost"},
    )
    dir_path = "/user/testuser/"
    file_path = f"{dir_path}/testfile.txt"

    fs.mkdir(dir_path)

    time.sleep(1)

    # Check first created time for directories
    created_dir_date: datetime = fs.created(dir_path)

    # I think it is the only thing we can assume, but I'm not sure if the server has a different time
    assert created_dir_date < datetime.now()

    # Create a file and check created time again
    with fs.open(file_path, "wb") as f:
        f.write(b"test content")

    time.sleep(1)

    created_file_date: datetime = fs.created(file_path)
    assert created_file_date > created_dir_date
    assert created_file_date < datetime.now()


@pytest.mark.parametrize("mode", [0o755, "755"], ids=["int", "str"])
def test_chmod_sends_octal_permission(monkeypatch, mode):
    fs = WebHDFS("localhost", skip_instance_cache=True)
    sent = {}

    def request(method, url, params, **kwargs):
        sent.update(params)
        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(fs.session, "request", request)
    fs.chmod("/user/testuser/file", mode)
    assert sent["permission"] == "755"
