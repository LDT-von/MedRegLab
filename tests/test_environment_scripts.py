"""Exercise setup safeguards with a fake Python; no packages are installed."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _bash():
    candidate = shutil.which("bash")
    if candidate:
        return candidate
    git = shutil.which("git")
    if os.name == "nt" and git:
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    pytest.skip("bash is not installed")


def _shell_path(bash, path):
    if os.name != "nt":
        return str(path)
    return subprocess.check_output([bash, "-c", 'cygpath -u -- "$1"', "_", str(path)], text=True).strip()


def _fixture(tmp_path):
    bash = _bash()
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "setup_medreg_env.sh"
    shutil.copyfile(source, scripts / "setup_medreg_env.sh")
    (project / "requirements.txt").write_text("", encoding="utf-8")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fakepython = fakebin / "python3"
    fakepython.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s\\n' \"${PIP_CACHE_DIR-UNSET}\" \"${TMPDIR-UNSET}\" \"$*\" >> \"$MEDREG_TEST_LOG\"\n"
        "if [[ ${1:-} == -c ]]; then exit ${MEDREG_FAKE_VERSION_EXIT:-0}; fi\n"
        "if [[ ${1:-} == -m && ${2:-} == venv ]]; then\n"
        "  mkdir -p -- \"$3/bin\"\n"
        "  cp -- \"$0\" \"$3/bin/python\"\n"
        "  chmod +x \"$3/bin/python\"\n"
        "fi\n"
        "exit 0\n", encoding="utf-8", newline="\n")
    subprocess.run([bash, "-c", 'chmod +x "$1"', "_", _shell_path(bash, fakepython)], check=True)
    env = os.environ.copy()
    env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
    env["MEDREG_TEST_LOG"] = _shell_path(bash, tmp_path / "fake-python.log")
    env.pop("PIP_CACHE_DIR", None)
    env.pop("TMPDIR", None)
    env.pop("MEDREG_VENV_PATH", None)
    env.pop("MEDREG_FAKE_VERSION_EXIT", None)
    return bash, project, env, tmp_path / "fake-python.log"


def _run(bash, project, env):
    return subprocess.run([bash, "scripts/setup_medreg_env.sh"], cwd=project, env=env,
                          text=True, capture_output=True, timeout=30)


def test_default_cache_and_tmp_stay_inside_new_project(tmp_path):
    bash, project, env, logfile = _fixture(tmp_path)
    result = _run(bash, project, env)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = logfile.read_text(encoding="utf-8").splitlines()
    expected_cache = _shell_path(bash, project / ".cache" / "pip")
    expected_tmp = _shell_path(bash, project / ".cache" / "tmp")
    pip_lines = [line for line in lines if "-m pip " in line]
    assert len(pip_lines) == 4
    assert all(line.startswith(f"{expected_cache}|{expected_tmp}|") for line in pip_lines)
    assert (project / ".cache" / "pip").is_dir()
    assert (project / ".cache" / "tmp").is_dir()


def test_explicit_cache_and_tmp_are_respected(tmp_path):
    bash, project, env, logfile = _fixture(tmp_path)
    cache, temporary = tmp_path / "custom-cache", tmp_path / "custom-tmp"
    env["PIP_CACHE_DIR"] = _shell_path(bash, cache)
    env["TMPDIR"] = _shell_path(bash, temporary)
    result = _run(bash, project, env)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = logfile.read_text(encoding="utf-8").splitlines()
    assert all(line.startswith(f"{env['PIP_CACHE_DIR']}|{env['TMPDIR']}|")
               for line in lines if "-m pip " in line)
    assert cache.is_dir() and temporary.is_dir()
    assert not (project / ".cache").exists()


def test_existing_environment_and_old_python_are_rejected_before_creation(tmp_path):
    bash, project, env, logfile = _fixture(tmp_path)
    existing = project / ".venv"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = _run(bash, project, env)
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not logfile.exists()
    env["MEDREG_VENV_PATH"] = _shell_path(bash, project / "new-env")
    env["MEDREG_FAKE_VERSION_EXIT"] = "9"
    result = _run(bash, project, env)
    assert result.returncode != 0
    assert not (project / "new-env").exists()
    assert not (project / ".cache").exists()
