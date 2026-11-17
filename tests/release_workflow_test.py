import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.version import Version

SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def distributions(tmp_path):
    def make(wheel_version="1.2.3", sdist_version=None, name="PowerGenome"):
        sdist_version = wheel_version if sdist_version is None else sdist_version
        with zipfile.ZipFile(tmp_path / "powergenome.whl", "w") as wheel:
            wheel.writestr(
                "powergenome.dist-info/METADATA",
                f"Name: {name}\nVersion: {wheel_version}\n",
            )
        metadata = f"Name: {name}\nVersion: {sdist_version}\n".encode()
        with tarfile.open(tmp_path / "powergenome.tar.gz", "w:gz") as sdist:
            info = tarfile.TarInfo("powergenome/PKG-INFO")
            info.size = len(metadata)
            sdist.addfile(info, io.BytesIO(metadata))
        return tmp_path

    return make


@pytest.mark.parametrize(
    "version,tag,prerelease",
    [
        ("1.2.3", "v1.2.3", False),
        ("1.2.3rc1", "v1.2.3-rc.1", True),
        ("1.2.3.dev1", "v1.2.3.dev1", True),
        ("1.2.3.post1", "v1.2.3.post1", False),
    ],
)
def test_release_version_matches_normalized_tag(
    distributions, version, tag, prerelease
):
    result = release.check_release(distributions(version), tag)
    assert result == Version(version)
    assert result.is_prerelease == prerelease


@pytest.mark.parametrize("tag", ["v1.2.4", "main", "1.2.3"])
def test_release_rejects_wrong_tag(distributions, tag):
    with pytest.raises(ValueError, match="does not match package version"):
        release.check_release(distributions(), tag)


def test_release_rejects_inconsistent_artifacts(distributions):
    with pytest.raises(ValueError, match="versions do not match"):
        release.check_release(distributions("1.2.3", "1.2.4"), "v1.2.3")


def test_release_rejects_other_package(distributions):
    with pytest.raises(ValueError, match="PowerGenome package"):
        release.check_release(distributions(name="another-package"), "v1.2.3")


def test_release_requires_both_artifacts(distributions):
    path = distributions()
    (path / "powergenome.whl").unlink()
    with pytest.raises(ValueError, match="exactly one wheel and one source"):
        release.check_release(path, "v1.2.3")


def test_pull_request_validates_artifacts_without_tag(distributions):
    assert release.check_release(distributions()) == Version("1.2.3")
