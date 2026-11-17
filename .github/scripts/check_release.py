"""Check built package versions without importing PowerGenome."""

import os
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


def check_release(dist_dir, tag=""):
    wheels = list(Path(dist_dir).glob("*.whl"))
    sdists = list(Path(dist_dir).glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")

    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_paths = [
            p for p in wheel.namelist() if p.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        wheel_metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))

    with tarfile.open(sdists[0]) as sdist:
        metadata_paths = [
            p
            for p in sdist.getnames()
            if len(Path(p).parts) == 2 and p.endswith("/PKG-INFO")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one source distribution PKG-INFO file")
        with sdist.extractfile(metadata_paths[0]) as metadata_file:
            sdist_metadata = BytesParser().parsebytes(metadata_file.read())

    for metadata in (wheel_metadata, sdist_metadata):
        if canonicalize_name(metadata["Name"]) != "powergenome":
            raise ValueError("Distribution must contain the PowerGenome package")
    version = Version(wheel_metadata["Version"])
    if version != Version(sdist_metadata["Version"]):
        raise ValueError("Wheel and source distribution versions do not match")
    if tag and (not tag.startswith("v") or Version(tag[1:]) != version):
        raise ValueError(f"Tag {tag!r} does not match package version {version}")
    return version


if __name__ == "__main__":
    tag = os.environ.get("RELEASE_TAG", "")
    version = check_release(sys.argv[1], tag)
    print(f"Validated PowerGenome {version} (tag: {tag or 'PR validation'})")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"tag={tag}\nprerelease={str(version.is_prerelease).lower()}\n")
