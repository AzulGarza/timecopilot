import pathlib

import pytest
from mktestdocs import check_md_file


@pytest.mark.live
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.live
def test_readme():
    check_md_file("README.md", memory=True)
