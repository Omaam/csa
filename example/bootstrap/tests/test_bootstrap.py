import pytest
import sys
sys.path.append('..')

from bootstrap import _get_savename

@pytest.fixture
def dire(tmpdir):
    tmpfile = tmpdir.join('dire')
    yield str(tmpfile)
    tmpfile.remove()


def test_getsavename_name_01():
    assert _get_savename('name', 'dire') == 'name_00.dat'
