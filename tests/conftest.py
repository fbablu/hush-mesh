import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "edge"))

@pytest.fixture
def project_root():
    """Return project root directory"""
    return Path(__file__).parent.parent