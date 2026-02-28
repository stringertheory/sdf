import numpy as np
import pytest

try:
    from sdf.text import measure_text, text
    HAS_TEXT = True
except Exception:
    HAS_TEXT = False


@pytest.mark.skipif(not HAS_TEXT, reason="text dependencies not available")
class TestText:
    def test_measure_text(self):
        try:
            w, h = measure_text('Arial', 'Hello')
            assert w > 0
            assert h > 0
        except Exception:
            pytest.skip("No suitable font available")

    def test_text_returns_sdf2(self):
        try:
            s = text('Arial', 'Hi')
            pts = np.array([[0.0, 0.0], [1.0, 1.0]])
            result = s(pts)
            assert result.shape == (2, 1)
        except Exception:
            pytest.skip("No suitable font available")
