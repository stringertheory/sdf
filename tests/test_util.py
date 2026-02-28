import math
from sdf.util import pi, degrees, radians


class TestConstants:
    def test_pi(self):
        assert pi == math.pi

    def test_degrees(self):
        assert degrees(math.pi) == 180.0

    def test_radians(self):
        assert radians(180) == math.pi
