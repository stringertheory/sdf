import numpy as np
import struct
import tempfile
import os

from sdf.stl import write_binary_stl
from sdf.step import write_step


class TestWriteBinarySTL:
    def test_write_known_triangles(self):
        # Two triangles forming a quad
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype='float32')

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            path = f.name
        try:
            write_binary_stl(path, points)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            # 80 header + 4 count + 2 triangles * 50 bytes each
            assert size == 80 + 4 + 2 * 50

            with open(path, 'rb') as fp:
                fp.read(80)  # skip header
                count = struct.unpack('<I', fp.read(4))[0]
                assert count == 2
        finally:
            os.unlink(path)


class TestWriteStep:
    def test_write_known_triangles(self):
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype='float64')

        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
            path = f.name
        try:
            write_step(path, points)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            with open(path, 'r') as fp:
                content = fp.read()
            assert 'ISO-10303-21' in content
            assert 'CLOSED_SHELL' in content
            assert 'ADVANCED_FACE' in content
        finally:
            os.unlink(path)

    def test_empty_mesh(self):
        points = np.empty((0, 3))
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
            path = f.name
        try:
            write_step(path, points)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
