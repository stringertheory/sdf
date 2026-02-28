"""Tests for sdf.viewer with mocked pyvista."""
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sdf


@pytest.fixture
def mock_pyvista():
    """Mock pyvista module so tests don't need a display."""
    mock_pv = MagicMock()
    mock_mesh = MagicMock()
    mock_plotter = MagicMock()
    mock_plotter.screenshot.return_value = np.zeros((100, 100, 3), dtype='uint8')
    mock_pv.PolyData.return_value = mock_mesh
    mock_pv.Plotter.return_value = mock_plotter
    with patch.dict(sys.modules, {'pyvista': mock_pv}):
        yield mock_pv, mock_mesh, mock_plotter


class TestShow:
    def test_show_with_sdf(self, mock_pyvista):
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        s = sdf.sphere(1)
        result = viewer.show(s, step=0.5)
        mock_pv.PolyData.assert_called_once()
        mock_plotter.add_mesh.assert_called_once()
        mock_plotter.screenshot.assert_called_once()
        assert result is not None

    def test_show_default_samples(self, mock_pyvista):
        """show() should use SHOW_SAMPLES (2**18) by default, not generate's 2**22."""
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        s = sdf.sphere(1)
        with patch.object(type(s), 'generate', return_value=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype='float64')) as mock_gen:
            viewer.show(s)
            kwargs = mock_gen.call_args[1]
            assert kwargs['samples'] == 2 ** 18

    def test_show_custom_samples_override(self, mock_pyvista):
        """User-provided samples should override the default."""
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        s = sdf.sphere(1)
        with patch.object(type(s), 'generate', return_value=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype='float64')) as mock_gen:
            viewer.show(s, samples=2 ** 20)
            kwargs = mock_gen.call_args[1]
            assert kwargs['samples'] == 2 ** 20

    def test_show_with_points(self, mock_pyvista):
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype='float64')
        viewer.show(points)
        mock_pv.PolyData.assert_called_once()
        args = mock_pv.PolyData.call_args
        verts = args[0][0]
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        mock_plotter.screenshot.assert_called_once()

    def test_show_empty_mesh(self, mock_pyvista, capsys):
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        points = np.empty((0, 3))
        result = viewer.show(points)
        mock_pv.PolyData.assert_not_called()
        assert result is None
        captured = capsys.readouterr()
        assert 'No triangles' in captured.out

    def test_sdf3_show_method_exists(self):
        s = sdf.sphere(1)
        assert hasattr(s, 'show')
        assert callable(s.show)

    def test_show_without_pyvista_gives_helpful_error(self):
        """When pyvista is not installed, show() should raise ImportError with install hint."""
        with patch.dict(sys.modules, {'pyvista': None}):
            from sdf import viewer
            with pytest.raises(ImportError, match="pip install pyvista"):
                viewer.show(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='float64'))

    def test_show_deduplicates_vertices(self, mock_pyvista):
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        # Two triangles sharing an edge (4 unique verts, 6 total)
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype='float64')
        viewer.show(points)
        args = mock_pv.PolyData.call_args
        verts = args[0][0]
        assert len(verts) == 4  # deduplicated from 6 to 4

    def test_show_returns_image_array(self, mock_pyvista):
        mock_pv, mock_mesh, mock_plotter = mock_pyvista
        from sdf import viewer
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype='float64')
        result = viewer.show(points)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
