"""Tests for sdf.viewer."""
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sdf


SAMPLE_POINTS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0],
    [1, 0, 0], [1, 1, 0], [0, 1, 0],
], dtype='float64')


class TestMeshViewer:
    """Test the interactive anywidget-based viewer."""

    def test_show_returns_widget(self):
        from sdf import viewer
        widget = viewer.show(SAMPLE_POINTS)
        assert hasattr(widget, 'vertices_flat')
        assert hasattr(widget, 'faces_flat')

    def test_show_with_sdf(self):
        s = sdf.sphere(1)
        widget = s.show(step=0.5)
        assert len(widget.vertices_flat) > 0
        assert len(widget.faces_flat) > 0

    def test_show_default_samples(self):
        """show() should use SHOW_SAMPLES (2**18) by default."""
        from sdf import viewer
        s = sdf.sphere(1)
        with patch.object(type(s), 'generate', return_value=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype='float64')) as mock_gen:
            viewer.show(s)
            kwargs = mock_gen.call_args[1]
            assert kwargs['samples'] == 2 ** 18

    def test_show_custom_samples_override(self):
        """User-provided samples should override the default."""
        from sdf import viewer
        s = sdf.sphere(1)
        with patch.object(type(s), 'generate', return_value=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype='float64')) as mock_gen:
            viewer.show(s, samples=2 ** 20)
            kwargs = mock_gen.call_args[1]
            assert kwargs['samples'] == 2 ** 20

    def test_show_empty_mesh(self, capsys):
        from sdf import viewer
        result = viewer.show(np.empty((0, 3)))
        assert result is None
        captured = capsys.readouterr()
        assert 'No triangles' in captured.out

    def test_show_deduplicates_vertices(self):
        from sdf import viewer
        widget = viewer.show(SAMPLE_POINTS)
        # 6 input vertices, 4 unique
        assert len(widget.vertices_flat) == 4 * 3

    def test_sdf3_show_method_exists(self):
        s = sdf.sphere(1)
        assert hasattr(s, 'show')
        assert callable(s.show)


class TestFallback:
    """Test fallback to pyvista when anywidget is not available."""

    def test_fallback_to_pyvista(self):
        """When anywidget is missing, should fall back to pyvista screenshot."""
        mock_pv = MagicMock()
        mock_mesh = MagicMock()
        mock_plotter = MagicMock()
        mock_plotter.screenshot.return_value = np.zeros((100, 100, 3), dtype='uint8')
        mock_pv.PolyData.return_value = mock_mesh
        mock_pv.Plotter.return_value = mock_plotter

        with patch.dict(sys.modules, {'anywidget': None, 'pyvista': mock_pv}):
            from sdf import viewer
            result = viewer.show(SAMPLE_POINTS)
            assert isinstance(result, np.ndarray)
            mock_pv.Plotter.assert_called_once()

    def test_no_backends_gives_helpful_error(self):
        """When both anywidget and pyvista are missing, show helpful error."""
        with patch.dict(sys.modules, {'anywidget': None, 'pyvista': None}):
            from sdf import viewer
            with pytest.raises(ImportError, match="anywidget"):
                viewer.show(SAMPLE_POINTS)
