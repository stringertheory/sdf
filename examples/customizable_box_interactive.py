import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from sdf import Z, rounded_box, slab
    from sdf.glsl import ShaderViewer

    return (ShaderViewer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Customizable Box Designer

    Adjust the sliders to customize the box and lid dimensions.
    Drag to rotate, scroll to zoom, right-click to pan.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    width = mo.ui.slider(4, 20, value=12, step=0.5, label="Width")
    height = mo.ui.slider(2, 12, value=6, step=0.5, label="Height")
    depth = mo.ui.slider(0.5, 6, value=2, step=0.25, label="Depth")
    rows = mo.ui.slider(1, 6, value=3, step=1, label="Rows")
    cols = mo.ui.slider(1, 8, value=5, step=1, label="Columns")

    wall_thickness = mo.ui.slider(0.1, 0.5, value=0.25, step=0.05, label="Wall Thickness")
    wall_radius = mo.ui.slider(0.1, 1.0, value=0.5, step=0.05, label="Wall Radius")
    bottom_radius = mo.ui.slider(0.05, 0.5, value=0.25, step=0.05, label="Bottom Radius")
    top_fillet = mo.ui.slider(0.025, 0.5, value=0.125, step=0.025, label="Top Fillet")

    divider_thickness = mo.ui.slider(0.1, 0.4, value=0.2, step=0.05, label="Divider Thickness")
    row_divider_depth = mo.ui.slider(0.5, 4.0, value=1.75, step=0.25, label="Row Divider Depth")
    col_divider_depth = mo.ui.slider(0.5, 4.0, value=1.5, step=0.25, label="Col Divider Depth")
    divider_fillet = mo.ui.slider(0.05, 0.3, value=0.1, step=0.025, label="Divider Fillet")

    lid_thickness = mo.ui.slider(0.1, 0.5, value=0.25, step=0.05, label="Lid Thickness")
    lid_depth = mo.ui.slider(0.25, 1.5, value=0.75, step=0.05, label="Lid Depth")
    lid_radius = mo.ui.slider(0.025, 0.5, value=0.125, step=0.025, label="Lid Radius")

    show_lid = mo.ui.switch(value=False, label="Show Lid")

    mo.vstack([
        mo.md("### Box Dimensions"),
        mo.hstack([width, height, depth, rows, cols], justify="start"),
        mo.md("### Walls"),
        mo.hstack([wall_thickness, wall_radius, bottom_radius, top_fillet], justify="start"),
        mo.md("### Dividers"),
        mo.hstack([divider_thickness, row_divider_depth, col_divider_depth, divider_fillet], justify="start"),
        mo.md("### Lid"),
        mo.hstack([lid_thickness, lid_depth, lid_radius, show_lid], justify="start"),
    ])
    return (
        bottom_radius,
        col_divider_depth,
        cols,
        depth,
        divider_fillet,
        divider_thickness,
        height,
        lid_depth,
        lid_radius,
        lid_thickness,
        row_divider_depth,
        rows,
        show_lid,
        top_fillet,
        wall_radius,
        wall_thickness,
        width,
    )


@app.cell
def _(ShaderViewer):
    viewer = ShaderViewer({
        "width": 12.0,
        "height": 6.0,
        "depth": 2.0,
        "rows": 3.0,
        "cols": 5.0,
        "wall_thickness": 0.25,
        "wall_radius": 0.5,
        "bottom_radius": 0.25,
        "top_fillet": 0.125,
        "divider_thickness": 0.2,
        "row_divider_depth": 1.75,
        "col_divider_depth": 1.5,
        "divider_fillet": 0.1,
        "lid_thickness": 0.25,
        "lid_depth": 0.75,
        "lid_radius": 0.125,
        "show_lid": 0.0,
    })
    viewer
    return (viewer,)


@app.cell
def _(
    bottom_radius,
    col_divider_depth,
    cols,
    depth,
    divider_fillet,
    divider_thickness,
    height,
    lid_depth,
    lid_radius,
    lid_thickness,
    row_divider_depth,
    rows,
    show_lid,
    top_fillet,
    viewer,
    wall_radius,
    wall_thickness,
    width,
):
    viewer.uniforms = {
        "width": float(width.value),
        "height": float(height.value),
        "depth": float(depth.value),
        "rows": float(rows.value),
        "cols": float(cols.value),
        "wall_thickness": float(wall_thickness.value),
        "wall_radius": float(wall_radius.value),
        "bottom_radius": float(bottom_radius.value),
        "top_fillet": float(top_fillet.value),
        "divider_thickness": float(divider_thickness.value),
        "row_divider_depth": float(row_divider_depth.value),
        "col_divider_depth": float(col_divider_depth.value),
        "divider_fillet": float(divider_fillet.value),
        "lid_thickness": float(lid_thickness.value),
        "lid_depth": float(lid_depth.value),
        "lid_radius": float(lid_radius.value),
        "show_lid": 1.0 if show_lid.value else 0.0,
    }
    return


if __name__ == "__main__":
    app.run()
