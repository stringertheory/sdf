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

    return Z, rounded_box, slab


@app.cell
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

    mo.vstack([
        mo.md("### Box Dimensions"),
        mo.hstack([width, height, depth, rows, cols], justify="start"),
        mo.md("### Walls"),
        mo.hstack([wall_thickness, wall_radius, bottom_radius, top_fillet], justify="start"),
        mo.md("### Dividers"),
        mo.hstack([divider_thickness, row_divider_depth, col_divider_depth, divider_fillet], justify="start"),
        mo.md("### Lid"),
        mo.hstack([lid_thickness, lid_depth, lid_radius], justify="start"),
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
        top_fillet,
        wall_radius,
        wall_thickness,
        width,
    )


@app.cell
def _(
    Z,
    bottom_radius,
    col_divider_depth,
    cols,
    depth,
    divider_fillet,
    divider_thickness,
    height,
    rounded_box,
    row_divider_depth,
    rows,
    slab,
    top_fillet,
    wall_radius,
    wall_thickness,
    width,
):
    _W = width.value
    _H = height.value
    _D = depth.value
    _R = int(rows.value)
    _C = int(cols.value)
    _WT = wall_thickness.value
    _WR = wall_radius.value
    _BR = bottom_radius.value
    _TF = top_fillet.value
    _DT = divider_thickness.value
    _RDD = row_divider_depth.value
    _CDD = col_divider_depth.value
    _DF = divider_fillet.value

    # Build dividers
    _col_spacing = _W / _C
    _row_spacing = _H / _R
    _c = rounded_box((_DT, 1e9, _CDD), _DF)
    _c = _c.translate(Z * _CDD / 2)
    _c = _c.repeat((_col_spacing, 0, 0))
    _r = rounded_box((1e9, _DT, _RDD), _DF)
    _r = _r.translate(Z * _RDD / 2)
    _r = _r.repeat((0, _row_spacing, 0))
    if _C % 2 != 0:
        _c = _c.translate((_col_spacing / 2, 0, 0))
    if _R % 2 != 0:
        _r = _r.translate((0, _row_spacing / 2, 0))
    _divs = _c | _r

    # Build box shell
    _p = _WT
    _f = rounded_box((_W - _p, _H - _p, 1e9), _WR)
    _f &= slab(z0=_p / 2).k(_BR)
    _divs &= _f
    _f = _f.shell(_WT)
    _f &= slab(z1=_D).k(_TF)
    _box = _f | _divs

    _box.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Lid
    """)
    return


@app.cell
def _(
    height,
    lid_depth,
    lid_radius,
    lid_thickness,
    rounded_box,
    slab,
    top_fillet,
    wall_radius,
    wall_thickness,
    width,
):
    _W = width.value
    _H = height.value
    _WT = wall_thickness.value
    _WR = wall_radius.value
    _TF = top_fillet.value
    _LT = lid_thickness.value
    _LD = lid_depth.value
    _LR = lid_radius.value

    _p = _WT
    _f = rounded_box((_W + _p, _H + _p, 1e9), _WR)
    _f &= slab(z0=_p / 2).k(_LR)
    _f = _f.shell(_LT)
    _f &= slab(z1=_LD).k(_TF)

    _f.show()
    return


if __name__ == "__main__":
    app.run()
