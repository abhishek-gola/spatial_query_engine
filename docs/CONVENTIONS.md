# Conventions

Everything in this repo obeys the following. If a number looks flipped, check here first.

## World frame

* Right-handed, metres.
* `+Z` is **up**, i.e. anti-gravity. ScanNet++ `mesh_aligned_0.05.ply` and the
  ARKitScenes 3DOD annotations already come gravity-aligned this way; loaders
  verify it and re-align if the floor normal disagrees by more than a degree or
  two, rather than trusting the dataset blindly.
* `X` and `Y` are *not* meaningful on their own. Any statement that depends on
  them ("the left wall") has to go through a reference frame.

## Camera frame

* OpenCV convention: `+x` right, `+y` down, `+z` forward along the optical axis.
* A `pose` is **camera-to-world** 4x4 unless the variable name says `w2c`.
  COLMAP stores world-to-camera, so the loaders invert it once, at the edge.

## Reference frame basis

A reference frame is an orthonormal triple stored as the columns of a 3x3 matrix

```
B = [ r | f | u ]      r = right, f = forward, u = up
```

right-handed with `r x f = u`. Given a world displacement `d`, its
frame-local coordinates are `B.T @ d`, so:

| local component | positive means |
|---|---|
| `x = d . r` | to the **right** |
| `y = d . f` | **forward** / further away from the frame's viewpoint |
| `z = d . u` | **above** |

So "A is left of B" is `x < 0` for `d = centre(A) - centre(B)`, and "in front of"
is the direction the frame's `f` points, which for a viewer frame means *towards*
the viewer, not away. That sign trap is handled in one place
(`sqe/relations/projective.py`) and nowhere else.

For every frame we build, `u` is world up. Frames never tilt out of the
gravity plane, even when the phone was tilted, with the single exception of
`camera_image`, which deliberately keeps camera roll because it models
"left in the picture".

## Object local frame

An object's oriented box is yaw-only: `u` is world up, and the horizontal axes
come from the minimum-area rectangle of its footprint. Local `x` is the **longer**
horizontal side. That leaves the usual 180-degree flip, which is *not* resolved by
the box fit; it is resolved separately and with a confidence value by
`sqe/perception/orientation.py`. Nothing downstream may assume local `+x` or
`+y` is the front.

## Units and thresholds

Distances in metres, angles in radians internally and degrees in configs and
reports. Every threshold with a physical meaning lives in
`configs/relations.yaml`, never inline in a function.
