# Dual Arm Camera Perspective Test

This test opens the left and right robo arm USB cameras by their unique
DirectShow device paths, matches visual features automatically, estimates a
left-to-right homography with RANSAC, and draws real-time match lines between
the two camera views.

Run:

```powershell
py -3 dual_camera_perspective.py
```

The top half of the window shows both feeds side by side with lines between
matched points. The bottom half shows the right feed next to a blended
right-feed + warped-left-feed view when a homography is available.

Press `q` or `Esc` to quit.

This is feature-based perspective matching. It works best when both cameras see
the same textured object or surface. Plain objects, motion blur, reflections,
and very different viewpoints can reduce or prevent matches.

Stability behavior:

- symmetric ORB feature matching filters one-way false matches,
- RANSAC rejects outlier point pairs,
- homography estimates must pass inlier-ratio and geometry sanity checks,
- the accepted perspective is smoothed over time,
- the last good perspective is held briefly if a frame has weak matches.

Useful tuning flags:

```powershell
py -3 dual_camera_perspective.py --min-matches 20 --min-inlier-ratio 0.4 --smoothing 0.88
```
