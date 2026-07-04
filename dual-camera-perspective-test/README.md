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

Press `q` or `Esc` to quit. Press `r` to reset the locked perspective and
recalibrate from the live feeds.

This is feature-based perspective matching. It works best when both cameras see
the same textured object or surface. Plain objects, motion blur, reflections,
and very different viewpoints can reduce or prevent matches.

Stability behavior:

- symmetric ORB feature matching filters one-way false matches,
- RANSAC rejects outlier point pairs,
- homography estimates must pass inlier-ratio and geometry sanity checks,
- the accepted perspective is smoothed over time,
- after several stable frames, fixed inlier anchor lines are locked,
- locked anchor lines stay in the same places instead of jumping every frame.

Useful tuning flags:

```powershell
py -3 dual_camera_perspective.py --min-matches 20 --min-inlier-ratio 0.4 --smoothing 0.88
```

If the scene is static but the points jump, wait for `H=LOCKED`. Before lock,
the display is still showing live acquisition matches. After lock, only the
frozen anchor correspondences are drawn.
