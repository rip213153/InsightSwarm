# InsightSwarm 15s Promo

Self-contained 15-second promo composition for InsightSwarm.

## Files

- `index.html` - 1920x1080 responsive HTML composition with CSS/JS timeline.

## Preview

Open `index.html` in a browser. The animation is timed to 15 seconds and includes a progress bar/timecode.

## Render

If HyperFrames CLI is available, use this HTML file as the composition entry and render a 15-second 16:9 video. Keep outputs inside this folder, for example:

```powershell
hyperframes render .\promo\insightswarm-15s\index.html --duration 15 --output .\promo\insightswarm-15s\insightswarm-15s.mp4
```

If using browser capture instead, record the page at 1920x1080 for 15 seconds.
