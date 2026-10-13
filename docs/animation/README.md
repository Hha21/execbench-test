# solx round 9 animation

HTML motion graphics (`index.html?t=SECONDS` draws one frame; numbers come from `problems/035_*/rounds/r9` and `poc/results/b200_portal.csv`).

Render (needs google-chrome, ffmpeg):

    WORK=/tmp/solx-anim ./render.sh

Writes `docs/media/solx-round9.gif` (960x540, 14 fps, 15 s). Frames stay in `$WORK`, outside the repo.
