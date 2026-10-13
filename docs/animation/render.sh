#!/bin/bash
# usage: ./render.sh  -> writes ../media/solx-round9.gif (needs google-chrome, ffmpeg)
set -e
D=${WORK:-/tmp/solx-anim}; FPS=14; DUR=15; N=$((FPS*DUR)); H="$(cd "$(dirname "$0")" && pwd)"
rm -rf $D; mkdir -p $D
seq 0 $((N-1)) | xargs -P4 -I{} sh -c 't=$(python3 -c "print({}/'$FPS')"); google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars --window-size=960,540 --screenshot='$D'/f$(printf %04d {}).png "file://'$H'/index.html?t=$t" >/dev/null 2>&1'
ffmpeg -y -loglevel error -framerate $FPS -i $D/f%04d.png -vf "palettegen=max_colors=96:stats_mode=diff" $D/p.png
ffmpeg -y -loglevel error -framerate $FPS -i $D/f%04d.png -i $D/p.png -lavfi "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 "$H/../media/solx-round9.gif"
