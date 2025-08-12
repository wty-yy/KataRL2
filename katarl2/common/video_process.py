import os
import subprocess
from pathlib import Path

in_mp4 = "/data/user/wutianyang/Coding/KataRL2/logs/debug/sac/basic_continuous_mlp/Hopper-v4__gymnasium/seed_42_42/20250811-163248/videos/rl-video-episode-0.mp4"
out_gif = "/data/user/wutianyang/Coding/KataRL2/logs/debug/sac/basic_continuous_mlp/Hopper-v4__gymnasium/seed_42_42/20250811-163248/videos/rl-video-episode-0.gif"

def cvt_to_gif(in_path, out_path=None) -> Path:
    if out_path is None:
        out_path = Path(in_path).parent / (Path(in_path).stem + '.gif')

    fps = 25
    width = 320
    cmd = (
        f'ffmpeg -y -i {str(in_path)} '
        f'-vf "fps={fps},scale={width}:-1:flags=lanczos,'
        f'split[a][b];[a]palettegen=stats_mode=full:max_colors=32[p];'
        f'[b][p]paletteuse=dither=none:diff_mode=rectangle" '
        f'-loop 0 {str(out_path)}'
    )

    subprocess.run(cmd, shell=True, check=True)
    in_size = os.path.getsize(in_path) / 1024
    out_size = os.path.getsize(out_path) / 1024
    print(f"Convert mp4 to gif: {in_size:.2f}Kb -> {out_size:.2f}Kb from {in_path} to {out_path}")
    return Path(out_path)

if __name__ == '__main__':
    cvt_to_gif("/data/user/wutianyang/Coding/KataRL2/logs/debug/sac/basic_continuous_mlp/Hopper-v4__gymnasium/seed_42_42/20250811-163248/videos/rl-video-episode-0.mp4")
