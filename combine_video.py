from moviepy.editor import VideoFileClip, clips_array

# 動画ファイルのパスを指定
video_path1 = "./datasets/video/sp_env.mp4"
video_path2 = "./results/ap_env_400/Scene2ver2_400/test_latest/thermal_sp_env.mp4"

# 各動画を読み込み
clip1 = VideoFileClip(video_path1)
clip2 = VideoFileClip(video_path2)

# 動画の高さを合わせるため、短い方の高さに合わせてリサイズ
height = min(clip1.size[1], clip2.size[1])
clip1_resized = clip1.resize(height=height)
clip2_resized = clip2.resize(height=height)

# 横並びで動画を結合
final_clip = clips_array([[clip1_resized, clip2_resized]])

# 出力ファイルを保存
output_path = "combined_video.mp4"
final_clip.write_videofile(output_path, codec="libx264")
