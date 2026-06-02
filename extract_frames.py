import cv2
import os

os.makedirs("debug_frames", exist_ok=True)

videos = [
    ("data/cctv/CAM 1.mp4", "debug_frames/cam1.jpg"),
    ("data/cctv/CAM 2.mp4", "debug_frames/cam2.jpg"),
    ("data/cctv/CAM 3.mp4", "debug_frames/cam3.jpg"),
    ("data/cctv/CAM 4.mp4", "debug_frames/cam4.jpg"),
    ("data/cctv/CAM 5.mp4", "debug_frames/cam5.jpg"),
]

for video_path, output_path in videos:
    cap = cv2.VideoCapture(video_path)

    frame_no = 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")
    else:
        print(f"Failed: {video_path}")

    cap.release()

print("Done")