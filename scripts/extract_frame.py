import cv2

def extract_frames():
    cap = cv2.VideoCapture("assets/annotated_pipeline_run.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")

    # Baseline frame (e.g., frame 20)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ret, frame1 = cap.read()
    if ret:
        cv2.imwrite("assets/pipeline_run_frame.jpg", frame1)
        print("Wrote assets/pipeline_run_frame.jpg")

    # Intrusion frame (e.g., frame 150 or middle)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(150, total_frames - 10))
    ret, frame2 = cap.read()
    if ret:
        cv2.imwrite("assets/pipeline_intrusion_frame.jpg", frame2)
        print("Wrote assets/pipeline_intrusion_frame.jpg")

    cap.release()

if __name__ == "__main__":
    extract_frames()
