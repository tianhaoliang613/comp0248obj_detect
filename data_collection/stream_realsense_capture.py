import os, cv2, json
import numpy as np
import pyrealsense2 as rs

# ===================================================================

CLIP = 5
STUDENT_NO = "25100812"
SURNAME = "Liang"

# ==================== Configuration Parameters ====================
MAX_DISTANCE_M = 4       # Maximum distance of interest
FPS = 30                 # Camera frame rate: 30fps
SAVE_FPS = 3             # Save frame rate: 3fps (save 3 frames per second)
CLIP_DURATION = 5.0      # Recording duration: 5 seconds
NUM_SAVED = int(SAVE_FPS * CLIP_DURATION)

GESTURES = [
    ("G01_call", "Call"), 
    ("G02_dislike", "Dislike"), 
    ("G03_like", "Like"),
    ("G04_ok", "OK"), 
    ("G05_one", "One"), 
    ("G06_palm", "Palm"),
    ("G07_peace", "Peace"), 
    ("G08_rock", "Rock"), 
    ("G09_stop", "Stop"),
    ("G10_three", "Three")
]

def create_dirs(base_dir, gesture_id, clip_id):
    """Create gesture data directories"""
    clip_dir = os.path.join(base_dir, gesture_id, clip_id)
    paths = {
        "rgb": os.path.join(clip_dir, "rgb"),
        "depth": os.path.join(clip_dir, "depth"),
        "raw": os.path.join(clip_dir, "depth_raw"),
        "anno": os.path.join(clip_dir, "annotation"),
        "clip": clip_dir  # Add clip directory path
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def save_depth_scale(clip_dir, depth_scale):
    """Save depth_scale to JSON file"""
    metadata = {
        "depth_scale": float(depth_scale),
        "unit": "meters per depth unit",
        "description": "Multiply raw depth values by depth_scale to get depth in meters"
    }
    json_path = os.path.join(clip_dir, "depth_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"💾 Depth scale saved: {depth_scale:.6f} m/unit -> {json_path}")

def add_text_to_frame(frame, text, position, font_scale=1.0, color=(0, 0, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame

def visualize_frames(color_np, depth_img, status_text, timer_text, status_color, instruction_text=""):
    """Display RGB and depth images using OpenCV"""
    # Convert depth to visualization format & add text
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    add_text_to_frame(depth_img, "DEPTH", (30, 60))

    # Add text to RGB image
    color_display = color_np.copy()
    color_display = add_text_to_frame(color_display, status_text, (30, 60), color=status_color)
    color_display = add_text_to_frame(color_display, timer_text, (30, 120), color=status_color)
    color_display = add_text_to_frame(color_display, instruction_text, (30, 180))

    # Stack & Display
    combined = np.hstack([color_display, depth_img])
    cv2.imshow("RealSense D455 - Gesture Collection", combined)

def main(base_dir, clip_id):
    print("\n" + "="*70)
    print("🎥 RealSense D455 Gesture Data Collection System")
    print("="*70)
    print(f"Student Info: {base_dir}")
    print(f"Recording Clip: {clip_id}")
    print(f"Number of Gestures: {len(GESTURES)}")
    print(f"Duration per Gesture: {CLIP_DURATION} seconds")
    print(f"Camera Frame Rate: {FPS} fps")
    print(f"Save Frame Rate: {SAVE_FPS} fps (save {SAVE_FPS} frames per second)")
    print(f"Frames Saved per Gesture: {NUM_SAVED} frames")
    print("="*70)
    print("\nPress Enter to start...")
    input()

    # Initialize RealSense
    IN_W = 1280

    IN_H = 720
    W = 640
    H = 480
    LW = int((IN_W / 2) - (W / 2))
    UW = int(LW + W)
    LH = int((IN_H / 2) - (H /2))
    UH = int(LH + H)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IN_W, IN_H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, IN_W, IN_H, rs.format.rgb8, FPS)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    print(f"\n📏 Depth Scale: {depth_scale:.6f} meters/unit")

    for _ in range(5):
        pipeline.wait_for_frames()

    try:
        for idx, (gid, gname) in enumerate(GESTURES):
            paths = create_dirs(base_dir, gid, clip_id)
            
            # Save depth_scale once per clip
            save_depth_scale(paths["clip"], depth_scale)

            print(f"\n[{idx+1}/{len(GESTURES)}] Preparing to collect {NUM_SAVED} frames for: {clip_id} of {gname}")
            print("Press SPACE to start recording...")

            frame_idx = 0
            frames_seen = 0
            recording = False
            status_text = f"NEXT: {gname}"
            timer_text = "READY"
            instruction_text = "Press SPACE to start recording"
            text_colour = (0, 255, 0)

            while frame_idx < NUM_SAVED:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_np = np.asanyarray(color_frame.as_frame().get_data())
                depth_np_raw = np.asanyarray(depth_frame.as_frame().get_data())  # Keep as uint16

                color_np = color_np[LH:UH, LW:UW]
                depth_np_raw = depth_np_raw[LH:UH, LW:UW]

                color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                
                # For visualization: convert to meters
                depth_np_meters = depth_np_raw.astype(np.float32) * depth_scale
                depth_img = np.clip(depth_np_meters * 255 / MAX_DISTANCE_M, 0, 255).astype(np.uint8)

                if recording:
                    if (frames_seen % int(FPS / SAVE_FPS)) == 0:
                        frame_idx += 1
                        print(f"📹 Saved frame: {frame_idx}")
                        cv2.imwrite(os.path.join(paths["rgb"], f"frame_{frame_idx:03d}.png"), color_np)
                        cv2.imwrite(os.path.join(paths["depth"], f"frame_{frame_idx:03d}.png"), depth_img)
                        # Save raw depth as uint16 (no scaling)
                        np.save(os.path.join(paths["raw"], f"frame_{frame_idx:03d}.npy"), depth_np_raw)
                        timer_text = f"Saved {frame_idx}/{NUM_SAVED} frames"
                    frames_seen += 1
                
                visualize_frames(color_np, depth_img, status_text, timer_text, text_colour, instruction_text)

                # Check key press
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE key
                    print(f"🔴 Recording started: {gname}")
                    recording = True
                    status_text = f"RECORDING: {gname}"
                    timer_text = f"{frame_idx}/{NUM_SAVED} frames saved"
                    instruction_text = ""
                    text_colour = (0, 0, 255)
                elif key == 27:  # ESC key
                    raise KeyboardInterrupt

            print(f"✅ Completed: {clip_id} for {gname} - Saved {frame_idx} of {frames_seen} frames")
            print(f"Save path: {os.path.join(base_dir, gid, clip_id)}")

        # Completion
        print("\n" + "="*70)
        print("🎉 All gestures collection completed!")
        print("="*70)
        print(f"✅ Total gestures collected: {len(GESTURES)}")
        print(f"✅ Data saved to: {base_dir}")
        print("="*70)

        cv2.waitKey(2000)

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n📷 Camera closed")

if __name__ == "__main__":
    base_dir = os.path.join("dataset", f"{STUDENT_NO}_{SURNAME}")
    clip_id = f"clip{CLIP:02d}"

    main(base_dir, clip_id)