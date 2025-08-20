# Traffic Vehicle Counting with YOLOv8 + BoT-SORT

This project counts vehicles in a traffic video using **YOLOv8 (small variant)** for object detection and **BoT-SORT** for object tracking.  
It also assigns vehicles to lanes and saves results in both a CSV file and a demo video with lane overlays.

---

## 🚀 Features
- Vehicle detection using **YOLOv8s**
- Robust tracking with **BoT-SORT**
- Lane-based counting (supports 4 lanes)
- Outputs:
  - `vehicle_counts.csv` → detailed per-vehicle data
  - `demo.mp4` → processed video with overlays
  - Console summary with lane counts & FPS

---

## 📂 Project Structure
├── traffic_counter.py # Main script
├── README.md # Documentation
├── requirements.txt # Dependencies
├── .gitignore # Ignore unnecessary files



---

## ⚙️ Installation

cd traffic-counter
2. Create & Activate Virtual Environment (recommended)

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac
3. Install Requirements

pip install -r requirements.txt
▶️ Usage
Run the script
python traffic_counter.py


Input & Output
Input video is defined in the script:
video_path = r"video_used_for_processing_the_vehicle_count.mp4"

Outputs:

vehicle_counts.csv

demo.mp4

📝 Example Output

✅ Processing video_used_for_processing_the_vehicle_count.mp4 with BoT-SORT...
✅ vehicle_counts.csv written with 23 total vehicles
✅ demo.mp4 saved
✅ Final lane counts: {1: 7, 2: 6, 3: 5, 4: 5}
✅ Average processing speed: 4.1 FPS
📌 Notes
Accuracy depends on video quality, lighting, and camera angle.

For better performance, consider using YOLOv8m/l models instead of YOLOv8s.

Lane positions may need to be adjusted depending on your video.

📽 Demo
You can generate a short demo video by trimming input and re-running the script.
