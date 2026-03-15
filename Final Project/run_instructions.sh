
#!/bin/bash

echo "Running classification pipeline..."
python phase2_rigorous_cv_pipeline_ready.py

echo "Running YOLO detection pipeline..."
python phase2_yolo_detection_pipeline_ready_v4.py

echo "Pipelines completed."
