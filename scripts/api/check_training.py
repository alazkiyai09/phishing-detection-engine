#!/usr/bin/env python3
"""Check training progress."""
import subprocess
import time

while True:
    try:
        # Check if process is still running
        result = subprocess.run(
            ["pgrep", "-f", "quick_train_distilbert.py"],
            capture_output=True
        )

        if result.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] DistilBERT training still running...")
            print("  Check full logs: tail -f /tmp/claude/-home-ubuntu-21Days-Project/tasks/*/output")
        else:
            print("\n✅ Training completed!")
            print("Check results:")
            print("  ls -lh /home/ubuntu/21Days_Project/models/day3_distilbert/")
            break

        time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        break
