"""
FinShield — main orchestrator
Run: python src/main.py
"""

from src import generate_data, pipeline, anomaly_detection, visualizations


def main():
    print("=== FinShield Pipeline ===\n")

    print("[1/4] Generating data...")
    generate_data.run()

    print("[2/4] Running ETL pipeline...")
    pipeline.run()

    print("[3/4] Running anomaly detection...")
    anomaly_detection.run()

    print("[4/4] Generating visualizations...")
    visualizations.run()

    print("\nDone. Outputs written to outputs/")


if __name__ == "__main__":
    main()
