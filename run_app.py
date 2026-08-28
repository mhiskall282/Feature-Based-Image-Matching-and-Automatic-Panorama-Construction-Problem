"""
run_app.py
==========
Root launcher for the CSCD608 Interactive Panorama Web Dashboard.

Usage:
    python run_app.py
    python run_app.py --port 5000 --host 127.0.0.1
"""

import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.server import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch CSCD608 Panorama Web App")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CSCD608 ADVANCED COMPUTER VISION - PANORAMA WEB APPLICATION")
    print("="*70)
    print(f"  • Main Dashboard:   http://{args.host}:{args.port}/")
    print(f"  • Documentation:    http://{args.host}:{args.port}/docs")
    print(f"  • API Endpoints:    http://{args.host}:{args.port}/api/presets")
    print("="*70 + "\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
