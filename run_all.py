#!/usr/bin/env python3
"""
Batch runner: process every MusicXML file in the inputs/ folder and save
outputs to outputs/ with the song name as a prefix on each file.

Usage:
    python run_all.py [options]

Any extra options are forwarded directly to findOptimalHandPos.py, e.g.:
    python run_all.py --speed 15 --gap 8
"""

import os
import subprocess
import sys

INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
SCRIPT = "findOptimalHandPos.py"


def main():
    xml_files = sorted(
        f for f in os.listdir(INPUTS_DIR)
        if f.lower().endswith((".musicxml", ".xml"))
    )

    if not xml_files:
        print(f"No .musicxml files found in {INPUTS_DIR}/")
        sys.exit(1)

    print(f"Found {len(xml_files)} file(s) to process:\n")
    for f in xml_files:
        print(f"  • {f}")
    print()

    # Extra args to forward (everything passed to this script)
    extra_args = sys.argv[1:]

    failed = []
    for xml_file in xml_files:
        song_name = os.path.splitext(xml_file)[0]
        input_path = os.path.join(INPUTS_DIR, xml_file)

        print("=" * 60)
        print(f"Processing: {xml_file}  →  prefix '{song_name}'")
        print("=" * 60)

        cmd = [
            sys.executable, SCRIPT,
            input_path,
            "--output", OUTPUTS_DIR,
            "--prefix", song_name,
        ] + extra_args

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n❌ Failed: {xml_file}\n")
            failed.append(xml_file)
        else:
            print(f"\n✓ Done: {xml_file}\n")

    print("=" * 60)
    if failed:
        print(f"Completed with {len(failed)} failure(s):")
        for f in failed:
            print(f"  ❌ {f}")
    else:
        print(f"All {len(xml_files)} file(s) processed successfully.")
    print(f"Outputs saved to: {OUTPUTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
