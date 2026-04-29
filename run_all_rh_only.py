#!/usr/bin/env python3
"""
Right-hand-only batch runner: process every MusicXML file in inputs/ and save
fingering plans to output_rh_only/ with all notes assigned to the right hand.

Usage:
    python run_all_rh_only.py [options]

Any extra options are forwarded to findOptimalHandPos.py.
"""

import os
import subprocess
import sys

INPUTS_DIR = "inputs"
OUTPUTS_DIR = "output_rh_only"
SCRIPT = "findOptimalHandPos.py"


def main():
    xml_files = sorted(
        f for f in os.listdir(INPUTS_DIR)
        if f.lower().endswith((".musicxml", ".xml"))
    )

    if not xml_files:
        print(f"No .musicxml files found in {INPUTS_DIR}/")
        sys.exit(1)

    print(f"Found {len(xml_files)} file(s) to process (right-hand-only mode):\n")
    for f in xml_files:
        print(f"  • {f}")
    print()

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
            "--rh-only",
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
