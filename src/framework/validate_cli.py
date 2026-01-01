from __future__ import annotations

import os
import sys
import argparse

from .validators import validate_output_structure, validate_links


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate forecast outputs: structure (split mode) and link integrity in calibration.md."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to the output directory. Defaults to <this file>/../output",
    )
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default output dir: src/output (sibling of this file's directory)
    default_output = os.path.normpath(os.path.join(script_dir, "..", "output"))
    output_dir = args.output_dir or default_output

    if not os.path.isdir(output_dir):
        print(f"FAIL: Output directory not found: {output_dir}")
        return 2

    print(f"Validating output structure in: {output_dir}")
    ok_struct, msgs_struct = validate_output_structure(output_dir)
    if ok_struct:
        print("PASS: Expected split-mode files and assets present")
    else:
        print("FAIL: Output structure issues detected:")
        for m in msgs_struct:
            print(f" - {m}")

    # Validate calibration.md links
    cal_md = os.path.join(output_dir, "calibration.md")
    ok_links, msgs_links = validate_links(cal_md, base_dir=output_dir)
    if ok_links:
        print("PASS: All links in calibration.md resolve")
    else:
        print("FAIL: Broken links found in calibration.md:")
        for m in msgs_links:
            print(f" - {m}")

    overall_ok = ok_struct and ok_links
    print("\nRESULT:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
