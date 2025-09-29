# src/week7/run_all.py
import argparse, subprocess, sys, re, time
from pathlib import Path

def extract_commands(md_path: Path):
    text = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cmds = []
    in_code = False
    for ln in text:
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        line = ln.strip()
        if line.startswith("python "):
            cmds.append(line)
        elif in_code and line.startswith("python "):
            cmds.append(line)
    return cmds

def run(cmd: str, logf: Path) -> int:
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for ch in p.stdout:
            sys.stdout.write(ch)
            logf.write_text(logf.read_text() + ch if logf.exists() else ch)
        return p.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="src/week7/commands.md")
    ap.add_argument("--log", default="reports/week7/run.log")
    ap.add_argument("--start", type=int, default=0, help="start index (0-based)")
    ap.add_argument("--continue_on_error", type=int, default=0)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    md = Path(args.file)
    cmds = extract_commands(md)
    if not cmds:
        print(f"No commands found in {md}")
        sys.exit(1)

    logp = Path(args.log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    logp.write_text(f"# Run started {time.strftime('%Y-%m-%d %H:%M:%S')}\nFile: {md}\nTotal cmds: {len(cmds)}\nStart: {args.start}\n\n")

    for i, cmd in enumerate(cmds[args.start:], start=args.start):
        hdr = f"\n\n=== [{i+1}/{len(cmds)}] {cmd} ===\n"
        print(hdr.strip())
        logp.write_text(logp.read_text() + hdr)
        if args.dry_run:
            print("DRY RUN — skipped.")
            logp.write_text(logp.read_text() + "DRY RUN — skipped.\n")
            continue
        code = subprocess.call(cmd, shell=True)
        logp.write_text(logp.read_text() + f"\nEXIT {code}\n")
        if code != 0 and not args.continue_on_error:
            print(f"Stopped on error (exit {code}).")
            sys.exit(code)

if __name__ == "__main__":
    main()
