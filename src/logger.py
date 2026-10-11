from pathlib import Path
saved_log = []
log_file = Path(__file__).resolve().parent.parent / "log" / "log.txt"

def log(line):
    line = str(line)
    print(line)
    saved_log.append(line)

def set_log_file(filename = ""):
    global log_file 
    log_file = Path(__file__).resolve().parent.parent / "log" / f"{filename}.txt"

def save_logs():
    log_file.parent.mkdir(parents = True, exist_ok = True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(saved_log) + "\n")