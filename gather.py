from pathlib import Path
dir_path = Path("workspace")

gen_file = dir_path / "gen.txt"

file_list = [
    "init_tasklist.jsonl",
    "onestep_perception.jsonl",
    "onestep.jsonl",
]

with open(gen_file, "r", encoding="utf-8") as f:
    for line in f:
        gen_info = eval(line)
        for file_name in file_list:
            file_path = dir_path / gen_info["dir"] / file_name
            with open(file_path, "r", encoding="utf-8") as f_in, open(dir_path / file_name, "a", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(line)
        