from pathlib import Path
file_dir = Path(r"C:\Users\DP\Downloads\AppEvalPilot-main\AppEvalPilot-main\workspace\202506232013\gen")
max_filename_length = max([len(file.stem) for file in file_dir.glob("*.json")])
file_path_list = [file for file in file_dir.glob("*.json") if len(file.stem) == max_filename_length]

# 获取已完成的文件名列表（不包含扩展名）
# 假设gen_finished_file_stem_list来自gen目录，而file_path_list来自state目录
gen_dir = file_dir  # 这里应该是gen目录
state_dir = file_dir.parent  # 这里应该是state目录

# 从gen目录获取已完成的文件名
gen_finished_file_stem_list = [file.stem for file in gen_dir.glob("*.json") if len(file.stem) == max_filename_length]

# 从state目录获取需要处理的文件，并过滤掉已完成的
file_path_list = [file for file in state_dir.glob("*.json") if len(file.stem) == max_filename_length]
file_path_list = [file for file in file_path_list if file.stem not in gen_finished_file_stem_list]

print(len(file_path_list))
breakpoint()

import json
with open(file_dir / "0.json", encoding="utf-8") as f:
    data = json.load(f)
    print(len(data["gen_instruction"]))
    breakpoint()        