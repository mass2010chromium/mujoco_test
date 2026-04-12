import json
import math

with open("targets_log.txt", "r") as errors_file:
    data = [int(s.strip().split(maxsplit=1)[0][:-1]) for s in errors_file.readlines()]
    missing = list(set(data))

splits = 4
split_size = math.ceil(len(missing) / splits)
missing_splits = [missing[i:i+split_size] for i in range(0, len(missing), split_size)]

with open('missing_splits.json', 'w') as outfile:
    json.dump(missing_splits, outfile)
