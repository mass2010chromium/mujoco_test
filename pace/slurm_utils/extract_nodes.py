import sys

with open('sinfo.log', 'r') as infile:
    # CPUS, MEMORY, GRES, NODES(A/I/O/T), PARTITION, NODELIST
    lines = [[x.strip() for x in l.split()] for l in infile.readlines()]

avail_gpus = set()
gpu_to_nodes = dict()

for line in lines:
    gpu_info = line[2]
    if ':' not in gpu_info:
        continue
    gpu_name = gpu_info.split(':')[1]
    avail_gpus.add(gpu_name)

    if gpu_name not in gpu_to_nodes:
        gpu_to_nodes[gpu_name] = []
    gpu_to_nodes[gpu_name].append(line[-1])

avail_gpus = list(avail_gpus)
print("Available GPUS:", avail_gpus)
for gpu in avail_gpus:
    all_nodes = ','.join(gpu_to_nodes[gpu])
    with open(f"../nodes_{gpu}.txt", 'w') as out_file:
        print(all_nodes, file=out_file)
