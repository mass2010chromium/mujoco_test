import json
import sys

fname = sys.argv[1]

with open(fname, 'r') as infile:
    data = json.load(infile)

i = 0
while True:
    if str(i) not in data:
        break
    episode = data[str(i)]
    for segment in episode['segments']:
        if segment['target'] is None:
            segment['target'] = {
                k: "NULL" for k in ['appearance', 'id', 'location', 'type']
            }
        if segment['target'].get('image_point') is None:
            segment['target']['image_point'] = [0, 0]
            segment['target']['location'] = 'NULL'
    i += 1

with open(fname, 'w') as outfile:
    json.dump(data, outfile, indent=4)
