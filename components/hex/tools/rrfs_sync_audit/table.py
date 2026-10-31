"""Print the driver API census; runtime rows remain in JSON as a cross-check."""
import json
import sys
from collections import Counter
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
counts = Counter()
for row in data['cuda']:
    if row['layer'] == 'driver' and row['step'] is not None:
        counts[row['step'], row['op'], row['category'], row['api'], row['blocking']] += row['count']
print('step | op | category | API | blocking | calls')
for key, count in sorted(counts.items()):
    print(' | '.join(map(str, (*key, count))))
print('callback errors:', data['errors'])
