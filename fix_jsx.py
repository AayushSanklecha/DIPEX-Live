import re

path = '/app/src/pages/RunPipeline.jsx'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix broken line: useState(false) missing semicolon + stray );
fixed = re.sub(
    r'(const \[optionalOpen, setOptionalOpen\] = useState\(false\))\s*\n[\s\S]*?\);',
    r'\1;',
    content,
    count=1
)

if content != fixed:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(fixed)
    print('FIXED: Repaired broken optionalOpen useState line')
else:
    # Show context to debug
    lines = content.split('\n')
    for i, line in enumerate(lines[410:420], start=411):
        print(f'{i}: {repr(line)}')
    print('Pattern not found - showing raw lines above')
