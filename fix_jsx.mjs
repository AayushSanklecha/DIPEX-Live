import { readFileSync, writeFileSync } from 'fs';

const path = '/app/src/pages/RunPipeline.jsx';
let content = readFileSync(path, 'utf-8');

// Show raw context
const idx = content.indexOf('optionalOpen, setOptionalOpen');
console.log('RAW:', JSON.stringify(content.slice(idx - 5, idx + 150)));

// Fix broken: useState(false) followed by whitespace/newlines then );
const fixed = content.replace(
    /const \[optionalOpen, setOptionalOpen\] = useState\(false\)[\s\S]{1,80}?\);/,
    'const [optionalOpen, setOptionalOpen] = useState(false);'
);

if (fixed !== content) {
    writeFileSync(path, fixed, 'utf-8');
    console.log('SUCCESS: Fixed corrupted optionalOpen useState line');
} else {
    console.log('No change applied.');
}
