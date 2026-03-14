const fs = require('fs');
const path = '/app/src/pages/RunPipeline.jsx';
let content = fs.readFileSync(path, 'utf-8');

// Show 200 chars around optionalOpen for debug
const idx = content.indexOf('optionalOpen, setOptionalOpen');
const chunk = content.slice(idx - 5, idx + 150);
console.log('BEFORE:', JSON.stringify(chunk));

// Fix: replace broken useState(false) [missing semicolon + stray );] with correct version
// The corruption looks like: useState(false)\n\n\n                );
const fixed = content.replace(
    /const \[optionalOpen, setOptionalOpen\] = useState\(false\)[\s\S]{1,50}?\);/,
    'const [optionalOpen, setOptionalOpen] = useState(false);'
);

if (fixed !== content) {
    fs.writeFileSync(path, fixed, 'utf-8');
    console.log('SUCCESS: Fixed broken optionalOpen line');
} else {
    console.log('No fix applied. Raw context above shows what was found.');
}
