import { readFileSync, writeFileSync } from 'fs';

const path = '/app/src/pages/RunPipeline.jsx';
let content = readFileSync(path, 'utf-8');

const idx = content.indexOf('optionalOpen, setOptionalOpen');
// Show bytes individually
const chunk = content.slice(idx - 5, idx + 120);
for (let i = 0; i < chunk.length; i++) {
    process.stdout.write(chunk.charCodeAt(i) + ':' + JSON.stringify(chunk[i]) + ' ');
}
console.log('\n---');
