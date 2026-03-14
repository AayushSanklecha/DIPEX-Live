/**
 * dataAnalyzer.js
 * Schema inference, statistical computation, and chart recommendation for the
 * dynamic Power BI style dashboard.  Works on any row-array of objects
 * regardless of source (file, DB, Kafka, REST API).
 */

// ── Type inference ─────────────────────────────────────────────────────────

const DATE_PATTERNS = [
    /^\d{4}-\d{2}-\d{2}/,       // ISO
    /^\d{2}\/\d{2}\/\d{4}/,     // US
    /^\d{2}-\d{2}-\d{4}/,       // EU
    /^\w{3}\s\d{1,2},?\s\d{4}/, // "Jan 1, 2024"
];

export function inferType(values) {
    const sample = values.filter(v => v !== null && v !== undefined && v !== '').slice(0, 50);
    if (!sample.length) return 'unknown';

    const isNumeric = sample.every(v => !isNaN(Number(v)));
    if (isNumeric) return 'numeric';

    const isDate = sample.every(v =>
        DATE_PATTERNS.some(p => p.test(String(v))) || !isNaN(Date.parse(String(v)))
    );
    if (isDate) return 'temporal';

    // Categorical: <= 30 unique values relative to sample size
    const uniq = new Set(sample.map(v => String(v)));
    if (uniq.size <= Math.min(30, sample.length * 0.8)) return 'categorical';

    return 'text';
}

// ── Schema analysis ─────────────────────────────────────────────────────────

export function analyzeSchema(rows) {
    if (!rows || !rows.length) return { columns: [], numericCols: [], categoricalCols: [], temporalCols: [] };

    const keys = Object.keys(rows[0]);
    const columns = keys.map(key => {
        const values = rows.map(r => r[key]);
        const type = inferType(values);
        return { key, type };
    });

    return {
        columns,
        numericCols: columns.filter(c => c.type === 'numeric').map(c => c.key),
        categoricalCols: columns.filter(c => c.type === 'categorical').map(c => c.key),
        temporalCols: columns.filter(c => c.type === 'temporal').map(c => c.key),
        textCols: columns.filter(c => c.type === 'text').map(c => c.key),
    };
}

// ── Statistical summaries ───────────────────────────────────────────────────

export function computeStats(rows, numericCols) {
    return numericCols.slice(0, 6).map(col => {
        const values = rows
            .map(r => parseFloat(r[col]))
            .filter(v => !isNaN(v));

        if (!values.length) return null;

        const sorted = [...values].sort((a, b) => a - b);
        const sum = values.reduce((s, v) => s + v, 0);
        const mean = sum / values.length;
        const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;

        return {
            col,
            count: values.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            mean: mean,
            median: sorted[Math.floor(sorted.length / 2)],
            stddev: Math.sqrt(variance),
            sum,
        };
    }).filter(Boolean);
}

// ── Distribution data ───────────────────────────────────────────────────────

export function getDistribution(rows, col, topN = 12) {
    const freq = {};
    rows.forEach(r => {
        const k = String(r[col] ?? 'N/A');
        freq[k] = (freq[k] || 0) + 1;
    });

    return Object.entries(freq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topN)
        .map(([name, value]) => ({ name, value }));
}

// ── Time-series data ────────────────────────────────────────────────────────

export function getTimeSeries(rows, timeCol, valueCol) {
    const parsed = rows
        .map(r => ({
            ts: new Date(r[timeCol]).getTime(),
            raw: r[timeCol],
            v: parseFloat(r[valueCol]),
        }))
        .filter(r => !isNaN(r.ts) && !isNaN(r.v))
        .sort((a, b) => a.ts - b.ts);

    // Downsample if > 200 points
    if (parsed.length > 200) {
        const step = Math.ceil(parsed.length / 100);
        return parsed.filter((_, i) => i % step === 0).map(r => ({
            date: new Date(r.ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            [valueCol]: r.v,
        }));
    }

    return parsed.map(r => ({
        date: new Date(r.ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        [valueCol]: r.v,
    }));
}

// ── Scatter data ────────────────────────────────────────────────────────────

export function getScatterData(rows, xCol, yCol, maxPoints = 300) {
    const out = [];
    const step = Math.max(1, Math.floor(rows.length / maxPoints));
    for (let i = 0; i < rows.length; i += step) {
        const x = parseFloat(rows[i][xCol]);
        const y = parseFloat(rows[i][yCol]);
        if (!isNaN(x) && !isNaN(y)) out.push({ x, y });
    }
    return out;
}

// ── Chart recommendation engine ─────────────────────────────────────────────

export function getHistogramData(rows, col, binCount = 10) {
    const values = rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
    if (!values.length) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) {
        return [{ bin: `${min}`, range: [min, min], count: values.length }];
    }

    const step = (max - min) / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
        bin: `${(min + i * step).toFixed(1)} - ${(min + (i + 1) * step).toFixed(1)}`,
        range: [min + i * step, min + (i + 1) * step],
        count: 0
    }));

    values.forEach(v => {
        let idx = Math.floor((v - min) / step);
        if (idx >= binCount) idx = binCount - 1;
        bins[idx].count++;
    });

    return bins;
}

export function getBoxPlotStats(rows, col) {
    const values = rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    if (!values.length) return null;

    const min = values[0];
    const max = values[values.length - 1];

    const getPercentile = (p) => {
        const idx = (values.length - 1) * p;
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        const weight = idx - lower;
        if (upper >= values.length) return values[lower];
        return values[lower] * (1 - weight) + values[upper] * weight;
    };

    const q1 = getPercentile(0.25);
    const median = getPercentile(0.5);
    const q3 = getPercentile(0.75);
    const iqr = q3 - q1;

    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;

    const outliers = values.filter(v => v < lowerFence || v > upperFence);

    return {
        min, max, q1, median, q3,
        lowerFence: Math.max(min, lowerFence),
        upperFence: Math.min(max, upperFence),
        outliers
    };
}

/**
 * Returns an ordered list of charts to render based on the schema.
 * Each entry: { type, title, ...configProps }
 */
export function recommendCharts(schema, rows) {
    const charts = [];
    const { numericCols, categoricalCols, temporalCols, textCols } = schema;

    // RULE 1: Time-series (temporal + numeric) -> Area Chart
    if (temporalCols.length && numericCols.length) {
        charts.push({
            type: 'line',
            title: `${numericCols[0]} over Time`,
            timeCol: temporalCols[0],
            valueCol: numericCols[0],
            data: getTimeSeries(rows, temporalCols[0], numericCols[0]),
        });
    }

    // RULE 2: Numeric Distributions -> Bar or Histogram + BoxPlot
    numericCols.slice(0, 3).forEach(col => {
        const values = rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
        const uniqueCount = new Set(values).size;

        if (uniqueCount <= 10 && uniqueCount > 0) {
            // Low cardinality numeric
            charts.push({
                type: 'bar',
                title: `${col} Breakdown`,
                col,
                data: getDistribution(rows, col)
            });
        } else if (uniqueCount > 10) {
            // High cardinality -> Histogram + BoxPlot
            charts.push({
                type: 'histogram',
                title: `${col} Distribution`,
                col,
                data: getHistogramData(rows, col, 10)
            });
            charts.push({
                type: 'boxplot',
                title: `${col} Statistical Spread`,
                col,
                stats: getBoxPlotStats(rows, col)
            });
        }
    });

    // RULE 3: Scatter (2+ numeric cols) -> Correlation
    if (numericCols.length >= 2) {
        charts.push({
            type: 'scatter',
            title: `${numericCols[0]} vs ${numericCols[1]}`,
            xCol: numericCols[0],
            yCol: numericCols[1],
            data: getScatterData(rows, numericCols[0], numericCols[1]),
        });
    }

    // RULE 4: Categorical Ranking -> Donut / Horizontal Bar / Treemap
    categoricalCols.slice(0, 3).forEach(col => {
        const validRows = rows.filter(r => r[col]);
        const uniqueCount = new Set(validRows.map(r => String(r[col]))).size;
        const dist = getDistribution(rows, col, Math.min(uniqueCount, 30));

        if (uniqueCount <= 5) {
            charts.push({
                type: 'pie',
                title: `${col} Share`,
                col,
                data: dist,
            });
        } else if (uniqueCount <= 20) {
            charts.push({
                type: 'hbar',
                title: `Top ${Math.min(dist.length, 10)} ${col}`,
                col,
                data: dist.slice(0, 10),
            });
        } else {
            // High cardinality category mapping to Treemap
            charts.push({
                type: 'treemap',
                title: `Dense Hierarchy: ${col}`,
                col,
                data: dist,
            });
        }
    });

    // RULE 5: Failsafe Data Table
    charts.push({
        type: 'table',
        title: `Raw Data Preview`,
        data: rows.slice(0, 50),
        columns: [...numericCols, ...categoricalCols, ...temporalCols, ...textCols].slice(0, 15)
    });

    return charts;
}

// ── CSV parser (browser-side, no dependency) ────────────────────────────────

export function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.replace(/^"|"$/g, '').trim());
    return lines.slice(1).map(line => {
        const vals = line.split(',').map(v => v.replace(/^"|"$/g, '').trim());
        const obj = {};
        headers.forEach((h, i) => { obj[h] = vals[i] ?? ''; });
        return obj;
    });
}

// ── Palette ─────────────────────────────────────────────────────────────────

export const PALETTE = [
    '#6366f1', '#22d3ee', '#a78bfa', '#34d399', '#fb923c',
    '#f472b6', '#facc15', '#38bdf8', '#4ade80', '#f87171',
];
