/**
 * Analytics.jsx – Single-page Power BI-style Analytics Dashboard  (v4)
 *
 * All 20+ chart types rendered on ONE scrollable page, no tabs.
 * Sections (with sticky section headers):
 *   1. KPI Summary Cards
 *   2. Pipeline Health  — Gauge · Donut · Pie · Funnel · Stacked Bar
 *   3. Model Performance — Column Chart + metrics
 *   4. Feature Intelligence — Treemap + FI bars
 *   5. EDA – Distributions — Histograms grid
 *   6. EDA – Correlations — Heatmap + table
 *   7. EDA – Anomaly Analysis — Scatter · Bubble · Box Plot
 *   8. Column Statistics Matrix
 *   9. Regulatory Compliance — Shape Map · Pie · violations
 *  10. Statistical Tests — Normality · Stationarity grids
 *  11. Bias & Fairness — Radar · Dumbbell · table
 *  12. RL Agent — Progress Ring · Line/Area · Waterfall
 *  13. Key Insights
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import './Analytics.css';

// ── API helpers ──────────────────────────────────────────────────────────────
// Use relative URL so this works on Hugging Face Spaces, localhost, and any host.
// window.ADAP_API_BASE can be set via a global config script if needed.
const API = window.ADAP_API_BASE || '';
const fetchJSON = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
};

// ── Colour palette ───────────────────────────────────────────────────────────
const PALETTE       = ['#6366f1','#8b5cf6','#10b981','#f59e0b','#3b82f6','#ec4899','#f97316','#06b6d4','#84cc16','#a78bfa'];
const PALETTE_WARM  = ['#ef4444','#f97316','#f59e0b','#eab308','#84cc16'];

// ═══════════════════════════════════════════════════════════════════════════════
// CANVAS CHART COMPONENTS  (all zero external deps)
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Gauge Chart ──────────────────────────────────────────────────────────────
const GaugeChart = ({ value=0, min=0, max=1, label='', color='#6366f1', height=160 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||280; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const cx=W/2, cy=height*0.78, r=Math.min(W,height)*0.38;
    const pct=Math.max(0,Math.min(1,(value-min)/(max-min)));
    const needleA=Math.PI+pct*Math.PI;
    ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,2*Math.PI);
    ctx.lineWidth=14; ctx.strokeStyle='rgba(99,102,241,0.1)'; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx,cy,r,Math.PI,needleA);
    ctx.lineWidth=14; ctx.strokeStyle=color; ctx.lineCap='round'; ctx.stroke();
    ctx.save(); ctx.translate(cx,cy); ctx.rotate(needleA);
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(r*0.82,0);
    ctx.lineWidth=2.5; ctx.strokeStyle='#f1f5f9'; ctx.lineCap='round'; ctx.stroke(); ctx.restore();
    ctx.beginPath(); ctx.arc(cx,cy,5,0,2*Math.PI); ctx.fillStyle='#e2e8f0'; ctx.fill();
    ctx.fillStyle='#f1f5f9'; ctx.font=`bold ${Math.round(r*0.32)}px Inter,sans-serif`; ctx.textAlign='center';
    ctx.fillText((pct*100).toFixed(1)+'%',cx,cy-r*0.22);
    ctx.font=`${Math.round(r*0.2)}px Inter,sans-serif`; ctx.fillStyle='#64748b';
    ctx.fillText(label,cx,cy+2);
    ctx.font='9px Inter,sans-serif'; ctx.fillStyle='#475569';
    ctx.textAlign='left';  ctx.fillText(min,cx-r-2,cy+14);
    ctx.textAlign='right'; ctx.fillText(max,cx+r+2,cy+14);
  },[value,min,max,color,label,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Donut Chart ──────────────────────────────────────────────────────────────
const DonutChart = ({ slices=[], height=180, innerText='' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!slices.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||280; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const total=slices.reduce((s,x)=>s+(x.value||0),0); if(!total) return;
    const cx=W/2,cy=height/2,ro=Math.min(W,height)*0.42,ri=ro*0.6;
    let angle=-Math.PI/2;
    slices.forEach((sl,i)=>{
      const sweep=(sl.value/total)*2*Math.PI;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,ro,angle,angle+sweep); ctx.closePath();
      ctx.fillStyle=sl.color||PALETTE[i%PALETTE.length]; ctx.fill();
      angle+=sweep;
    });
    ctx.beginPath(); ctx.arc(cx,cy,ri,0,2*Math.PI); ctx.fillStyle='#0d1122'; ctx.fill();
    if(innerText){
      const lines=innerText.split('\n');
      ctx.fillStyle='#e2e8f0'; ctx.textAlign='center'; ctx.textBaseline='middle';
      lines.forEach((ln,i)=>{
        ctx.font=i===0?`bold 12px Inter,sans-serif`:`10px Inter,sans-serif`;
        ctx.fillStyle=i===0?'#e2e8f0':'#64748b';
        ctx.fillText(ln,cx,cy+(i-( lines.length-1)/2)*14);
      });
    }
  },[slices,height,innerText]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Pie Chart ────────────────────────────────────────────────────────────────
const PieChart = ({ slices=[], height=180 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!slices.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||280; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const total=slices.reduce((s,x)=>s+(x.value||0),0); if(!total) return;
    const legendW=90,cx=(W-legendW)/2,cy=height/2,r=Math.min(cx*1.5,cy*0.85);
    let angle=-Math.PI/2;
    slices.forEach((sl,i)=>{
      const sweep=(sl.value/total)*2*Math.PI,midA=angle+sweep/2;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,r,angle,angle+sweep); ctx.closePath();
      ctx.fillStyle=sl.color||PALETTE[i%PALETTE.length]; ctx.fill();
      ctx.strokeStyle='#0d1122'; ctx.lineWidth=2; ctx.stroke();
      const pct=Math.round((sl.value/total)*100);
      if(pct>5){
        ctx.fillStyle='#fff'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(pct+'%',cx+Math.cos(midA)*r*0.65,cy+Math.sin(midA)*r*0.65);
      }
      angle+=sweep;
    });
    const lx0=W-legendW+8;
    slices.forEach((sl,i)=>{
      const ly=18+i*20;
      ctx.fillStyle=sl.color||PALETTE[i%PALETTE.length]; ctx.fillRect(lx0,ly,10,10);
      ctx.fillStyle='#94a3b8'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='left'; ctx.textBaseline='middle';
      ctx.fillText(sl.label,lx0+14,ly+5);
    });
  },[slices,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Stacked Bar ──────────────────────────────────────────────────────────────
const StackedBar = ({ bars=[], height=160, xLabel='', yLabel='' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!bars.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:48,r:12,t:8,b:30};
    const maxTotal=Math.max(...bars.map(b=>b.segments.reduce((s,sg)=>s+(sg.value||0),0)),1);
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const barW=Math.min(44,dW/bars.length*0.65),gap=dW/bars.length;
    [0,0.25,0.5,0.75,1].forEach(f=>{
      const y=pad.t+dH*(1-f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y);
      ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      const v=Math.round(maxTotal*f);
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
      ctx.fillText(v>999?`${(v/1000).toFixed(1)}k`:v,pad.l-4,y+3);
    });
    bars.forEach((bar,bi)=>{
      const x=pad.l+bi*gap+(gap-barW)/2; let y=pad.t+dH;
      bar.segments.forEach((sg,si)=>{
        const h=(sg.value/maxTotal)*dH; y-=h;
        ctx.fillStyle=sg.color||PALETTE[si%PALETTE.length]; ctx.fillRect(x,y,barW,h);
      });
      ctx.fillStyle='#64748b'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center';
      ctx.fillText(bar.label,x+barW/2,height-8);
    });
    if(xLabel){ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(xLabel,W/2,height-1);}
    if(yLabel){ctx.save();ctx.translate(11,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[bars,height,xLabel,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Column Chart ─────────────────────────────────────────────────────────────
const ColumnChart = ({ groups=[], labels=[], height=180, xLabel='', yLabel='' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!groups.length||!labels.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:44,r:8,t:12,b:40};
    const maxVal=Math.max(...groups.flatMap(g=>g.values),0.001);
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const groupW=dW/labels.length,barsPerGroup=groups.length,barW=Math.min(20,groupW/barsPerGroup-2);
    [0,0.25,0.5,0.75,1].forEach(f=>{
      const y=pad.t+dH*(1-f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y);
      ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
      ctx.fillText((maxVal*f).toFixed(2),pad.l-4,y+3);
    });
    labels.forEach((lbl,li)=>{
      const gx=pad.l+li*groupW;
      groups.forEach((grp,gi)=>{
        const v=grp.values[li]??0,bh=(v/maxVal)*dH,bx=gx+gi*(barW+2)+(groupW-barsPerGroup*(barW+2))/2,by=pad.t+dH-bh;
        const grad=ctx.createLinearGradient(0,by,0,by+bh);
        grad.addColorStop(0,grp.color||PALETTE[gi%PALETTE.length]);
        grad.addColorStop(1,(grp.color||PALETTE[gi%PALETTE.length])+'44');
        ctx.fillStyle=grad; ctx.beginPath(); ctx.roundRect(bx,by,barW,bh,2); ctx.fill();
      });
      ctx.fillStyle='#64748b'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center';
      ctx.fillText(lbl,gx+groupW/2,height-8);
    });
    groups.forEach((g,i)=>{
      ctx.fillStyle=g.color||PALETTE[i]; ctx.fillRect(pad.l+i*80,height-28,8,8);
      ctx.fillStyle='#94a3b8'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='left';
      ctx.fillText(g.label,pad.l+i*80+12,height-21);
    });
    if(xLabel){ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(xLabel,W/2,height-1);}
    if(yLabel){ctx.save();ctx.translate(11,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[groups,labels,height,xLabel,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Histogram ────────────────────────────────────────────────────────────────
const HistogramCanvas = ({ bins=[], counts=[], color='#6366f1', height=130, xLabel='', yLabel='Freq' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!counts.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||280; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const max=Math.max(...counts,1),pad={l:32,r:6,t:6,b:18};
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b,barW=dW/counts.length;
    const grad=ctx.createLinearGradient(0,0,0,dH+pad.t);
    grad.addColorStop(0,color+'cc'); grad.addColorStop(1,color+'22');
    [0.5,1].forEach(f=>{
      const y=pad.t+dH*(1-f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y);
      ctx.strokeStyle='rgba(99,102,241,0.06)'; ctx.stroke();
    });
    counts.forEach((cnt,i)=>{
      const bh=(cnt/max)*dH,x=pad.l+i*barW,y=pad.t+dH-bh;
      ctx.fillStyle=grad; ctx.fillRect(x+1,y,barW-2,bh);
    });
    ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
    ctx.fillText(Math.round(max),pad.l-3,pad.t+9);
    if(bins.length>=2){
      ctx.textAlign='left';  ctx.fillText(Number(bins[0]).toFixed(0),pad.l,height-2);
      ctx.textAlign='right'; ctx.fillText(Number(bins[bins.length-1]).toFixed(0),W-pad.r,height-2);
    }
    if(xLabel){ctx.fillStyle='#374151';ctx.font='8px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(xLabel,W/2,height);}
    if(yLabel){ctx.save();ctx.translate(8,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#374151';ctx.font='8px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[bins,counts,color,height,xLabel,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`,borderRadius:'4px'}} />;
};

// ─── Heatmap ──────────────────────────────────────────────────────────────────
const HeatmapChart = ({ matrix=[], rowLabels=[], colLabels=[], height=260 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!matrix.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const rows=matrix.length,cols=matrix[0]?.length||0; if(!rows||!cols) return;
    const labelW=58,labelH=22,cellW=(W-labelW)/cols,cellH=(height-labelH)/rows;
    const allVals=matrix.flat(),minV=Math.min(...allVals),maxV=Math.max(...allVals)||1;
    function heatColor(v){
      const t=(v-minV)/(maxV-minV||1);
      if(t<0.5){ const f=t*2; return `rgb(${Math.round(14+232*f)},${Math.round(165-75*f)},${Math.round(233-233*f)})`; }
      else{ const f=(t-0.5)*2; return `rgb(${Math.round(246-7*f)},${Math.round(90+129*(1-f))},${Math.round(f*68)})`; }
    }
    matrix.forEach((row,ri)=>row.forEach((val,ci)=>{
      const x=labelW+ci*cellW,y=labelH+ri*cellH;
      ctx.fillStyle=heatColor(val); ctx.fillRect(x,y,cellW-1,cellH-1);
      ctx.fillStyle=val>0.3?'#0d1122':'#f1f5f9'; ctx.font='7px Inter,sans-serif';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(val.toFixed(2),x+cellW/2,y+cellH/2);
    }));
    colLabels.forEach((l,i)=>{
      ctx.fillStyle='#64748b'; ctx.font='7px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='bottom';
      ctx.fillText(l.length>9?l.slice(0,8)+'…':l,labelW+(i+0.5)*cellW,labelH-2);
    });
    rowLabels.forEach((l,i)=>{
      ctx.fillStyle='#64748b'; ctx.font='7px Inter,sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
      ctx.fillText(l.length>7?l.slice(0,6)+'…':l,labelW-4,labelH+(i+0.5)*cellH);
    });
  },[matrix,rowLabels,colLabels,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Scatter Plot ─────────────────────────────────────────────────────────────
const ScatterPlot = ({ points=[], xLabel='', yLabel='', color='#6366f1', height=200 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!points.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:36,r:12,t:12,b:28};
    const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
    const xMin=Math.min(...xs),xMax=Math.max(...xs)||1,yMin=Math.min(...ys),yMax=Math.max(...ys)||1;
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const toX=x=>pad.l+(x-xMin)/(xMax-xMin)*dW,toY=y=>pad.t+dH-(y-yMin)/(yMax-yMin)*dH;
    [0,0.5,1].forEach(f=>{
      const y=pad.t+dH*(1-f),x=pad.l+dW*f;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+dH); ctx.stroke();
    });
    points.forEach(p=>{
      ctx.beginPath(); ctx.arc(toX(p.x),toY(p.y),3.5,0,2*Math.PI);
      ctx.fillStyle=color+'99'; ctx.fill(); ctx.strokeStyle=color; ctx.lineWidth=1; ctx.stroke();
    });
    ctx.fillStyle='#64748b'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center'; ctx.fillText(xLabel,W/2,height-2);
    ctx.save(); ctx.translate(10,height/2); ctx.rotate(-Math.PI/2); ctx.textAlign='center'; ctx.fillText(yLabel,0,0); ctx.restore();
  },[points,xLabel,yLabel,color,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Bubble Chart ─────────────────────────────────────────────────────────────
const BubbleChart = ({ bubbles=[], xLabel='', yLabel='', height=220 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!bubbles.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:40,r:14,t:14,b:30};
    const xs=bubbles.map(b=>b.x),ys=bubbles.map(b=>b.y),rs=bubbles.map(b=>b.r);
    const xMin=Math.min(...xs),xMax=Math.max(...xs)||1,yMin=Math.min(...ys),yMax=Math.max(...ys)||1,maxR=Math.max(...rs)||1;
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const toX=x=>pad.l+(x-xMin)/(xMax-xMin)*dW,toY=y=>pad.t+dH-(y-yMin)/(yMax-yMin)*dH,toR=r=>(r/maxR)*Math.min(dW,dH)*0.15;
    [0,0.5,1].forEach(f=>{
      const y=pad.t+dH*(1-f),x=pad.l+dW*f;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+dH); ctx.stroke();
    });
    bubbles.forEach((b,i)=>{
      const px=toX(b.x),py=toY(b.y),pr=Math.max(5,toR(b.r)),col=b.color||PALETTE_WARM[i%PALETTE_WARM.length];
      ctx.beginPath(); ctx.arc(px,py,pr,0,2*Math.PI);
      ctx.fillStyle=col+'55'; ctx.fill(); ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.stroke();
      if(b.label){ ctx.fillStyle='#e2e8f0'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(b.label,px,py); }
    });
    ctx.fillStyle='#64748b'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center'; ctx.fillText(xLabel,W/2,height-2);
    ctx.save(); ctx.translate(10,height/2); ctx.rotate(-Math.PI/2); ctx.textAlign='center'; ctx.fillText(yLabel,0,0); ctx.restore();
  },[bubbles,xLabel,yLabel,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Box Plot ─────────────────────────────────────────────────────────────────
const BoxPlot = ({ boxes=[], height=200, yLabel='Value' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!boxes.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:40,r:8,t:14,b:32};
    const allVals=boxes.flatMap(b=>[b.min,b.max]),minV=Math.min(...allVals),maxV=Math.max(...allVals)||1;
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const toY=v=>pad.t+dH-(v-minV)/(maxV-minV)*dH;
    const bW=Math.min(24,dW/boxes.length*0.5),gap=dW/boxes.length;
    [0,0.25,0.5,0.75,1].forEach(f=>{
      const y=pad.t+dH*(1-f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
      ctx.fillText((minV+(maxV-minV)*f).toFixed(0),pad.l-4,y+3);
    });
    boxes.forEach((box,i)=>{
      const cx=pad.l+i*gap+gap/2,col=box.color||PALETTE[i%PALETTE.length];
      const y1=toY(box.min),yQ1=toY(box.q1),yMed=toY(box.median),yQ3=toY(box.q3),yMx=toY(box.max);
      ctx.strokeStyle=col+'88'; ctx.lineWidth=1.5;
      [[cx,y1,cx,yQ1],[cx,yQ3,cx,yMx],[cx-bW/3,y1,cx+bW/3,y1],[cx-bW/3,yMx,cx+bW/3,yMx]].forEach(([x1,y1_,x2,y2])=>{
        ctx.beginPath(); ctx.moveTo(x1,y1_); ctx.lineTo(x2,y2); ctx.stroke();
      });
      ctx.fillStyle=col+'22'; ctx.fillRect(cx-bW/2,yQ3,bW,yQ1-yQ3);
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.strokeRect(cx-bW/2,yQ3,bW,yQ1-yQ3);
      ctx.beginPath(); ctx.moveTo(cx-bW/2,yMed); ctx.lineTo(cx+bW/2,yMed);
      ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.stroke();
      ctx.fillStyle='#64748b'; ctx.font='7px Inter,sans-serif'; ctx.textAlign='center';
      ctx.fillText(box.label.length>7?box.label.slice(0,6)+'…':box.label,cx,height-5);
    });
    if(yLabel){ctx.save();ctx.translate(11,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[boxes,height,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Waterfall Chart ─────────────────────────────────────────────────────────
const WaterfallChart = ({ bars=[], height=210, yLabel='Value' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!bars.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:44,r:8,t:14,b:50};
    let running=0;
    const segments=bars.map(b=>{ const start=b.isTotal?0:running; running+=b.value; return {...b,start,end:b.isTotal?running:running}; });
    const allVals=segments.flatMap(s=>[s.start,s.end]);
    const minV=Math.min(0,...allVals),maxV=Math.max(...allVals)||0.01,range=maxV-minV;
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const toY=v=>pad.t+dH-(v-minV)/range*dH;
    const barW=Math.min(28,dW/segments.length*0.6),gap=dW/segments.length;
    const zeroY=toY(0);
    ctx.beginPath(); ctx.moveTo(pad.l,zeroY); ctx.lineTo(W-pad.r,zeroY);
    ctx.strokeStyle='rgba(99,102,241,0.25)'; ctx.lineWidth=1; ctx.stroke();
    [0.25,0.5,0.75,1].forEach(f=>{
      const y=toY(minV+range*f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y);
      ctx.strokeStyle='rgba(99,102,241,0.06)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
      ctx.fillText((minV+range*f).toFixed(2),pad.l-4,y+3);
    });
    segments.forEach((seg,i)=>{
      const cx=pad.l+i*gap+(gap-barW)/2,y1=toY(seg.start),y2=toY(seg.end);
      const top=Math.min(y1,y2),boxH=Math.abs(y2-y1)||2;
      const col=seg.isTotal?'#6366f1':seg.value>=0?'#10b981':'#ef4444';
      ctx.fillStyle=col+'88'; ctx.fillRect(cx,top,barW,boxH);
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.strokeRect(cx,top,barW,boxH);
      if(i<segments.length-1&&!seg.isTotal){
        const nx=cx+barW+(gap-barW);
        ctx.beginPath(); ctx.moveTo(cx+barW,toY(seg.end)); ctx.lineTo(nx,toY(seg.end));
        ctx.strokeStyle='rgba(99,102,241,0.2)'; ctx.lineWidth=1; ctx.setLineDash([3,3]); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.fillStyle='#64748b'; ctx.font='7px Inter,sans-serif'; ctx.textAlign='center';
      ctx.save(); ctx.translate(cx+barW/2,height-6); ctx.rotate(-Math.PI/4); ctx.textAlign='right';
      ctx.fillText(seg.label.replace(/_/g,' '),0,0); ctx.restore();
    });
    if(yLabel){ctx.save();ctx.translate(11,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[bars,height,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Radar / Spider Chart ─────────────────────────────────────────────────────
const RadarChart = ({ series=[], labels=[], height=220, centerLabel='' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!labels.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||300; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const cx=W/2,cy=height/2,r=Math.min(W,height)*0.36,N=labels.length;
    const angle=i=>(i/N)*2*Math.PI-Math.PI/2;
    [0.25,0.5,0.75,1].forEach(f=>{
      ctx.beginPath();
      for(let i=0;i<N;i++){const a=angle(i),x=cx+Math.cos(a)*r*f,y=cy+Math.sin(a)*r*f; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
      ctx.closePath(); ctx.strokeStyle='rgba(99,102,241,0.12)'; ctx.lineWidth=1; ctx.stroke();
    });
    for(let i=0;i<N;i++){
      const a=angle(i);
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);
      ctx.strokeStyle='rgba(99,102,241,0.1)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#64748b'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(labels[i].length>9?labels[i].slice(0,8)+'…':labels[i],cx+Math.cos(a)*(r+16),cy+Math.sin(a)*(r+16));
    }
    series.forEach((s,si)=>{
      const col=s.color||PALETTE[si%PALETTE.length];
      ctx.beginPath();
      s.values.forEach((v,i)=>{const a=angle(i); i===0?ctx.moveTo(cx+Math.cos(a)*r*v,cy+Math.sin(a)*r*v):ctx.lineTo(cx+Math.cos(a)*r*v,cy+Math.sin(a)*r*v);});
      ctx.closePath(); ctx.fillStyle=col+'33'; ctx.fill(); ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.stroke();
    });
    if(centerLabel){ctx.fillStyle='#475569';ctx.font='8px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(centerLabel,cx,4);}
  },[series,labels,height,centerLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Line / Area Chart ────────────────────────────────────────────────────────
const LineAreaChart = ({ lines=[], height=180, xLabel='Episode', yLabel='Reward' }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!lines.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:38,r:14,t:12,b:24};
    const allData=lines.flatMap(l=>l.data);
    const minV=Math.min(...allData),maxV=Math.max(...allData)||0.001,maxLen=Math.max(...lines.map(l=>l.data.length),1);
    const dW=W-pad.l-pad.r,dH=height-pad.t-pad.b;
    const toX=i=>pad.l+i/(maxLen-1)*dW,toY=v=>pad.t+dH-(v-minV)/(maxV-minV)*dH;
    [0,0.25,0.5,0.75,1].forEach(f=>{
      const y=pad.t+dH*(1-f);
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='right';
      ctx.fillText((minV+(maxV-minV)*f).toFixed(2),pad.l-4,y+3);
    });
    lines.forEach((line,li)=>{
      const col=line.color||PALETTE[li%PALETTE.length]; if(!line.data.length) return;
      if(line.area!==false){
        ctx.beginPath(); ctx.moveTo(toX(0),toY(line.data[0]));
        line.data.forEach((v,i)=>ctx.lineTo(toX(i),toY(v)));
        ctx.lineTo(toX(line.data.length-1),pad.t+dH); ctx.lineTo(toX(0),pad.t+dH); ctx.closePath();
        const aGrad=ctx.createLinearGradient(0,pad.t,0,pad.t+dH);
        aGrad.addColorStop(0,col+'44'); aGrad.addColorStop(1,col+'00'); ctx.fillStyle=aGrad; ctx.fill();
      }
      ctx.beginPath(); ctx.moveTo(toX(0),toY(line.data[0]));
      line.data.forEach((v,i)=>ctx.lineTo(toX(i),toY(v)));
      ctx.strokeStyle=col; ctx.lineWidth=2; ctx.lineJoin='round'; ctx.stroke();
      line.data.forEach((v,i)=>{ctx.beginPath(); ctx.arc(toX(i),toY(v),3,0,2*Math.PI); ctx.fillStyle=col; ctx.fill();});
    });
    lines.forEach((l,i)=>{
      ctx.fillStyle=l.color||PALETTE[i]; ctx.fillRect(pad.l+i*90,height-14,10,3);
      ctx.fillStyle='#94a3b8'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='left'; ctx.fillText(l.label,pad.l+i*90+13,height-10);
    });
    if(xLabel){ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(xLabel,W/2,height-1);}
    if(yLabel){ctx.save();ctx.translate(11,pad.t+dH/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillStyle='#475569';ctx.font='9px Inter,sans-serif';ctx.fillText(yLabel,0,0);ctx.restore();}
  },[lines,height,xLabel,yLabel]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Progress Ring ────────────────────────────────────────────────────────────
const ProgressRing = ({ value=0, max=100, label='', subLabel='', color='#8b5cf6', height=140 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||180; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const cx=W/2,cy=height/2,r=Math.min(W,height)*0.38,pct=Math.min(1,value/Math.max(max,1));
    ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI); ctx.strokeStyle='rgba(99,102,241,0.1)'; ctx.lineWidth=10; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx,cy,r,-Math.PI/2,-Math.PI/2+pct*2*Math.PI);
    ctx.strokeStyle=color; ctx.lineWidth=10; ctx.lineCap='round'; ctx.stroke();
    ctx.fillStyle='#f1f5f9'; ctx.font=`bold ${Math.round(r*0.33)}px Inter,sans-serif`;
    ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(`${value}/${max}`,cx,cy-6);
    ctx.font=`${Math.round(r*0.2)}px Inter,sans-serif`; ctx.fillStyle='#64748b'; ctx.fillText(label,cx,cy+r*0.35);
    if(subLabel){ctx.font='8px Inter,sans-serif';ctx.fillStyle='#374151';ctx.fillText(subLabel,cx,height-4);}
  },[value,max,label,subLabel,color,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Funnel Chart ─────────────────────────────────────────────────────────────
const FunnelChart = ({ stages=[], height=200 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!stages.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||360; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const maxVal=stages[0]?.value||1,slotH=height/stages.length,padX=16;
    stages.forEach((s,i)=>{
      const pct=s.value/maxVal,prevPct=i===0?1:stages[i-1].value/maxVal;
      const topW=(W-padX*2)*prevPct,botW=(W-padX*2)*pct,topX=(W-topW)/2,botX=(W-botW)/2,y=i*slotH;
      const col=s.color||PALETTE[i%PALETTE.length];
      ctx.beginPath(); ctx.moveTo(topX,y); ctx.lineTo(topX+topW,y); ctx.lineTo(botX+botW,y+slotH-2); ctx.lineTo(botX,y+slotH-2); ctx.closePath();
      ctx.fillStyle=col+'77'; ctx.fill(); ctx.strokeStyle=col; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#e2e8f0'; ctx.font='bold 10px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(`${s.label}  ${s.value?.toLocaleString()}`,W/2,y+slotH/2);
    });
  },[stages,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Treemap ──────────────────────────────────────────────────────────────────
const TreemapChart = ({ items=[], height=120 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!items.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const total=items.reduce((s,x)=>s+x.value,0)||1; let x=0;
    items.forEach((item,i)=>{
      const w=Math.round((item.value/total)*W),col=item.color||PALETTE[i%PALETTE.length];
      ctx.fillStyle=col+'55'; ctx.fillRect(x,0,w-1,height);
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.strokeRect(x,0,w-1,height);
      if(w>32){
        ctx.fillStyle='#f1f5f9'; ctx.font=`bold ${Math.min(11,w*0.18)}px Inter,sans-serif`;
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(item.label.length>7?item.label.slice(0,6)+'…':item.label,x+w/2,height/2-7);
        ctx.font=`${Math.min(9,w*0.14)}px Inter,sans-serif`; ctx.fillStyle='#94a3b8';
        ctx.fillText(`${(item.value*100).toFixed(1)}%`,x+w/2,height/2+8);
      }
      x+=w;
    });
  },[items,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`,borderRadius:'6px'}} />;
};

// ─── Dumbbell Chart ───────────────────────────────────────────────────────────
const DumbbellChart = ({ rows=[], height=200 }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c=ref.current; if(!c||!rows.length) return;
    const ctx=c.getContext('2d');
    const W=c.offsetWidth||400; c.width=W; c.height=height; ctx.clearRect(0,0,W,height);
    const pad={l:90,r:24,t:14,b:14};
    const allVals=rows.flatMap(r=>[r.lo,r.hi]),minV=Math.min(...allVals),maxV=Math.max(...allVals)||1;
    const dW=W-pad.l-pad.r,rowH=(height-pad.t-pad.b)/rows.length;
    const toX=v=>pad.l+(v-minV)/(maxV-minV)*dW;
    [0,0.25,0.5,0.75,1].forEach(f=>{
      const x=pad.l+dW*f;
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,height-pad.b);
      ctx.strokeStyle='rgba(99,102,241,0.07)'; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle='#475569'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='center';
      ctx.fillText((minV+(maxV-minV)*f).toFixed(2),x,height-pad.b+10);
    });
    rows.forEach((row,i)=>{
      const cy=pad.t+i*rowH+rowH/2,lx=toX(row.lo),hx=toX(row.hi);
      ctx.beginPath(); ctx.moveTo(lx,cy); ctx.lineTo(hx,cy); ctx.strokeStyle='rgba(99,102,241,0.3)'; ctx.lineWidth=2; ctx.stroke();
      ctx.beginPath(); ctx.arc(lx,cy,5,0,2*Math.PI); ctx.fillStyle=row.loColor||'#ef4444'; ctx.fill();
      ctx.beginPath(); ctx.arc(hx,cy,5,0,2*Math.PI); ctx.fillStyle=row.hiColor||'#10b981'; ctx.fill();
      ctx.fillStyle='#94a3b8'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='right'; ctx.fillText(row.label,pad.l-8,cy+3);
    });
  },[rows,height]);
  return <canvas ref={ref} style={{width:'100%',height:`${height}px`}} />;
};

// ─── Shape Map ────────────────────────────────────────────────────────────────
const ShapeMap = ({ tiles=[] }) => (
  <div className="shape-map-grid">
    {tiles.map((t,i)=>{
      const col=t.status==='PASS'?'#10b981':t.status==='WARN'?'#f59e0b':'#ef4444';
      const bg=t.status==='PASS'?'rgba(16,185,129,0.08)':t.status==='WARN'?'rgba(245,158,11,0.08)':'rgba(239,68,68,0.08)';
      return (
        <div key={i} className="shape-map-tile" style={{border:`1.5px solid ${col}44`,background:bg}}>
          <div className="shape-map-status">{t.status==='PASS'?'✅':t.status==='WARN'?'⚠️':'❌'}</div>
          <div className="shape-map-label">{t.label}</div>
          <div className="shape-map-sub" style={{color:col}}>{t.passed}/{t.rules} rules</div>
          <div className="shape-map-bar"><div className="shape-map-bar-fill" style={{width:`${Math.round((t.passed/Math.max(t.rules,1))*100)}%`,background:col}}/></div>
        </div>
      );
    })}
  </div>
);

// ─── Matrix Table ─────────────────────────────────────────────────────────────
const MatrixTable = ({ rows=[], cols=[], data=[] }) => {
  const allVals=data.flat().filter(v=>v!=null),minV=Math.min(...allVals)||0,maxV=Math.max(...allVals)||1;
  const cellBg=v=>{ if(v==null) return 'transparent'; const t=(v-minV)/(maxV-minV||1); return `rgba(99,102,241,${(t*0.5+0.05).toFixed(2)})`; };
  return (
    <div className="analytics-table-wrap">
      <table className="analytics-table">
        <thead><tr><th>Column</th>{cols.map((c,i)=><th key={i}>{c}</th>)}</tr></thead>
        <tbody>{rows.map((row,ri)=>(
          <tr key={ri}>
            <td style={{fontWeight:600,color:'#94a3b8'}}>{row}</td>
            {cols.map((_,ci)=><td key={ci} className="cell-mono" style={{background:cellBg(data[ri]?.[ci]),textAlign:'right'}}>{data[ri]?.[ci]!=null?data[ri][ci].toFixed(3):'—'}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
};

// ─── Shared UI ────────────────────────────────────────────────────────────────
const KpiCard = ({ label, value, sub, icon, color='#6366f1', loading=false }) => (
  <div className="kpi-card" style={{'--kpi-color':`linear-gradient(90deg,${color},${color}88)`}}>
    <div className="kpi-card-label">{label}</div>
    {loading?(<><div className="analytics-shimmer" style={{width:'60%',height:24}}/><div className="analytics-shimmer" style={{width:'40%'}}/></>):
      (<><div className="kpi-card-value">{value??'—'}</div>{sub&&<div className="kpi-card-sub">{sub}</div>}</>)}
    {icon&&<span className="kpi-card-icon">{icon}</span>}
  </div>
);

// Chart card wrapper — title + badge + one-line description + children
const ChartCard = ({ title, badge, desc, children, wide=false, style={} }) => (
  <div className={`chart-card${wide?' chart-wide':''}`} style={style}>
    <div className="chart-card-header">
      <div className="chart-card-title">{title}</div>
      {badge && <span className="chart-card-badge">{badge}</span>}
    </div>
    {desc && <div className="chart-card-desc">{desc}</div>}
    {children}
  </div>
);

const FiBar = ({ label, score, maxScore=1 }) => (
  <div className="fi-bar-wrap">
    <div className="fi-bar-label" title={label}>{label}</div>
    <div className="fi-bar-track"><div className="fi-bar-fill" style={{width:`${Math.round((score/Math.max(maxScore,0.001))*100)}%`}}/></div>
    <div className="fi-bar-value">{(score*100).toFixed(1)}%</div>
  </div>
);

const ViolationItem = ({ rule_name, severity='warning', message, remediation, domain, column, offending_count, type: vtype }) => {
  const sev = severity?.toLowerCase();
  const isAml   = rule_name?.toLowerCase().includes('aml') || rule_name?.toLowerCase().includes('suspicious');
  const isCrit  = sev === 'critical';
  const isErr   = sev === 'error';

  // Severity colour
  const sevColor  = isCrit ? '#ef4444' : isErr ? '#f87171' : '#f59e0b';
  const sevBorder = isCrit ? 'rgba(239,68,68,0.25)' : isErr ? 'rgba(248,113,113,0.2)' : 'rgba(245,158,11,0.2)';
  const sevBg     = isCrit ? 'rgba(239,68,68,0.06)' : isErr ? 'rgba(248,113,113,0.05)' : 'rgba(245,158,11,0.05)';

  // Human-readable rule label
  const ruleLabel = (rule_name || 'Unknown Rule')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div style={{
      background: sevBg,
      border: `1px solid ${sevBorder}`,
      borderRadius: '10px',
      padding: '0.95rem 1.1rem',
      marginBottom: '0.65rem',
      display: 'flex',
      gap: '1rem',
      alignItems: 'flex-start',
    }}>
      {/* Left: severity pill */}
      <div style={{
        minWidth: '72px',
        textAlign: 'center',
        paddingTop: '0.1rem',
      }}>
        <div style={{
          background: sevColor,
          color: '#fff',
          borderRadius: '6px',
          fontSize: '0.68rem',
          fontWeight: 800,
          letterSpacing: '0.06em',
          padding: '3px 8px',
          textTransform: 'uppercase',
        }}>{severity}</div>
        {isAml && (
          <div style={{
            marginTop: '0.4rem',
            background: 'rgba(139,92,246,0.18)',
            border: '1px solid rgba(139,92,246,0.35)',
            color: '#a78bfa',
            borderRadius: '5px',
            fontSize: '0.62rem',
            fontWeight: 700,
            padding: '2px 6px',
            letterSpacing: '0.04em',
          }}>SAR</div>
        )}
      </div>

      {/* Right: audit detail */}
      <div style={{flex: 1, minWidth: 0}}>

        {/* Rule name + domain */}
        <div style={{display:'flex', alignItems:'center', gap:'0.5rem', flexWrap:'wrap', marginBottom:'0.35rem'}}>
          <span style={{fontWeight: 700, fontSize: '0.88rem', color: '#e2e8f0'}}>
            {ruleLabel}
          </span>
          {domain && (
            <span style={{
              fontSize: '0.72rem', color: '#818cf8', fontWeight: 600,
              background: 'rgba(99,102,241,0.12)', borderRadius: '4px', padding: '1px 7px',
            }}>{domain.toUpperCase()}</span>
          )}
          {isAml && (
            <span style={{fontSize:'0.72rem', color:'#94a3b8', fontStyle:'italic'}}>
              FATF Rec. 10 / PMLA
            </span>
          )}
        </div>

        {/* Where: column + record count */}
        {(column && column !== 'N/A') && (
          <div style={{
            display: 'flex', gap: '1.2rem', marginBottom: '0.4rem',
            fontSize: '0.78rem', color: '#94a3b8',
          }}>
            <span>📌 Column: <strong style={{color:'#cbd5e1'}}>{column}</strong></span>
            {offending_count != null && offending_count > 0 && (
              <span>🔢 Records flagged: <strong style={{color: sevColor}}>{offending_count.toLocaleString()}</strong></span>
            )}
          </div>
        )}

        {/* Why: full message */}
        <div style={{fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.55, marginBottom: remediation ? '0.4rem' : 0}}>
          {message}
        </div>

        {/* What to do: remediation */}
        {remediation && (
          <div style={{
            fontSize: '0.79rem', color: '#818cf8', lineHeight: 1.5,
            paddingTop: '0.35rem',
            borderTop: '1px solid rgba(99,102,241,0.12)',
            display: 'flex', gap: '0.4rem',
          }}>
            <span style={{flexShrink: 0}}>→</span>
            <span>{remediation}</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Section header with sticky scroll anchor
const SectionHeader = ({ id, icon, title }) => (
  <div id={id} className="pbi-section-header">
    <span className="pbi-section-icon">{icon}</span>
    <span className="pbi-section-title">{title}</span>
    <div className="pbi-section-line"/>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════
const Analytics = () => {
  const [runList,     setRunList]     = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [data,        setData]        = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);

  const loadRunList = useCallback(async () => {
    try {
      const result = await fetchJSON(`${API}/api/results`);
      const list = Array.isArray(result) ? result : (result.runs||[]);
      list.sort((a,b)=>(b.timestamp||b.run_id||'').localeCompare(a.timestamp||a.run_id||''));
      setRunList(list);
      if(list.length>0&&!selectedRun) setSelectedRun(list[0].run_id||list[0]);
    } catch { setRunList([]); setSelectedRun(''); }
  },[selectedRun]);

  const loadRunData = useCallback(async (runId) => {
    if(!runId||runId==='demo-run'){ setData(getMockData()); setLoading(false); return; }
    setLoading(true); setError(null);
    try { setData(await fetchJSON(`${API}/api/analytics/${runId}`)); }
    catch { try { setData(await fetchJSON(`${API}/api/results/${runId}`)); } catch { setData(getMockData()); } }
    finally { setLoading(false); }
  },[]);

  useEffect(()=>{ loadRunList(); },[]);
  useEffect(()=>{ if(selectedRun) loadRunData(selectedRun); },[selectedRun]);

  // ── Derived data ──────────────────────────────────────────────────────────
  const summary    = data?.eda_report?.summary || data?.summary || {};
  const numStats   = data?.eda_report?.numeric_stats || {};
  const insights   = data?.insights || [];
  const regSummary = data?.regulatory_summary || {};
  const violations = data?.cross_domain_flags || [];
  const fi         = data?.feature_importance || {};
  const statTests  = data?.statistical_tests || {};
  const biasFair   = data?.bias_fairness_report || {};
  const anomDeep   = data?.anomaly_deep_dive || {};
  const rlSummary  = data?.rl_agent_summary || {};
  const modelM     = data?.model_metrics || {};
  const govSummary = data?.governance_summary || {};
  const lineage    = data?.data_lineage || {};

  // ── Regulatory domain context ─────────────────────────────────────────────
  const domainEnforced  = regSummary.domain_enforced  || (regSummary.domains_checked||[]).length > 0;
  const domainUsed      = regSummary.domain_used      || (regSummary.domains_checked||[])[0] || '';
  const domainListUsed  = regSummary.domain_list_used || regSummary.domains_checked || [];
  const cleanPass       = domainEnforced && violations.length === 0 && (regSummary.rules_failed||0) === 0 && (regSummary.rules_warned||0) === 0;

  const nRows      = summary.n_rows    || data?.row_count || 0;
  const nCols      = summary.n_cols    || data?.col_count || 0;
  const nullPct    = summary.overall_null_pct || 0;
  const anomalyPct = summary.anomaly_pct || 0;
  const confidence = data?.confidence_score || 0;
  const gateDec    = data?.gate_decision || data?.overall_decision || '—';
  const gateClass  = gateDec==='PASS'?'pass':gateDec==='WARN'?'warn':'fail';

  const fiSorted   = Object.entries(fi).sort((a,b)=>b[1]-a[1]).slice(0,12);
  const fiMax      = fiSorted[0]?.[1]||1;
  const numEntries = Object.entries(numStats);
  const corrData   = data?.eda_report?.correlations||[];
  const normTests  = statTests.normality||[];
  const biasResults= biasFair.results||[];
  const rwc        = rlSummary.reward_components||{};
  const episodes   = rlSummary.episode_count||0;

  // ── Computed chart data ───────────────────────────────────────────────────
  // Correlation matrix
  const corrCols = [...new Set(corrData.flatMap(c=>[c.col_a||c.a,c.col_b||c.b]))].slice(0,9);
  const corrMatrix = corrCols.map(row=>corrCols.map(col=>{
    if(row===col) return 1;
    const f=corrData.find(c=>((c.col_a||c.a)===row&&(c.col_b||c.b)===col)||((c.col_a||c.a)===col&&(c.col_b||c.b)===row));
    return f?+(f.correlation??f.r??0):0;
  }));

  // Scatter
  const scatterPts = numEntries.slice(0,20).map(([,s])=>({x:+(s.mean||0),y:+(s.std||0)}));

  // Bubble anomaly
  const bubbles = (anomDeep.per_column||[]).map((c,i)=>({
    x:c.z_score_max||0, y:Math.abs(c.if_score_mean||0)*10, r:c.anomaly_count||1, label:c.col, color:PALETTE_WARM[i%PALETTE_WARM.length],
  }));

  // Box plots
  const boxes = numEntries.slice(0,8).map(([col,s],i)=>({
    label:col, min:s.min||0, q1:s.q1||(s.mean-(s.std||0)), median:s.median||s.mean||0,
    q3:s.q3||(s.mean+(s.std||0)), max:s.max||0, color:PALETTE[i%PALETTE.length],
  }));

  // Matrix cols
  const matrixRows = numEntries.slice(0,10).map(([col])=>col);
  const matrixData = numEntries.slice(0,10).map(([,s])=>[s.mean,s.std,s.skewness,s.null_pct].map(v=>v!=null?+v:null));

  // Funnel
  const funnelStages = ['raw','bronze','silver','gold'].filter(s=>lineage[s]).map((s,i)=>({
    label:s.toUpperCase(), value:lineage[s]?.rows||0,
    color:['#6366f1','#b47832','#94a3b8','#eab308'][i],
  }));

  // Stacked bar lineage
  const stackedBars = ['raw','bronze','silver','gold'].filter(s=>lineage[s]).map((s,i)=>({
    label:s.toUpperCase(), segments:[
      {label:'Rows',value:lineage[s].rows||0,color:PALETTE[i]},
      {label:'Dropped',value:i===0?0:Math.max(0,(lineage[['raw','bronze','silver','gold'][i-1]]?.rows||0)-(lineage[s].rows||0)),color:'#334155'},
    ],
  }));

  // Treemap
  const treemapItems = fiSorted.map(([lbl,v],i)=>({label:lbl,value:v,color:PALETTE[i%PALETTE.length]}));

  // Column chart model metrics
  const modelLabels = Object.keys(modelM);
  const modelGroups = modelLabels.length ? [{label:'Score',color:'#6366f1',values:modelLabels.map(k=>+(modelM[k])||0)}] : [];

  // Donut slices
  const donutSlices = [{label:'Valid',value:1-nullPct,color:'#10b981'},{label:'Missing',value:nullPct,color:'#ef4444'}];
  // Pie slices: when domain ran clean with 0 violations, show at least 1 passed so chart renders
  const _rPass = regSummary.rules_passed || (domainEnforced && !(regSummary.rules_total) ? 1 : 0);
  const pieSlices = [
    {label:'Passed',value:_rPass,color:'#10b981'},
    {label:'Warned',value:regSummary.rules_warned||0,color:'#f59e0b'},
    {label:'Failed',value:regSummary.rules_failed||0,color:'#ef4444'},
  ];

  // Domain compliance tiles — build from domain_compliance, or domains_checked, or domainListUsed fallback
  const domains = regSummary.domains_checked || domainListUsed || [];
  const domainTiles = data?.domain_compliance || domains.map(d => {
    const dViolations = violations.filter(v => v.domain === d || v.domain === d.toLowerCase());
    const hasFail = dViolations.some(v => (v.severity||'').toUpperCase() === 'ERROR' || (v.severity||'').toUpperCase() === 'CRITICAL');
    const hasWarn = dViolations.some(v => (v.severity||'').toUpperCase() === 'WARNING');
    const perDomainTotal  = Math.max(Math.ceil((regSummary.rules_total  || 1) / Math.max(domains.length, 1)), 1);
    const perDomainPassed = Math.ceil((regSummary.rules_passed || _rPass) / Math.max(domains.length, 1));
    return {
      label:  d.toUpperCase(),
      status: hasFail ? 'FAIL' : hasWarn ? 'WARN' : 'PASS',
      rules:  perDomainTotal,
      passed: perDomainPassed,
    };
  });


  // Radar bias
  const maxSample = Math.max(...biasResults.map(r=>r.sample_size||0),1);
  const radarLabels = biasResults.length ? ['Sample Sz','Pos Rate','Dis. Impact','Status'] : [];
  const radarSeries = biasResults.map((r,i)=>({
    label:`${r.group_col}=${r.group_value}`,color:PALETTE[i%PALETTE.length],
    values:[(r.sample_size||0)/maxSample,Math.min(1,r.positive_rate||0),Math.min(1,r.disparate_impact||0),r.status==='PASS'?1:0.2],
  }));

  // Dumbbell
  const dumbbellCols=[...new Set(biasResults.map(r=>r.group_col))];
  const dumbbellRows=dumbbellCols.map(col=>{
    const dis=biasResults.filter(r=>r.group_col===col).map(r=>r.disparate_impact||0);
    return{label:col,lo:Math.min(...dis),hi:Math.max(...dis),loColor:'#ef4444',hiColor:'#10b981'};
  });

  // Waterfall reward
  const waterfallBars = Object.entries(rwc).filter(([k])=>k!=='total').map(([k,v])=>({label:k,value:+v}));
  if(rwc.total!=null) waterfallBars.push({label:'Total',value:+rwc.total,isTotal:true});

  // RL line
  const rewardHistory = rlSummary.reward_history||Array.from({length:episodes||5},(_,i)=>+(Math.random()*0.3+0.4).toFixed(3));
  const rewardLines   = [{label:'Episode Reward',data:rewardHistory,color:'#8b5cf6',area:true}];

  // ─── Render ───────────────────────────────────────────────────────────────
  if(!loading && !data && runList.length===0) return (
    <div className="analytics-page">
      <div className="analytics-header">
        <div className="analytics-header-icon">📊</div>
        <div className="analytics-header-title"><h1>Analytics &amp; Reports</h1><p>Power BI-style single-page intelligence dashboard</p></div>
      </div>
      <div className="analytics-empty" style={{padding:'5rem 2rem'}}>
        <div className="analytics-empty-icon" style={{fontSize:'4rem'}}>📊</div>
        <h3 style={{fontSize:'1.4rem',color:'#e2e8f0',margin:'1rem 0 0.5rem'}}>No Pipeline Runs Yet</h3>
        <p style={{fontSize:'0.9rem',maxWidth:'420px',lineHeight:1.6}}>Upload a dataset and run the pipeline to populate this dashboard.</p>
        <Link to="/run" style={{display:'inline-flex',alignItems:'center',gap:'0.5rem',marginTop:'1.5rem',padding:'0.75rem 1.5rem',
          background:'linear-gradient(135deg,#6366f1,#8b5cf6)',color:'white',borderRadius:'10px',textDecoration:'none',fontWeight:700}}>🚀 Run Pipeline</Link>
      </div>
    </div>
  );

  return (
    <div className="analytics-page">

      {/* ── Sticky top bar ── */}
      <div className="analytics-header">
        <div className="analytics-header-icon">📊</div>
        <div className="analytics-header-title">
          <h1>Analytics &amp; Reports</h1>
          <p>Power BI-style intelligence dashboard — all visuals on one page</p>
        </div>

        {/* Jump-to nav */}
        <nav className="pbi-jumpnav">
          {[['#kpi','KPIs'],['#pipeline','Pipeline'],['#features','Features'],['#eda','EDA'],
            ['#regulatory','Regulatory'],['#stats','Stats'],['#bias','Bias'],['#rl','RL']].map(([href,label])=>(
            <a key={href} href={href} className="pbi-jumpnav-item">{label}</a>
          ))}
        </nav>

        <div className="analytics-header-actions">
          <button className="analytics-btn secondary" onClick={loadRunList}>🔄 Refresh</button>
          {data&&<button className="analytics-btn secondary" onClick={()=>{
            const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
            const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
            a.download=`analytics_${selectedRun}.json`; a.click();
          }}>⬇️ Export</button>}
        </div>
      </div>

      <div className="analytics-content">

        {/* Run selector + gate banner */}
        <div className="run-selector-bar">
          <label htmlFor="run-select">Run:</label>
          <select id="run-select" className="run-select" value={selectedRun} onChange={e=>setSelectedRun(e.target.value)}>
            {runList.map((r,i)=><option key={i} value={r.run_id||r}>{r.label||r.run_id||r}</option>)}
          </select>
          {gateDec&&gateDec!=='—'&&<span className={`run-badge ${gateClass}`}>{gateClass==='pass'?'✅':gateClass==='warn'?'⚠️':'❌'} {gateDec}</span>}
          {loading&&<span style={{fontSize:'0.8rem',color:'#64748b'}}>Loading...</span>}
          {error&&<span style={{fontSize:'0.8rem',color:'#ef4444'}}>{error}</span>}
        </div>

        {/* ── Run provenance bar: when / what / how / domain ─────────────────── */}
        {data && !loading && (()=>{
          const ts       = data.timestamp;
          const fmtDate  = ts ? new Date(ts).toLocaleString('en-IN',{dateStyle:'medium',timeStyle:'short'}) : null;
          const dsId     = data.dataset_id;
          const srcKind  = data.source_kind;
          const domainTxt= domainListUsed?.length ? domainListUsed.map(d=>d.toUpperCase()).join(' + ')
                         : domainUsed ? domainUsed.toUpperCase() : null;
          const chips = [
            fmtDate  && { icon:'🕐', label:'Triggered', value: fmtDate },
            dsId     && { icon:'📁', label:'Dataset',   value: dsId },
            srcKind  && { icon:'🔌', label:'Source',    value: srcKind.replace('_',' ') },
            (nRows||nCols) && { icon:'📊', label:'Size', value: `${nRows.toLocaleString()} rows × ${nCols} cols` },
            domainTxt && { icon:'⚖️', label:'Domain',  value: domainTxt, highlight: true },
          ].filter(Boolean);

          if (!chips.length) return null;
          return (
            <div style={{
              display:'flex', flexWrap:'wrap', gap:'0.5rem',
              marginBottom:'1.5rem',
              padding:'0.7rem 1rem',
              background:'rgba(99,102,241,0.05)',
              border:'1px solid rgba(99,102,241,0.15)',
              borderRadius:'10px',
            }}>
              {chips.map((c,i)=>(
                <div key={i} style={{
                  display:'flex', alignItems:'center', gap:'0.35rem',
                  padding:'0.25rem 0.7rem',
                  borderRadius:'6px',
                  background: c.highlight ? 'rgba(99,102,241,0.12)' : 'rgba(255,255,255,0.04)',
                  border: c.highlight ? '1px solid rgba(99,102,241,0.3)' : '1px solid rgba(255,255,255,0.07)',
                  fontSize:'0.78rem',
                }}>
                  <span>{c.icon}</span>
                  <span style={{color:'#64748b', fontWeight:500}}>{c.label}:</span>
                  <span style={{color: c.highlight ? '#a5b4fc' : '#cbd5e1', fontWeight:600}}>{c.value}</span>
                </div>
              ))}
            </div>
          );
        })()}

        {/* ════════════════ SECTION 1: KPI CARDS ════════════════ */}
        <SectionHeader id="kpi" icon="📋" title="Key Performance Indicators"/>
        <div className="kpi-grid" style={{marginBottom:'2.5rem'}}>
          <KpiCard label="Dataset Size"     value={nRows?nRows.toLocaleString():'—'} sub={`${nCols} columns`} icon="📋" color="#6366f1" loading={loading}/>
          <KpiCard label="Confidence Score" value={confidence?`${(confidence*100).toFixed(1)}%`:'—'} sub="Pipeline confidence" icon="🎯" color="#10b981" loading={loading}/>
          <KpiCard label="Null Rate"        value={`${(nullPct*100).toFixed(1)}%`} sub="Overall missing" icon="❓" color={nullPct>0.2?'#ef4444':'#f59e0b'} loading={loading}/>
          <KpiCard label="Anomaly Rate"     value={`${(anomalyPct*100).toFixed(1)}%`} sub="Isolation Forest" icon="🔍" color={anomalyPct>0.06?'#ef4444':'#a78bfa'} loading={loading}/>
          <KpiCard label="Rules Passed"     value={regSummary.rules_passed??'—'} sub={`of ${regSummary.rules_total||0}`} icon="✅" color="#10b981" loading={loading}/>
          {modelM.roc_auc&&<KpiCard label="ROC-AUC" value={(+modelM.roc_auc).toFixed(4)} sub="Model" icon="📈" color="#3b82f6" loading={loading}/>}
          {modelM.f1&&<KpiCard label="F1 Score" value={(+modelM.f1).toFixed(4)} sub="Prec × Recall" icon="⚡" color="#ec4899" loading={loading}/>}
          <KpiCard label="PII Detected"     value={govSummary.pii_detected??'—'} sub={`${govSummary.redactions||0} redacted`} icon="🔒" color="#f43f5e" loading={loading}/>
          <KpiCard label="RL Episodes"      value={rlSummary.episode_count??'—'} sub={rlSummary.in_shadow_mode?'Shadow':'Active'} icon="🤖" color="#8b5cf6" loading={loading}/>
        </div>

        {/* Gate Decision Banner */}
        {gateDec&&gateDec!=='—'&&(
          <div style={{padding:'1rem 1.5rem',borderRadius:'10px',marginBottom:'2rem',display:'flex',alignItems:'center',gap:'0.75rem',
            background:gateClass==='pass'?'rgba(16,185,129,0.1)':gateClass==='warn'?'rgba(245,158,11,0.1)':'rgba(239,68,68,0.1)',
            border:`1px solid ${gateClass==='pass'?'rgba(16,185,129,0.3)':gateClass==='warn'?'rgba(245,158,11,0.3)':'rgba(239,68,68,0.3)'}`}}>
            <span style={{fontSize:'1.5rem'}}>{gateClass==='pass'?'✅':gateClass==='warn'?'⚠️':'❌'}</span>
            <div>
              <div style={{fontWeight:700,color:gateClass==='pass'?'#10b981':gateClass==='warn'?'#f59e0b':'#ef4444'}}>Gate Decision: {gateDec}</div>
              <div style={{fontSize:'0.8rem',color:'#64748b'}}>{insights.slice(0,2).join(' · ')}</div>
            </div>
          </div>
        )}

        {/* ════════════════ SECTION 2: PIPELINE HEALTH ════════════════ */}
        <SectionHeader id="pipeline" icon="⚙️" title="Pipeline Health"/>
        <div className="charts-grid-3" style={{marginBottom:'1.5rem'}}>
          <ChartCard title="🎯 Confidence Gauge" badge="Gauge"
            desc="A needle dial showing the overall pipeline confidence score from 0 (low) to 100% (high).">
            <GaugeChart value={confidence} min={0} max={1} label="Confidence" color="#10b981" height={155}/>
          </ChartCard>
          <ChartCard title="🔵 Data Coverage" badge="Donut"
            desc="Proportion of valid (non-null) vs missing field values across the entire dataset.">
            <DonutChart slices={donutSlices} height={155} innerText={`${((1-nullPct)*100).toFixed(0)}%\nvalid`}/>
            <div className="chart-legend-row">
              {donutSlices.map((s,i)=><span key={i} className="chart-legend-item"><span className="chart-legend-dot" style={{background:s.color}}/>{s.label}: {(s.value*100).toFixed(1)}%</span>)}
            </div>
          </ChartCard>
          <ChartCard title="⚖️ Rule Compliance" badge="Pie"
            desc="Share of regulatory/validation rules that passed, warned, or failed across this run.">
            <PieChart slices={pieSlices} height={155}/>
          </ChartCard>
        </div>

        {(funnelStages.length>0||stackedBars.length>1)&&(
          <div className="charts-grid" style={{marginBottom:'1.5rem'}}>
            {funnelStages.length>0&&(
              <ChartCard title="📦 Data Lineage Funnel" badge="Funnel"
                desc="Each stage shows how many rows remain after Bronze → Silver → Gold transformations; narrowing means row reduction.">
                <FunnelChart stages={funnelStages} height={170}/>
              </ChartCard>
            )}
            {stackedBars.length>1&&(
              <ChartCard title="📊 Row Count by Stage" badge="Stacked Bar"
                desc="X: pipeline stage  ·  Y: row count — stacked segments show retained vs dropped rows at each step.">
                <StackedBar bars={stackedBars} height={170} xLabel="Pipeline Stage" yLabel="Rows"/>
              </ChartCard>
            )}
          </div>
        )}

        {/* ════════════════ SECTION 3: MODEL PERFORMANCE ════════════════ */}
        {modelGroups.length>0&&(
          <>
            <SectionHeader id="model" icon="🤖" title="Model Performance"/>
            <ChartCard title="📈 Model Metrics Comparison" badge="Column Chart" style={{marginBottom:'2rem'}}
              desc="Side-by-side comparison of AutoML model scores — Y: metric score (0–1)  ·  X: metric name (AUC, F1, Accuracy).">
              <ColumnChart groups={modelGroups} labels={modelLabels} height={180} xLabel="Metric" yLabel="Score (0–1)"/>
            </ChartCard>
          </>
        )}

        {/* ════════════════ SECTION 4: FEATURE INTELLIGENCE ════════════════ */}
        {fiSorted.length>0&&(
          <>
            <SectionHeader id="features" icon="🏆" title="Feature Intelligence"/>
            <ChartCard title="🏆 Feature Importance" badge="Treemap + Bars" style={{marginBottom:'2rem'}}
              desc="Width of each tile (treemap) and bar length indicate predictive importance; wider/longer = more influential feature.">
              <TreemapChart items={treemapItems} height={110}/>
              <div style={{marginTop:'1rem'}}>
                {fiSorted.map(([feat,score])=><FiBar key={feat} label={feat} score={score} maxScore={fiMax}/>)}
              </div>
            </ChartCard>
          </>
        )}

        {/* ════════════════ SECTION 5: EDA – DISTRIBUTIONS ════════════════ */}
        {numEntries.length>0&&(
          <>
            <SectionHeader id="eda" icon="🔬" title="EDA — Column Distributions"/>
            <div className="charts-grid-4" style={{marginBottom:'2rem'}}>
              {numEntries.slice(0,8).map(([col,stats],ci)=>(
                <ChartCard key={col} title={`📊 ${col}`} badge={stats.skewness!=null?`skew ${(+stats.skewness||0).toFixed(2)}`:undefined}
                  desc={`Frequency distribution of '${col}'. X: value range  ·  Y: count of rows in that bin.`}>
                  {stats.histogram_bins?.length>0
                    ?<HistogramCanvas bins={stats.histogram_bins} counts={stats.histogram_counts} color={PALETTE[ci%PALETTE.length]} xLabel={col} yLabel="Count"/>
                    :<div style={{textAlign:'center',color:'#475569',padding:'1.5rem',fontSize:'0.78rem'}}>No data</div>}
                  <div style={{display:'flex',gap:'0.75rem',marginTop:'0.5rem',fontSize:'0.72rem',color:'#64748b',flexWrap:'wrap'}}>
                    {stats.mean!=null&&<span>μ={(+stats.mean).toFixed(2)}</span>}
                    {stats.std!=null&&<span>σ={(+stats.std).toFixed(2)}</span>}
                    {stats.null_pct!=null&&<span>∅={(+stats.null_pct*100).toFixed(1)}%</span>}
                  </div>
                </ChartCard>
              ))}
            </div>

            {/* Box plot */}
            {boxes.length>0&&(
            <ChartCard title="📦 Box & Whisker — All Columns" badge="Box Plot" style={{marginBottom:'2rem'}}
              desc="Each column shows min / Q1 / median / Q3 / max. Y: numeric value  ·  X: column name. Box = interquartile range; line = median.">
              <BoxPlot boxes={boxes} height={200} yLabel="Value"/>
            </ChartCard>
            )}
          </>
        )}

        {/* ════════════════ SECTION 6: EDA – CORRELATIONS ════════════════ */}
        {corrMatrix.length>1&&(
          <>
            <SectionHeader id="correlations" icon="🔗" title="EDA — Correlations"/>
            <div className="charts-grid" style={{marginBottom:'2rem'}}>
              <ChartCard title="🌡️ Correlation Heatmap" badge="Heatmap" wide
                desc="Colour intensity shows Pearson r between each pair of numeric columns. Dark blue = strong positive; red = strong negative.">
                <HeatmapChart matrix={corrMatrix} rowLabels={corrCols} colLabels={corrCols} height={Math.max(180,corrCols.length*28+28)}/>
              </ChartCard>
            </div>
            {corrData.length>0&&(
              <ChartCard title="🔗 Top Correlations" desc="Ranked list of the strongest pairwise linear relationships (|r|). Red = inverse; green = direct correlation.">
                <div className="analytics-table-wrap">
                  <table className="analytics-table">
                    <thead><tr><th>Column A</th><th>Column B</th><th>r</th><th>Strength</th></tr></thead>
                    <tbody>{corrData.sort((a,b)=>Math.abs(b.correlation??b.r??0)-Math.abs(a.correlation??a.r??0)).slice(0,12).map((c,i)=>{
                      const r=+(c.correlation??c.r??0),abs=Math.abs(r);
                      return(<tr key={i}><td>{c.col_a||c.a}</td><td>{c.col_b||c.b}</td>
                        <td className="cell-mono" style={{color:r>0?'#34d399':'#f87171'}}>{r.toFixed(4)}</td>
                        <td>{abs>0.7?'🔴 Strong':abs>0.4?'🟡 Moderate':'🟢 Weak'}</td></tr>);
                    })}</tbody>
                  </table>
                </div>
              </ChartCard>
            )}
          </>
        )}

        {/* ════════════════ SECTION 7: EDA – ANOMALY ANALYSIS ════════════════ */}
        {(scatterPts.length>1||bubbles.length>0)&&(
          <>
            <SectionHeader id="anomaly" icon="🔍" title="EDA — Anomaly Analysis"/>
            <div className="charts-grid" style={{marginBottom:'1.5rem'}}>
              {scatterPts.length>1&&(
                <div className="chart-card">
                  <div className="chart-card-header"><div className="chart-card-title">⚡ Mean vs Std Dev</div><span className="chart-card-badge">Scatter</span></div>
                  <ScatterPlot points={scatterPts} xLabel="Mean" yLabel="Std Dev" color="#6366f1" height={210}/>
                </div>
              )}
              {bubbles.length>0&&(
                <div className="chart-card">
                  <div className="chart-card-header"><div className="chart-card-title">🫧 Anomaly Bubbles</div><span className="chart-card-badge">Bubble</span></div>
                  <BubbleChart bubbles={bubbles} xLabel="Max Z-Score" yLabel="IF Score ×10" height={210}/>
                </div>
              )}
            </div>
          </>
        )}

        {/* Column Statistics Matrix */}
        {matrixRows.length>0&&(
          <ChartCard title="🗂️ Column Statistics Matrix" badge="Matrix" style={{marginBottom:'2rem'}}
            desc="Rows = dataset columns, Cells = statistical measures. Darker shade = higher magnitude. Quickly compare distribution shape across all features.">
            <MatrixTable rows={matrixRows} cols={['Mean','Std Dev','Skewness','Null %']} data={matrixData}/>
          </ChartCard>
        )}

        {/* ════════════════ SECTION 8: REGULATORY ════════════════ */}
        {(domainTiles.length>0||violations.length>0||domainEnforced)&&(
          <>
            <SectionHeader id="regulatory" icon="⚖️" title="Regulatory Compliance"/>

            {/* Domain enforced status banner — always shown when domain was active */}
            {domainEnforced&&(
              <div style={{
                display:'flex',alignItems:'center',gap:'0.85rem',
                padding:'0.85rem 1.25rem',borderRadius:'10px',marginBottom:'1.25rem',
                background: cleanPass
                  ? 'rgba(16,185,129,0.08)'
                  : violations.length>0 ? 'rgba(239,68,68,0.07)' : 'rgba(99,102,241,0.08)',
                border:`1px solid ${ cleanPass
                  ? 'rgba(16,185,129,0.25)'
                  : violations.length>0 ? 'rgba(239,68,68,0.2)' : 'rgba(99,102,241,0.2)' }`,
              }}>
                <span style={{fontSize:'1.6rem',lineHeight:1}}>{
                  cleanPass ? '✅' : violations.length>0 ? '❌' : '⚖️'
                }</span>
                <div>
                  <div style={{fontWeight:700,fontSize:'0.9rem',color:
                    cleanPass ? '#10b981' : violations.length>0 ? '#f87171' : '#818cf8'
                  }}>
                    {cleanPass
                      ? `✔️ All regulatory rules passed — ${domainUsed.toUpperCase()} compliance engine found zero violations`
                      : violations.length>0
                        ? `⚠️ ${violations.length} violation${violations.length!==1?'s':''} detected during ${domainUsed.toUpperCase()} compliance check`
                        : `⚖️ ${domainUsed.toUpperCase()} regulatory engine was active for this run`
                    }
                  </div>
                  <div style={{fontSize:'0.77rem',color:'#64748b',marginTop:'0.2rem'}}>
                    {domainListUsed.length>1
                      ? `Domains enforced: ${domainListUsed.map(d=>d.toUpperCase()).join(' + ')}`
                      : `Domain: ${domainUsed.toUpperCase()}${ cleanPass ? ' — dataset is fully compliant.' : '' }`
                    }&nbsp;&nbsp;·&nbsp;&nbsp;
                    Rules evaluated: {regSummary.rules_total||0}&nbsp;·&nbsp;
                    Passed: {regSummary.rules_passed||0}&nbsp;·&nbsp;
                    Warned: {regSummary.rules_warned||0}&nbsp;·&nbsp;
                    Failed: {regSummary.rules_failed||0}
                  </div>
                </div>
              </div>
            )}

            <div className="kpi-grid" style={{marginBottom:'1.5rem'}}>
              {/* Domain Active chip — always shown when a domain was enforced */}
              {domainEnforced&&<KpiCard
                label="Regulatory Domain"
                value={domainUsed ? domainUsed.toUpperCase() : '—'}
                sub={domainListUsed.length>1 ? `+${domainListUsed.slice(1).map(d=>d.toUpperCase()).join(', ')}` : 'active for this run'}
                icon="⚖️"
                color={cleanPass ? '#10b981' : '#8b5cf6'}
                loading={loading}
              />}
              <KpiCard label="Domains" value={(regSummary.domains_checked||[]).join(', ')||'—'} icon="🌐" color="#8b5cf6" loading={loading}/>
              <KpiCard label="Total Rules" value={regSummary.rules_total??'—'} icon="📋" color="#6366f1" loading={loading}/>
              <KpiCard label="Passed" value={regSummary.rules_passed??'—'} icon="✅" color="#10b981" loading={loading}/>
              <KpiCard label="Warnings" value={regSummary.rules_warned??'—'} icon="⚠️" color="#f59e0b" loading={loading}/>
              <KpiCard label="Failed" value={regSummary.rules_failed??'—'} icon="❌" color="#ef4444" loading={loading}/>
            </div>
            <div className="charts-grid" style={{marginBottom:'1.5rem'}}>
              {domainTiles.length>0&&(
                <ChartCard title="🗺️ Domain Compliance" badge="Shape Map"
                  desc="Each tile = a regulatory domain (GDPR, HIPAA…). Green = all rules passed; amber = warnings; red = violations found.">
                  <ShapeMap tiles={domainTiles}/>
                </ChartCard>
              )}
              <ChartCard title="📊 Pass / Warn / Fail Split" badge="Pie"
                desc="Overall share of regulatory rules that passed, generated warnings, or failed across all domains in this run.">
                <PieChart slices={pieSlices} height={200}/>
              </ChartCard>
            </div>
            {violations.length>0&&(
              <ChartCard title={`🚨 Violations (${violations.length})`} style={{marginBottom:'2rem'}}
                desc="Detailed list of every rule violation detected. Each entry shows the rule name, severity, description, and recommended remediation.">
                <div className="violation-list">{violations.map((v,i)=><ViolationItem key={i} {...v}/>)}</div>
              </ChartCard>
            )}
            {cleanPass&&(
              <div style={{
                textAlign:'center',padding:'1.25rem',borderRadius:'10px',
                background:'rgba(16,185,129,0.06)',border:'1px solid rgba(16,185,129,0.15)',
                color:'#34d399',fontSize:'0.85rem',marginBottom:'2rem',fontWeight:600,
              }}>
                ✅ Zero violations — all {domainUsed.toUpperCase()} regulatory rules evaluated and passed for this dataset.
              </div>
            )}
          </>
        )}


        {/* ════════════════ SECTION 9: STATISTICAL TESTS ════════════════ */}
        {(normTests.length>0||(statTests.stationarity||[]).length>0)&&(
          <>
            <SectionHeader id="stats" icon="📐" title="Statistical Tests"/>
            {normTests.length>0&&(
              <div style={{marginBottom:'1.5rem'}}>
                <div style={{fontSize:'0.82rem',fontWeight:700,color:'#64748b',marginBottom:'0.75rem',textTransform:'uppercase',letterSpacing:'0.05em'}}>Normality (Shapiro-Wilk)</div>
                <div className="stat-tests-grid">
                  {normTests.map((t,i)=>(
                    <div key={i} className="stat-test-card">
                      <div className="stat-test-col">{t.col}</div>
                      <div className="stat-test-vals">stat={t.statistic?.toFixed(4)} · p={t.p_value?.toFixed(4)}</div>
                      <div className={`stat-test-interp ${t.is_normal?'stat-test-pass':'stat-test-fail'}`}>
                        {t.is_normal?'✅ Normal':'⚠️ Non-normal'} — {t.interpretation}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {(statTests.stationarity||[]).length>0&&(
              <div style={{marginBottom:'2rem'}}>
                <div style={{fontSize:'0.82rem',fontWeight:700,color:'#64748b',marginBottom:'0.75rem',textTransform:'uppercase',letterSpacing:'0.05em'}}>Stationarity (ADF)</div>
                <div className="stat-tests-grid">
                  {(statTests.stationarity||[]).map((t,i)=>(
                    <div key={i} className="stat-test-card">
                      <div className="stat-test-col">{t.col}</div>
                      <div className="stat-test-vals">ADF={t.adf_stat?.toFixed(4)} · p={t.p_value?.toFixed(4)}</div>
                      <div className={`stat-test-interp ${t.is_stationary?'stat-test-pass':'stat-test-fail'}`}>
                        {t.is_stationary?'✅ Stationary':'⚠️ Non-stationary'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ════════════════ SECTION 10: BIAS & FAIRNESS ════════════════ */}
        {biasResults.length>0&&(
          <>
            <SectionHeader id="bias" icon="⚖️" title="Bias & Fairness Analysis"/>
            <div className="charts-grid" style={{marginBottom:'1.5rem'}}>
              <ChartCard title="🕸️ Fairness Radar" badge="Spider"
                desc="Each axis = one fairness metric (DI, TPR, FPR…). Filled area = actual values; radially symmetric = unbiased across all groups.">
                <RadarChart series={radarSeries} labels={radarLabels} height={230} centerLabel="Bias Metrics"/>
                <div className="chart-legend-row">
                  {radarSeries.map((s,i)=><span key={i} className="chart-legend-item"><span className="chart-legend-dot" style={{background:s.color}}/>{s.label}</span>)}
                </div>
              </ChartCard>
              {dumbbellRows.length>0&&(
                <ChartCard title="💪 Disparate Impact Range" badge="Dumbbell"
                  desc="Each row = one demographic group. Left dot = lowest DI score; right dot = highest. The 0.80 threshold marks fairness compliance.">
                  <DumbbellChart rows={dumbbellRows} height={160}/>
                  <div style={{fontSize:'0.72rem',color:'#475569',marginTop:'0.5rem',textAlign:'center'}}>
                    <span style={{color:'#ef4444'}}>● Low</span>&nbsp;&nbsp;<span style={{color:'#10b981'}}>● High</span>&nbsp;&nbsp;Disparate Impact
                  </div>
                </ChartCard>
              )}
            </div>
            <ChartCard title="⚖️ Bias Detail Table" style={{marginBottom:'2rem'}}
              desc="Full group-level breakdown: positive outcome rate per group, disparate impact ratio (< 0.80 = biased), and PASS/FAIL status.">
              <div className="analytics-table-wrap">
                <table className="analytics-table">
                  <thead><tr><th>Group</th><th>Value</th><th>Samples</th><th>Pos. Rate</th><th>Disparate Impact</th><th>Status</th></tr></thead>
                  <tbody>{biasResults.map((r,i)=>(
                    <tr key={i}><td>{r.group_col}</td><td>{r.group_value}</td><td>{r.sample_size?.toLocaleString()}</td>
                      <td className="cell-mono">{r.positive_rate?.toFixed(3)}</td>
                      <td className="cell-mono" style={{color:r.disparate_impact<0.8?'#ef4444':'#10b981'}}>{r.disparate_impact?.toFixed(3)}</td>
                      <td><span style={{padding:'0.15rem 0.5rem',borderRadius:'4px',fontSize:'0.72rem',fontWeight:700,
                        background:r.status==='PASS'?'rgba(16,185,129,0.15)':'rgba(239,68,68,0.15)',
                        color:r.status==='PASS'?'#10b981':'#ef4444'}}>{r.status}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </ChartCard>
          </>
        )}

        {/* ════════════════ SECTION 11: RL AGENT ════════════════ */}
        <SectionHeader id="rl" icon="🤖" title="RL Agent — PPO Training"/>
        <div className="kpi-grid" style={{marginBottom:'1.5rem'}}>
          <KpiCard label="Mode" value={rlSummary.in_shadow_mode?'Shadow':'Active PPO'} sub={rlSummary.in_shadow_mode?'Collecting data':'Deployed'} icon="🤖" color="#8b5cf6" loading={loading}/>
          <KpiCard label="Episodes" value={episodes||'—'} sub="of 20 for PPO" icon="📊" color="#6366f1" loading={loading}/>
          <KpiCard label="Last Reward" value={rlSummary.last_reward!=null?(+rlSummary.last_reward).toFixed(4):'—'} icon="🏆" color="#10b981" loading={loading}/>
        </div>
        <div className="charts-grid" style={{marginBottom:'1.5rem'}}>
          <ChartCard title="⭕ Shadow Mode Progress" badge="Progress Ring"
            desc="Ring fills as the agent completes shadow episodes. At 20 episodes the PPO policy activates and begins steering pipeline decisions.">
            <ProgressRing value={episodes} max={20} label="to PPO" subLabel={rlSummary.in_shadow_mode?`${20-episodes} more until active`:'Agent is live'} color="#8b5cf6" height={155}/>
          </ChartCard>
          <ChartCard title="📈 Reward History" badge="Line / Area"
            desc="X: episode number  ·  Y: cumulative reward score. Rising trend = agent improving; drops indicate bad pipeline outcomes used as negative feedback.">
            <LineAreaChart lines={rewardLines} height={155} xLabel="Episode" yLabel="Reward"/>
          </ChartCard>
        </div>
        {waterfallBars.length>1&&(
          <ChartCard title="💧 Reward Decomposition" badge="Waterfall" style={{marginBottom:'1.5rem'}}
            desc="Y: reward delta per agent decision. Green bars = positive contribution (good choices); red bars = penalty (bad choices). Final bar = net reward.">
            <WaterfallChart bars={waterfallBars} height={220} yLabel="Reward Delta"/>
          </ChartCard>
        )}
        {rlSummary.recommended_action&&(
          <ChartCard title="🎯 Policy Recommendation" style={{marginBottom:'2rem'}}
            desc="The RL agent's suggested configuration for the next pipeline run, derived from the highest-reward policy learned from prior episodes.">
            <div className="rl-stat-list">
              {Object.entries(rlSummary.recommended_action).map(([k,v])=>(
                <div key={k} className="rl-stat-row"><span className="rl-stat-label">{k.replace(/_/g,' ')}</span><span className="rl-stat-val highlight">{String(v)}</span></div>
              ))}
            </div>
          </ChartCard>
        )}

        {/* ════════════════ SECTION 12: KEY INSIGHTS ════════════════ */}
        {insights.length>0&&(
          <>
            <SectionHeader id="insights" icon="🔍" title="Key Findings & Insights"/>
            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(300px,1fr))',gap:'0.75rem',marginBottom:'3rem'}}>
              {insights.map((ins,i)=>(
                <div key={i} style={{padding:'0.85rem 1rem',background:'rgba(13,17,40,0.8)',border:'1px solid rgba(99,102,241,0.15)',
                  borderRadius:'10px',fontSize:'0.83rem',color:'#cbd5e1',lineHeight:1.5}}>
                  <span style={{color:'#818cf8',fontWeight:700,marginRight:'0.4rem'}}>{i+1}.</span>{ins}
                </div>
              ))}
            </div>
          </>
        )}

      </div>{/* /analytics-content */}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
//  MOCK DATA
// ═══════════════════════════════════════════════════════════════════════════════
const getMockData = () => ({
  run_id:'demo-run', gate_decision:'PASS', confidence_score:0.87, row_count:45231, col_count:23,
  timestamp: new Date(Date.now() - 12*60*1000).toISOString(),
  dataset_id:'banking_dataset_q1_2026.csv',
  source_kind:'file',

  summary:{n_rows:45231,n_cols:23,overall_null_pct:0.03,anomaly_pct:0.018},
  insights:[
    'income column is strongly right-skewed (skew=3.21) — log transform recommended',
    '3 pairs of strongly correlated features detected (r > 0.7)',
    'AML threshold: 142 transactions flagged above $10,000',
    'Model AUC 0.923 — excellent discriminative power',
    'Dataset passed all 12 banking regulatory rules',
    'PHI scan: 0 fields with unprotected personal identifiers',
  ],
  model_metrics:{roc_auc:0.9231,f1:0.8754,accuracy:0.9102},
  feature_importance:{income:0.18,credit_score:0.15,balance:0.12,transaction_amount:0.10,age:0.08,loan_amount:0.07,payment_history:0.06,debt_ratio:0.05,employment_years:0.04,num_accounts:0.03},
  regulatory_summary:{domains_checked:['banking','gdpr'],domain_used:'banking',domain_list_used:['banking','gdpr'],domain_enforced:true,rules_total:12,rules_passed:10,rules_warned:2,rules_failed:0},
  domain_compliance:[
    {label:'BANKING',status:'WARN',rules:8,passed:6},
    {label:'GDPR',status:'PASS',rules:4,passed:4},
  ],
  cross_domain_flags:[
    {
      rule_name:'aml_threshold', severity:'WARNING', domain:'banking',
      column:'transaction_amount', offending_count:142,
      type:'REGULATORY_VIOLATION',
      message:'[AML] 142 transaction(s) at or above the AML reporting threshold of 10,000.00. These require manual SAR review.',
      remediation:'Flag these transactions for Suspicious Activity Report (SAR) submission within the regulatory window (typically 30 days).',
    },
    {
      rule_name:'suspicious_transaction_pattern', severity:'WARNING', domain:'banking',
      column:'account_id', offending_count:7,
      type:'REGULATORY_VIOLATION',
      message:'[FATF Rec. 20] 7 account(s) exceed the velocity threshold of 50 transactions per day. May indicate structuring, layering, or automated fraud activity.',
      remediation:'Escalate flagged accounts to AML compliance team for manual review. File SAR if structuring pattern is confirmed.',
    },
    {
      rule_name:'currency_concentration', severity:'WARNING', domain:'banking',
      column:'currency', offending_count:42518,
      type:'REGULATORY_VIOLATION',
      message:'[BCBS239] Currency \'USD\' accounts for 94.0% of transactions, exceeding the concentration limit of 90%. High currency concentration reduces risk aggregation reliability.',
      remediation:'Diversify the dataset across multiple currencies, or ensure the concentration is intentional and documented in the risk framework.',
    },
  ],
  eda_report:{
    summary:{n_rows:45231,n_cols:23,overall_null_pct:0.03,anomaly_pct:0.018},
    numeric_stats:{
      income:        {mean:62400,std:28750,min:0,q1:38000,median:55000,q3:82000,max:350000,skewness:3.21,null_pct:0.01,histogram_bins:[0,35000,70000,105000,140000,175000,210000,245000,280000,315000,350000],histogram_counts:[8200,12400,9800,5600,3100,1800,900,450,230,120,50]},
      credit_score:  {mean:682,std:84,min:300,q1:620,median:690,q3:750,max:850,skewness:-0.42,null_pct:0.02,histogram_bins:[300,385,470,555,640,725,810,850],histogram_counts:[800,2400,5600,12000,14200,8900,3100,600]},
      balance:       {mean:15240,std:22100,min:0,q1:2800,median:8500,q3:21000,max:180000,skewness:2.87,null_pct:0.0,histogram_bins:[0,20000,40000,60000,80000,100000,120000,140000,160000,180000],histogram_counts:[28000,8500,4200,2100,1100,600,320,180,90,40]},
      transaction_amount:{mean:4820,std:9100,min:1,q1:250,median:1200,q3:4800,max:95000,skewness:4.52,null_pct:0.0,histogram_bins:[0,10000,20000,30000,40000,50000,60000,70000,80000,90000,95000],histogram_counts:[40200,3100,820,320,180,90,50,30,15,8,4]},
      age:           {mean:43.2,std:13.4,min:18,q1:33,median:43,q3:53,max:85,skewness:0.12,null_pct:0.0,histogram_bins:[18,28,38,48,58,68,78,85],histogram_counts:[3800,8200,11400,10200,7800,3200,900,200]},
      loan_amount:   {mean:22100,std:31400,min:0,q1:0,median:8000,q3:35000,max:250000,skewness:2.14,null_pct:0.15,histogram_bins:[0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000],histogram_counts:[22000,9800,4200,2100,1200,700,400,250,150,90,50]},
      payment_history:{mean:0.92,std:0.15,min:0,q1:0.88,median:0.96,q3:1.0,max:1.0,skewness:-2.1,null_pct:0.05,histogram_bins:[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],histogram_counts:[120,80,150,280,400,700,1200,2800,8500,14000,16000]},
      debt_ratio:    {mean:0.34,std:0.18,min:0,q1:0.20,median:0.32,q3:0.46,max:0.95,skewness:0.65,null_pct:0.0,histogram_bins:[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],histogram_counts:[2100,5800,8900,9200,7800,5400,3200,1800,800,300,100]},
    },
    correlations:[
      {col_a:'income',col_b:'credit_score',correlation:0.72},
      {col_a:'income',col_b:'balance',correlation:0.68},
      {col_a:'debt_ratio',col_b:'credit_score',correlation:-0.61},
      {col_a:'payment_history',col_b:'credit_score',correlation:0.81},
      {col_a:'loan_amount',col_b:'income',correlation:0.55},
      {col_a:'age',col_b:'credit_score',correlation:0.43},
      {col_a:'balance',col_b:'loan_amount',correlation:0.49},
      {col_a:'debt_ratio',col_b:'balance',correlation:-0.38},
      {col_a:'employment_years',col_b:'income',correlation:0.52},
    ],
  },
  statistical_tests:{
    normality:[
      {col:'income',statistic:0.7123,p_value:0.0001,is_normal:false,interpretation:'Strongly non-normal — log transform needed'},
      {col:'age',statistic:0.9812,p_value:0.042,is_normal:true,interpretation:'Approximately normal'},
      {col:'balance',statistic:0.8234,p_value:0.0003,is_normal:false,interpretation:'Right-skewed — sqrt transform'},
      {col:'credit_score',statistic:0.9645,p_value:0.031,is_normal:true,interpretation:'Mildly skewed, acceptable'},
      {col:'debt_ratio',statistic:0.9421,p_value:0.009,is_normal:false,interpretation:'Slightly non-normal'},
    ],
    stationarity:[
      {col:'transaction_date',adf_stat:-3.21,p_value:0.018,is_stationary:true},
      {col:'balance',adf_stat:-1.82,p_value:0.091,is_stationary:false},
    ],
  },
  bias_fairness_report:{results:[
    {group_col:'gender',group_value:'F',sample_size:18432,positive_rate:0.312,disparate_impact:0.89,status:'PASS'},
    {group_col:'gender',group_value:'M',sample_size:21543,positive_rate:0.350,disparate_impact:1.00,status:'PASS'},
    {group_col:'age_group',group_value:'18-25',sample_size:5231,positive_rate:0.198,disparate_impact:0.57,status:'FAIL'},
    {group_col:'age_group',group_value:'26-45',sample_size:22100,positive_rate:0.346,disparate_impact:0.99,status:'PASS'},
  ]},
  anomaly_deep_dive:{total_anomalies:814,if_contamination:0.018,per_column:[
    {col:'income',anomaly_count:312,z_score_max:8.41,if_score_mean:-0.21},
    {col:'transaction_amount',anomaly_count:289,z_score_max:6.72,if_score_mean:-0.18},
    {col:'balance',anomaly_count:213,z_score_max:5.31,if_score_mean:-0.15},
  ]},
  governance_summary:{pii_detected:0,redactions:0,governance_decision:'PASS'},
  data_lineage:{
    raw:{rows:48000,cols:28},bronze:{rows:46100,cols:26},silver:{rows:45800,cols:25},gold:{rows:45231,cols:23},
  },
  rl_agent_summary:{
    episode_count:7,in_shadow_mode:true,last_reward:0.7234,
    reward_history:[0.41,0.48,0.55,0.61,0.67,0.71,0.72],
    recommended_action:{cv_folds:5,cv_strategy:'stratified',confidence_threshold:0.70,imputation:'knn',outlier_policy:'winsorize',model_complexity:'high'},
    reward_components:{data_health_bonus:0.23,model_auc_bonus:0.21,pipeline_success_bonus:0.20,user_approval_bonus:0.0,drift_penalty:-0.01,total:0.7234},
  },
});

export default Analytics;
