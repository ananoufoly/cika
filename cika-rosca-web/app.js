import { DEFAULTS, runPath, runMonteCarlo, slotSchedule, potP, dividendShare } from './engine.js';

const fmt = (x) => Math.round(x).toLocaleString('en-US');
const fmtK = (x) => Math.abs(x) >= 1e6 ? (x/1e6).toFixed(2)+'M' : Math.abs(x)>=1e3 ? Math.round(x/1e3)+'k' : Math.round(x).toString();
const pct = (x) => (x*100).toFixed(x*100<1?1:0)+'%';
const $ = (id) => document.getElementById(id);

// ---- state ----
let P = { ...DEFAULTS };

// ---- tabs ----
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
}));

// ====================================================================
// 1 · FLOW DIAGRAM
// ====================================================================
const flowSteps = [
  { title:'Contributions', sub:'12 members × 25,000', key:'C',
    desc:'Every member pays their monthly contribution. The pot for this turn is N × c.' },
  { title:'Auction', sub:'bid for the slot', key:'auction',
    desc:'Members bid (forgo a discount) to take the pot now. Each slot has an auto-solved minimum price.' },
  { title:'Allocation', sub:'fraction now + deferred', key:'alloc',
    desc:'The winner takes an immediate slice; the rest is held in escrow, unlocked only if they stay to the end.' },
  { title:'Reserve', sub:'absorbs shocks · placed with bank', key:'reserve',
    desc:'The discount + surplus build a reserve that covers missed payments and is placed with the bank treasury between payouts.' },
  { title:'Distribution', sub:'shared with members', key:'dist',
    desc:'The treasury distribution flows back: the majority to members, a share to CIKA. Discipline finishes ahead.' },
];
let flowIdx = 0;
function renderFlow() {
  const pot = potP(P);
  const sch = slotSchedule(P);
  const s0 = sch[0];
  const vals = {
    C: fmt(pot)+' FCFA',
    auction: 'slot 1 asks '+pct(s0.askFrac),
    alloc: fmt(s0.immediate)+' now / '+fmt(s0.deferred)+' held',
    reserve: '~'+fmtK(s0.discount)+' to reserve',
    dist: pct(P.member_yield_share)+' to members',
  };
  $('flowDiagram').innerHTML = flowSteps.map((s,i)=>`
    <div class="node ${i===flowIdx?'active':i<flowIdx?'done':''}">
      <h4>${i+1}. ${s.title}</h4>
      <small>${s.sub}</small>
      <div class="v">${vals[s.key]}</div>
    </div>`).join('');
  $('flowStepLabel').textContent = flowSteps[flowIdx].desc;
}
$('flowNext').addEventListener('click',()=>{flowIdx=(flowIdx+1)%flowSteps.length;renderFlow();});
$('flowPrev').addEventListener('click',()=>{flowIdx=(flowIdx-1+flowSteps.length)%flowSteps.length;renderFlow();});

// ====================================================================
// 2 · SIMULATOR CONTROLS
// ====================================================================
const bindings = [
  ['cN','oN','N', v=>v],
  ['cc','oc','c', v=>fmt(v)],
  ['ccyc','ocyc','num_cycles', v=>v+' ('+(P.N*v)+' mo)'],
  ['cyld','oyld','invest_yield_annual', v=>pct(v)],
  ['cms','oms','member_yield_share', v=>pct(v)],
  ['cline','oline','bank_line', v=>fmt(v)],
  ['cpb','opb','_pb', v=>pct(v)],
  ['cpaths','opaths','_paths', v=>v],
];
function syncControls() {
  $('cN').value=P.N; $('cc').value=P.c; $('ccyc').value=P.num_cycles;
  $('cyld').value=P.invest_yield_annual; $('cms').value=P.member_yield_share;
  $('cline').value=P.bank_line; $('cpb').value=(P.p_lo+P.p_hi)/2; $('cpaths').value=P._paths||800;
  updateOutputs();
}
function updateOutputs() {
  $('oN').textContent=P.N; $('oc').textContent=fmt(P.c);
  $('ocyc').textContent=P.num_cycles+' ('+(P.N*P.num_cycles)+' mo)';
  $('oyld').textContent=pct(P.invest_yield_annual); $('oms').textContent=pct(P.member_yield_share);
  $('oline').textContent=fmt(P.bank_line); $('opb').textContent=pct((P.p_lo+P.p_hi)/2);
  $('opaths').textContent=P._paths||800;
}
$('cN').addEventListener('input',e=>{P.N=+e.target.value;updateOutputs();renderFlow();renderSlots();});
$('cc').addEventListener('input',e=>{P.c=+e.target.value;updateOutputs();renderFlow();renderSlots();});
$('ccyc').addEventListener('input',e=>{P.num_cycles=+e.target.value;updateOutputs();});
$('cyld').addEventListener('input',e=>{P.invest_yield_annual=+e.target.value;updateOutputs();});
$('cms').addEventListener('input',e=>{P.member_yield_share=+e.target.value;updateOutputs();});
$('cline').addEventListener('input',e=>{P.bank_line=+e.target.value;updateOutputs();});
$('cpb').addEventListener('input',e=>{const v=+e.target.value;P.p_lo=Math.max(0.5,v-0.035);P.p_hi=Math.min(0.999,v+0.035);updateOutputs();});
$('cpaths').addEventListener('input',e=>{P._paths=+e.target.value;updateOutputs();});
$('resetBtn').addEventListener('click',()=>{P={...DEFAULTS,_paths:800};syncControls();renderFlow();renderSlots();renderMoney(lastMC);});
$('runBtn').addEventListener('click',runSim);

let lastMC = null, lastPath = null;
function runSim() {
  $('runBtn').textContent='Running…';
  setTimeout(()=>{
    const nPaths = P._paths || 800;
    lastMC = runMonteCarlo(P, nPaths, 100);
    // representative path near the median discipline pBase
    lastPath = runPath(P, 12345, (P.p_lo+P.p_hi)/2);
    renderKPIs(lastMC);
    drawHistogram(lastMC.memberNets, P);
    drawReserve(lastPath, P);
    renderMoney(lastMC);
    $('runBtn').textContent='▶ Run simulation';
  }, 30);
}

function renderKPIs(mc) {
  const items = [
    { lbl:'Failure rate', val:pct(mc.pInsolv), cls: mc.pInsolv<=0.001?'ok':mc.pInsolv<0.05?'warn':'neg', sub:'P[reserve+line exhausted]' },
    { lbl:'Loyal member (median net)', val:(mc.loyalMedian>=0?'+':'')+fmt(mc.loyalMedian), cls:mc.loyalMedian>=0?'pos':'neg', sub:mc.loyalPctPos.toFixed(0)+'% finish ahead' },
    { lbl:'Defection (max exit net)', val:(mc.exitMax>0?'+':'')+fmt(mc.exitMax), cls:mc.exitMax<=1?'ok':'neg', sub:'quitting never pays ⇒ ≤ 0' },
    { lbl:'CIKA per circle', val:'+'+fmt(mc.fintechMean), cls:'pos', sub:'AUM-style, scales w/ volume' },
    { lbl:'Bank per circle', val:'+'+fmt(mc.bankMgmtPerCircle+mc.bankFeePerCircle), cls:'ok', sub:'treasury fee + line fees' },
    { lbl:'Bank line drawn (p95)', val:fmt(mc.drawP95), cls:mc.drawP95<5000?'ok':'warn', sub:'rarely touched' },
    { lbl:'Avg reserve placed', val:fmtK(mc.avgInvested), cls:'', sub:'treasury balance under mgmt' },
    { lbl:'Member distribution / circle', val:'+'+fmt(mc.yMemberPerCircle), cls:'pos', sub:'shared to members' },
  ];
  $('kpiGrid').innerHTML = items.map(i=>`
    <div class="kpi"><div class="lbl">${i.lbl}</div>
      <div class="val ${i.cls}">${i.val}</div><small>${i.sub}</small></div>`).join('');
}

// ---- histogram (canvas) ----
function drawHistogram(nets, p) {
  const cv=$('histCanvas'), ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
  const data = nets.filter(x=>x>-3*p.c*p.num_cycles*p.N); // trim extreme outliers for readability
  if(!data.length) return;
  const lo=Math.min(...data), hi=Math.max(...data);
  const bins=40, w=(hi-lo)/bins||1, counts=new Array(bins).fill(0);
  data.forEach(v=>{let b=Math.floor((v-lo)/w);b=Math.max(0,Math.min(bins-1,b));counts[b]++;});
  const maxc=Math.max(...counts);
  const floorVal=-p.loss_cap_months*p.c;
  const pad=30, bw=(W-pad*2)/bins;
  // zero line x
  const zx=pad+((0-lo)/(hi-lo))*(W-pad*2);
  counts.forEach((c,i)=>{
    const x=pad+i*bw, h=(c/maxc)*(H-40);
    const binMid=lo+(i+0.5)*w;
    ctx.fillStyle = binMid>=0?'#22c55e':(binMid<=floorVal+w?'#f59e0b':'#ef4444');
    ctx.fillRect(x,H-20-h,bw-1,h);
  });
  // zero axis
  ctx.strokeStyle='#9aa7bd'; ctx.setLineDash([4,4]); ctx.beginPath();ctx.moveTo(zx,8);ctx.lineTo(zx,H-18);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#9aa7bd';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('break-even',zx,H-4);
  ctx.textAlign='left';ctx.fillText(fmtK(lo),pad,H-4);
  ctx.textAlign='right';ctx.fillText(fmtK(hi),W-4,H-4);
}

// ---- reserve path (canvas) ----
function drawReserve(path, p) {
  const cv=$('reserveCanvas'), ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
  const rows=path.rows; const pad=34;
  const maxR=Math.max(...rows.map(r=>r.R),...rows.map(r=>r.Ldrawn),1);
  const xs=i=>pad+(i/(rows.length-1))*(W-pad-8);
  const ys=v=>H-20-(v/maxR)*(H-30);
  // reserve area
  ctx.beginPath();ctx.moveTo(xs(0),ys(0));
  rows.forEach((r,i)=>ctx.lineTo(xs(i),ys(r.R)));
  ctx.lineTo(xs(rows.length-1),ys(0));ctx.closePath();
  ctx.fillStyle='rgba(0,194,168,.18)';ctx.fill();
  ctx.strokeStyle='#00c2a8';ctx.lineWidth=2;ctx.beginPath();
  rows.forEach((r,i)=>i?ctx.lineTo(xs(i),ys(r.R)):ctx.moveTo(xs(i),ys(r.R)));ctx.stroke();
  // bank line drawn
  ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.beginPath();
  rows.forEach((r,i)=>i?ctx.lineTo(xs(i),ys(r.Ldrawn)):ctx.moveTo(xs(i),ys(r.Ldrawn)));ctx.stroke();
  // axes
  ctx.fillStyle='#9aa7bd';ctx.font='11px sans-serif';ctx.textAlign='left';
  ctx.fillText(fmtK(maxR),4,16);ctx.fillText('0',4,H-20);
  ctx.textAlign='center';ctx.fillText('month →',W/2,H-4);
}

// ====================================================================
// 3 · MONEY
// ====================================================================
function renderMoney(mc) {
  if(!mc) return;
  const bankTotal = mc.bankMgmtPerCircle + mc.bankFeePerCircle;
  const cards = [
    { cls:'member', who:'Members (collectively)', amt:'+'+fmt(mc.yMemberPerCircle),
      bullets:['Lump-sum allocation on their turn','Share of treasury distribution ('+pct(P.member_yield_share)+')','Capped downside, gains from forfeited exits','Disciplined median: '+(mc.loyalMedian>=0?'+':'')+fmt(mc.loyalMedian)] },
    { cls:'cika', who:'CIKA (platform)', amt:'+'+fmt(mc.fintechMean),
      bullets:['Share of treasury distribution','Optional auction fee','Scales with number of circles','No extraction from member pots'] },
    { cls:'bank', who:'Bank partner', amt:'+'+fmt(bankTotal),
      bullets:['Treasury management fee','Liquidity-line commitment + draw fees','Near-zero credit risk','Sticky low-cost balances'] },
  ];
  $('moneyGrid').innerHTML = cards.map(c=>`
    <div class="money-card ${c.cls}">
      <div class="who">${c.who}</div>
      <div class="amt">${c.amt}</div>
      <div class="hint">per circle / ${P.num_cycles*P.N} months</div>
      <ul>${c.bullets.map(b=>`<li>${b}</li>`).join('')}</ul>
    </div>`).join('');
  renderScaled(mc);
}
function renderScaled(mc) {
  const n = +($('cCircles').value||1000);
  $('oCircles').textContent=fmt(n);
  const cika=mc.fintechMean*n, bank=(mc.bankMgmtPerCircle+mc.bankFeePerCircle)*n, members=mc.yMemberPerCircle*n;
  $('scaledOut').innerHTML = `
    <div class="s"><div class="hint">CIKA total</div><div class="a" style="color:var(--accent)">${fmtK(cika)}</div></div>
    <div class="s"><div class="hint">Bank total</div><div class="a" style="color:var(--gold)">${fmtK(bank)}</div></div>
    <div class="s"><div class="hint">Members total</div><div class="a" style="color:var(--pos)">${fmtK(members)}</div></div>`;
}
$('cCircles').addEventListener('input',()=>{ if(lastMC) renderScaled(lastMC); });

// ====================================================================
// 4 · SLOTS
// ====================================================================
function renderSlots() {
  const sch = slotSchedule(P);
  const tb = $('slotTable').querySelector('tbody');
  tb.innerHTML = sch.map(r=>`
    <tr>
      <td>#${r.slot}</td>
      <td>${pct(r.escrow)}</td>
      <td>${pct(r.askFrac)} · ${fmt(r.discount)}</td>
      <td>${fmt(r.immediate)}</td>
      <td>${fmt(r.deferred)}</td>
      <td>${r.lag} mo</td>
      <td class="${r.exitNet<=1?'ok':'neg'}">${r.exitNet>0?'+':''}${fmt(r.exitNet)}</td>
    </tr>`).join('');
}

// ====================================================================
// 5 · OPTIONS
// ====================================================================
const optionDefs = [
  { cls:'core', tag:'Core — required', title:'Treasury manager', bullets:[
      'Bank places the idle reserve (T-bills / term placement — bank chooses instrument & yield)',
      'Earns a management fee on balances under management',
      'This is the engine that makes members finish ahead' ],
    risk:'Bank risk: ~none (managing balances, not lending).' },
  { cls:'', tag:'Optional — cheap insurance', title:'Liquidity backstop', bullets:[
      'Committed senior line (~1× pot) drawn only when the reserve is exhausted',
      'Reserve takes first loss; line is senior and capped',
      'Drawn ≈ 0 in 95% of paths — converts "~95% safe" into a guarantee' ],
    risk:'Bank risk: capped at the line, rarely drawn.' },
  { cls:'', tag:'Optional — growth', title:'Distribution & trust anchor', bullets:[
      'Bank lends name + branch network',
      '"Bank-secured circle" is a major trust signal in UEMOA',
      'Customer acquisition & cross-sell for the bank' ],
    risk:'Bank risk: none — reputational alignment only.' },
];
function renderOptions() {
  $('optionsGrid').innerHTML = optionDefs.map(o=>`
    <div class="opt ${o.cls}">
      <span class="tag">${o.tag}</span>
      <h3>${o.title}</h3>
      <ul>${o.bullets.map(b=>`<li>${b}</li>`).join('')}</ul>
      <div class="risk">${o.risk}</div>
    </div>`).join('');
}
$('compareBtn').addEventListener('click',()=>{
  const base = {...P};
  const configs = [
    { name:'A · Treasury only (no line)', over:{ bank_line:0 } },
    { name:'B · Treasury + line (recommended)', over:{ bank_line:300000 } },
    { name:'C · Treasury + larger line', over:{ bank_line:600000 } },
  ];
  $('compareBtn').textContent='Running…';
  setTimeout(()=>{
    const tb=$('compareTable').querySelector('tbody');
    tb.innerHTML = configs.map(c=>{
      const pp={...base,...c.over};
      const mc=runMonteCarlo(pp, 600, 100);
      return `<tr>
        <td>${c.name}</td>
        <td class="${mc.pInsolv<=0.001?'ok':mc.pInsolv<0.05?'':'neg'}">${pct(mc.pInsolv)}</td>
        <td class="${mc.loyalMedian>=0?'pos':'neg'}">${mc.loyalMedian>=0?'+':''}${fmt(mc.loyalMedian)}</td>
        <td class="pos">+${fmt(mc.fintechMean)}</td>
        <td class="ok">+${fmt(mc.bankMgmtPerCircle+mc.bankFeePerCircle)}</td>
        <td>${fmt(mc.drawP95)}</td>
      </tr>`;
    }).join('');
    $('compareBtn').textContent='Run comparison';
  },30);
});

// ====================================================================
// INIT
// ====================================================================
P._paths = 800;
syncControls();
renderFlow();
renderSlots();
renderOptions();
renderOptions();
$('buildNote').textContent = 'engine v1 · '+P.N+' members · '+P.num_cycles+' cycles default';
runSim();
