import { DEFAUTS, simulerPoolDetail, monteCarlo } from './moteur.js';

const fmt = x => Math.round(x).toLocaleString('fr-FR');
const fmtM = x => Math.abs(x) >= 1e6 ? (x / 1e6).toFixed(1) + 'M' : Math.round(x / 1e3) + 'k';
const pct = x => (x * 100).toFixed(x * 100 < 1 && x > 0 ? 1 : 0) + '%';
const $ = id => document.getElementById(id);

let P = { ...DEFAUTS, _runs: 80 };
let mode = "garantie";

// ---- navigation niveaux ----
document.querySelectorAll('.niv-btn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.niv-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  const niv = b.dataset.niv;
  $('vue-public').hidden = (niv !== 'public');
  $('vue-investisseur').hidden = (niv !== 'investisseur');
  if (niv === 'investisseur' && !dernierMC) lancerSim();
}));

// ============ ANIMATION GRAND PUBLIC ============
const MODE_HINTS = {
  nue: "Tontine classique : si quelqu'un disparaît avec l'argent, le groupe perd. Pas de frais, mais pas de sécurité.",
  garantie: "Votre tour est sécurisé : même si un membre disparaît, vous recevez votre dû. Ceux qui veulent passer tôt paient ; ceux qui attendent reçoivent gratuitement.",
};
let anim = null, animIdx = 0, playTimer = null;

function genererAnim() {
  const p = { ...P, mode, n_pools: 1, m_membres: 10, c: 100000, n_cycles: 1 };
  // choisir une graine qui produit au moins une fuite, pour la démo
  let g = 7;
  for (let tryG = 1; tryG < 60; tryG++) {
    const test = simulerPoolDetail(p, tryG);
    if (test.journal.some(j => j.fuite)) { g = tryG; break; }
  }
  anim = simulerPoolDetail(p, g);
  animIdx = 0;
  renderAnim();
}

function renderAnim() {
  if (!anim) return;
  const j = anim.journal[animIdx];
  $('moisNum').textContent = j.tour;
  // membres
  $('cercle').innerHTML = j.snap.map(s => {
    let cls = 'membre';
    if (s.aFui) cls += ' fui';
    else if (s.estBenef) cls += ' benef';
    else if (s.aEncaisse) cls += ' enc';
    else if (s.type === 'urgent') cls += ' urg';
    else cls += ' pat';
    let etat = '';
    if (s.aFui) etat = 'parti';
    else if (s.estBenef) etat = '← reçoit';
    else if (s.aEncaisse) etat = 'a reçu';
    return `<div class="${cls}"><div class="av">${s.nom[0]}</div><div class="nom">${s.nom}</div><div class="etat">${etat}</div></div>`;
  }).join('');
  // pot
  if (j.benef) {
    $('potBenef').textContent = j.benef.nom;
    $('potMontant').textContent = fmt(j.benef.recu);
    if (mode === 'garantie' && j.benef.bid > 0) {
      const det = j.benef.bideur
        ? `${j.benef.nom} a payé ${fmt(j.benef.bid)} pour passer tôt (intérêts + sécurité). Reçoit ${fmt(j.benef.recu)}.`
        : `${j.benef.nom} attend son tour : il reçoit ${fmt(j.benef.recu)} en payant peu (${fmt(j.benef.bid)}).`;
      $('potDetail').textContent = det;
    } else if (mode === 'garantie') {
      $('potDetail').textContent = `${j.benef.nom} a attendu son tour : il reçoit tout, gratuitement (épargne).`;
    } else {
      $('potDetail').textContent = `${j.benef.nom} reçoit le pot. Tontine simple : aucun frais.`;
    }
  } else {
    $('potBenef').textContent = '—'; $('potMontant').textContent = '0'; $('potDetail').textContent = '';
  }
  // événement
  const evDiv = $('evenement');
  if (j.fuite) {
    if (mode === 'garantie' && j.garantie && j.garantie.residuel < 1) {
      evDiv.className = 'evenement garantie';
      evDiv.innerHTML = `⚠️ <b>${j.fuite.nom} disparaît</b> après avoir reçu son argent (trou de ${fmt(j.fuite.trou)}). → La <b>garantie comble le trou</b> : les suivants sont servis normalement. <b>Personne ne perd.</b>`;
    } else if (mode === 'garantie') {
      evDiv.className = 'evenement fuite';
      evDiv.innerHTML = `⚠️ <b>${j.fuite.nom} disparaît</b> (trou ${fmt(j.fuite.trou)}). La garantie est dépassée — cas extrême et rare.`;
    } else {
      evDiv.className = 'evenement fuite';
      evDiv.innerHTML = `❌ <b>${j.fuite.nom} disparaît</b> avec l'argent (${fmt(j.fuite.trou)}). En tontine simple, <b>le groupe subit la perte</b> — il n'y a aucune protection.`;
    }
  } else {
    evDiv.className = 'evenement';
    evDiv.innerHTML = j.benef ? `${j.benef.nom} reçoit son tour. Tout se passe normalement.` : `Mois ${j.tour}.`;
  }
  // badge promesse
  const casse = (mode === 'nue' && j.fuite) || (j.garantie && j.garantie.residuel > 1);
  const badge = $('promesseBadge');
  if (mode === 'nue') {
    const fuiteAvant = anim.journal.slice(0, animIdx + 1).some(x => x.fuite);
    badge.textContent = fuiteAvant ? '✕ groupe exposé' : '— sans garantie';
    badge.className = 'promesse-badge' + (fuiteAvant ? ' casse' : '');
  } else {
    badge.textContent = casse ? '✕ promesse menacée' : '✓ tous servis';
    badge.className = 'promesse-badge' + (casse ? ' casse' : '');
  }
  $('animStep').textContent = `mois ${j.tour} / ${anim.journal.length}`;
}

$('btnNext').onclick = () => { if (animIdx < anim.journal.length - 1) { animIdx++; renderAnim(); } };
$('btnPrev').onclick = () => { if (animIdx > 0) { animIdx--; renderAnim(); } };
$('btnReset').onclick = () => { stopPlay(); animIdx = 0; renderAnim(); };
$('btnPlay').onclick = () => {
  if (playTimer) { stopPlay(); return; }
  $('btnPlay').textContent = '⏸ Pause';
  if (animIdx >= anim.journal.length - 1) animIdx = 0;
  playTimer = setInterval(() => {
    if (animIdx < anim.journal.length - 1) { animIdx++; renderAnim(); }
    else stopPlay();
  }, 1400);
};
function stopPlay() { clearInterval(playTimer); playTimer = null; $('btnPlay').textContent = '▶ Dérouler'; }

document.querySelectorAll('#modeSeg .seg-btn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#modeSeg .seg-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); mode = b.dataset.mode;
  $('modeHint').textContent = MODE_HINTS[mode];
  stopPlay(); genererAnim(); renderExplications();
}));

function renderExplications() {
  const ex = mode === 'garantie' ? [
    { t: 'Vous attendez ?', d: "Vous recevez votre tour <b>gratuitement</b> — c'est votre épargne, on ne vous prend rien." },
    { t: 'Vous êtes pressé ?', d: "Vous pouvez <b>payer pour passer tôt</b> (comme un petit crédit). Ce paiement finance la sécurité de tous." },
    { t: 'Quelqu\'un disparaît ?', d: "La <b>garantie comble le trou</b>. Les autres sont servis quand même. <b>Vous ne perdez jamais votre tour.</b>" },
  ] : [
    { t: 'Pas de frais', d: "La tontine simple ne coûte rien — c'est la rotation classique." },
    { t: 'Mais aucune sécurité', d: "Si quelqu'un part avec l'argent après l'avoir reçu, <b>le groupe perd</b>." },
    { t: 'Le risque est sur vous', d: "Vous comptez sur la confiance. La <b>tontine sécurisée</b> enlève ce risque." },
  ];
  $('explications').innerHTML = ex.map(e => `<div class="ex"><b>${e.t}</b><br>${e.d}</div>`).join('');
}

// ============ INVESTISSEUR ============
const SCEN = {
  nominal: p => p,
  comportemental: p => { p.comportemental_actif = true; p.choc_fuite = 0.06; p.bascule_urgents = 0.30; return p; },
  macro: p => { p.macro_actif = true; p.z_choc = -2.5; p.z_persistance = 4; return p; },
  combine: p => { p.comportemental_actif = true; p.choc_fuite = 0.06; p.bascule_urgents = 0.30; p.macro_actif = true; p.z_choc = -2.5; p.z_persistance = 4; return p; },
  bids_faibles: p => { p.part_urgent = 0.05; p.part_modere = 0.25; p.part_epargnant = 0.70; p.rho_mensuel = 0.01; return p; },
};
const SCEN_HINT = {
  nominal: "Conditions normales.", comportemental: "Plus de fuites (+6 pts) et de membres pressés.",
  macro: "Choc économique : fuites corrélées.", combine: "Comportemental + macro ensemble — le test le plus dur.",
  bids_faibles: "Peu de membres enchérissent (population d'épargnants).",
};
let scen = 'nominal', dernierMC = null;

const sliders = [['c_np', 'o_np', 'n_pools', v => v], ['c_m', 'o_m', 'm_membres', v => v], ['c_c', 'o_c', 'c', v => fmt(v)], ['c_pf', 'o_pf', 'p_fuite_base', v => (v * 100).toFixed(0) + '%'], ['c_pp', 'o_pp', 'prime_facteur_prudence', v => v.toFixed(1) + '×'], ['c_runs', 'o_runs', '_runs', v => v]];
function syncS() { sliders.forEach(([c, o, k]) => $(c).value = P[k]); updO(); }
function updO() { sliders.forEach(([c, o, k, f]) => $(o).textContent = f(P[k])); }
sliders.forEach(([c, o, k]) => $(c).addEventListener('input', e => { P[k] = +e.target.value; updO(); }));
document.querySelectorAll('#scenSeg .seg-btn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#scenSeg .seg-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); scen = b.dataset.scen; $('scenHint').textContent = SCEN_HINT[scen]; lancerSim();
}));
$('btnRun').onclick = lancerSim;

function lancerSim() {
  $('btnRun').textContent = 'Calcul…';
  setTimeout(() => {
    let p = SCEN[scen]({ ...DEFAUTS, n_pools: P.n_pools, m_membres: P.m_membres, c: P.c, p_fuite_base: P.p_fuite_base, prime_facteur_prudence: P.prime_facteur_prudence });
    const a = monteCarlo(p, P._runs, 12345);
    dernierMC = a;
    renderKPIs(a, p);
    drawExpo(a, p);
    drawCout(p);
    $('btnRun').textContent = '▶ Lancer';
  }, 20);
}

function renderKPIs(a, p) {
  const pot = (p.m_membres - 1) * p.c;
  const items = [
    { l: 'Promesse tenue', v: pct(a.taux_continuite), c: a.taux_continuite >= 0.999 ? 'ok' : 'bad', s: 'tous les tours servis' },
    { l: 'P&L Opérateur', v: fmtM(a.pnlOp.moy), c: a.pnlOp.moy > 0 ? 'ok' : 'bad', s: `${p.n_pools} pools` },
    { l: 'Marge / pool', v: fmtM(a.margePool.moy), c: 'brand', s: 'rentable dès le 1ᵉʳ pool' },
    { l: 'Risque porté SFD', v: fmtM(a.perteSfd.moy), c: 'brand', s: 'avances non récupérées' },
    { l: 'Coût membre tour 1', v: pct(a.coutTour1.moy / pot), c: a.coutTour1.moy / pot < 0.2 ? 'ok' : 'brand', s: 'le dernier tour ≈ 0' },
    { l: 'Fuites moyennes', v: Math.round(a.fuites.moy), c: 'brand', s: 'bénéficiaires disparus' },
  ];
  $('kpiGrid').innerHTML = items.map(i => `<div class="kpi"><div class="lbl">${i.l}</div><div class="val ${i.c}">${i.v}</div><small>${i.s}</small></div>`).join('');
}

function drawExpo(a, p) {
  const cv = $('expoCanvas'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
  const prof = a.expoProfil; if (!prof) return;
  const mx = Math.max(...prof, 1), pad = 36;
  const xs = i => pad + (i / (prof.length - 1)) * (W - pad - 8), ys = v => H - 22 - (v / mx) * (H - 36);
  ctx.beginPath(); ctx.moveTo(xs(0), ys(0)); prof.forEach((v, i) => ctx.lineTo(xs(i), ys(v))); ctx.lineTo(xs(prof.length - 1), ys(0)); ctx.closePath(); ctx.fillStyle = 'rgba(15,76,74,.12)'; ctx.fill();
  ctx.strokeStyle = '#0f4c4a'; ctx.lineWidth = 2; ctx.beginPath(); prof.forEach((v, i) => i ? ctx.lineTo(xs(i), ys(v)) : ctx.moveTo(xs(i), ys(v))); ctx.stroke();
  ctx.fillStyle = '#5b6b87'; ctx.font = '11px sans-serif'; ctx.fillText(fmtM(mx), 4, 14); ctx.fillText('0', 4, H - 22); ctx.textAlign = 'center'; ctx.fillText('mois →', W / 2, H - 4);
}

function drawCout(p) {
  const cv = $('coutCanvas'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
  const m = p.m_membres, pot = (m - 1) * p.c, rSfd = p.r_sfd_annuel / 12;
  const couts = [];
  for (let slot = 0; slot < m; slot++) {
    const duree = Math.max(1, m - (slot + 1)), avance = Math.max(0, pot - slot * p.c);
    const interets = pot * rSfd * duree, prime = avance <= 0 ? 0 : p.prime_facteur_prudence * p.p_fuite_base * (avance / 2) * (duree / (m - 1)), marge = p.prime_operateur_taux * pot;
    couts.push((interets + prime + marge) / pot * 100);
  }
  const mx = Math.max(...couts, 20), pad = 30, bw = (W - pad * 2) / m;
  couts.forEach((cv2, i) => { const x = pad + i * bw, h = (cv2 / mx) * (H - 40); ctx.fillStyle = '#0f4c4a'; ctx.fillRect(x + 4, H - 24 - h, bw - 8, h); ctx.fillStyle = '#5b6b87'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('T' + (i + 1), x + bw / 2, H - 10); ctx.fillText(cv2.toFixed(0) + '%', x + bw / 2, H - 28 - h); });
  // seuil 15%
  const ys = v => H - 24 - (v / mx) * (H - 40); ctx.strokeStyle = '#b18a3a'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad, ys(15)); ctx.lineTo(W - pad, ys(15)); ctx.stroke(); ctx.setLineDash([]);
}

// ---- init ----
$('modeHint').textContent = MODE_HINTS.garantie;
$('scenHint').textContent = SCEN_HINT.nominal;
syncS();
genererAnim();
renderExplications();
