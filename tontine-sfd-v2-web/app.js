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
  $('vue-parametres').hidden = (niv !== 'parametres');
  $('vue-doc').hidden = (niv !== 'doc');
  if (niv === 'investisseur' && !dernierMC) lancerSim();
  if (niv === 'parametres') renderParametres();
}));

// ============ VUE PARAMÈTRES (tous, interactifs) ============
// objet de config "maître" couvrant TOUS les paramètres (au-delà des 6 sliders investisseur)
let PARAMS = { ...DEFAUTS };
const SCHEMA = [
  { grp: '📐 Structure du cercle', desc: "La taille et la durée des pools.", items: [
    { k: 'n_pools', nom: 'Nombre de pools', d: "Combien de cercles tournent en parallèle (échelle du portefeuille).", t: 'range', min: 10, max: 100, step: 10 },
    { k: 'm_membres', nom: 'Membres par pool', d: "Taille d'un cercle. Détermine le pot = (M−1)×cotisation.", t: 'range', min: 6, max: 15, step: 1 },
    { k: 'c', nom: 'Cotisation mensuelle', d: "Ce que chaque membre verse chaque mois (XOF).", t: 'range', min: 25000, max: 200000, step: 25000, fmt: 'k' },
    { k: 'n_cycles', nom: 'Nombre de cycles', d: "Durée de vie du produit = M × cycles mois.", t: 'range', min: 1, max: 4, step: 1 },
    { k: 'k_max', nom: 'Max même secteur / pool', d: "Diversification : au plus K membres du même secteur par pool (limite la corrélation).", t: 'range', min: 1, max: 6, step: 1 },
  ]},
  { grp: '⚙️ Produit & mécanisme', desc: "Type de tontine et structure du coût.", items: [
    { k: 'mode', nom: 'Type de tontine', d: "Nue : sans garantie ni frais. Garantie : prime + couverture.", t: 'mode' },
    { k: 'prime_facteur_prudence', nom: 'Prudence de la prime', d: "1× = actuariel juste. >1 = prime majorée, plus robuste au stress (mais plus chère).", t: 'range', min: 0.8, max: 2.5, step: 0.1, fmt: 'x' },
    { k: 'prime_operateur_taux', nom: 'Marge Opérateur', d: "Marge de la plateforme, en % du pot, prélevée sur chaque encaissement.", t: 'range', min: 0, max: 0.04, step: 0.005, fmt: 'pct' },
    { k: 'r_sfd_annuel', nom: 'Taux SFD (avances)', d: "Taux annuel des avances de la SFD (intérêts du crédit-relais).", t: 'range', min: 0.06, max: 0.30, step: 0.02, fmt: 'pct' },
    { k: 'rho_mensuel', nom: 'Valeur-temps (ρ)', d: "Combien un membre pressé valorise l'accès anticipé → niveau des bids.", t: 'range', min: 0.005, max: 0.05, step: 0.005, fmt: 'pct' },
    { k: 'bid_plafond_frac_pot', nom: 'Plafond du surplus de bid', d: "Limite du surplus payé pour passer devant (compétitivité / usure).", t: 'range', min: 0.04, max: 0.25, step: 0.02, fmt: 'pct' },
  ]},
  { grp: '👥 Préférences de liquidité', desc: "Qui est pressé, qui est patient.", items: [
    { k: 'part_urgent', nom: 'Part d\'urgents', d: "Membres qui veulent leur argent tôt (bident le plus).", t: 'range', min: 0, max: 0.6, step: 0.05, fmt: 'pct' },
    { k: 'part_epargnant', nom: 'Part d\'épargnants', d: "Membres patients qui attendent (reçoivent gratuitement).", t: 'range', min: 0, max: 0.8, step: 0.05, fmt: 'pct' },
  ]},
  { grp: '🚪 Fuite & défaillance', desc: "Le risque que le modèle doit couvrir.", items: [
    { k: 'p_fuite_base', nom: 'Taux de fuite', d: "% des bénéficiaires qui disparaissent après avoir encaissé.", t: 'range', min: 0.02, max: 0.30, step: 0.02, fmt: 'pct' },
    { k: 'fuite_mult_tour_precoce', nom: 'Tentation tour précoce', d: "Multiplicateur de fuite au tour 1 (prendre tôt = plus tentant de fuir).", t: 'range', min: 1, max: 3, step: 0.2, fmt: 'x' },
    { k: 'charge_z_fuite', nom: 'Sensibilité macro', d: "À quel point un choc économique augmente les fuites (corrélation).", t: 'range', min: 0, max: 0.8, step: 0.05 },
    { k: 'taux_echec_friction', nom: 'Échec de prélèvement', d: "% de cotisations qui échouent temporairement (récupérable).", t: 'range', min: 0, max: 0.15, step: 0.01, fmt: 'pct' },
  ]},
  { grp: '🛡️ Couverture (3 étages)', desc: "Comment le trou d'une fuite est absorbé.", items: [
    { k: 'mitigation_active', nom: 'Mitigations', d: "Activer accès séquencé + garantie d'enchère + prélèvement auto.", t: 'bool' },
    { k: 't_restreint', nom: 'Tours réservés (historique)', d: "Les N premiers tours réservés aux membres avec historique.", t: 'range', min: 0, max: 5, step: 1 },
    { k: 'g_cotisations', nom: 'Consignation pour bider tôt', d: "Garantie (en nb de cotisations) saisie si fuite.", t: 'range', min: 0, max: 3, step: 1 },
    { k: 'fge_actif', nom: 'FGE (fonds de garantie)', d: "Le fonds endogène (primes + saisies) qui absorbe en premier.", t: 'bool' },
    { k: 'tranche_sfd_active', nom: 'Tranche SFD', d: "La SFD absorbe après le FGE (sa peau dans le jeu).", t: 'bool' },
    { k: 'plafond_tranche_sfd_frac', nom: 'Plafond tranche SFD', d: "Jusqu'où la SFD couvre, en % des avances. Au-delà = résiduel.", t: 'range', min: 0.01, max: 0.15, step: 0.01, fmt: 'pct' },
  ]},
  { grp: '🌩️ Stress', desc: "Tester le modèle en conditions dégradées.", items: [
    { k: 'comportemental_actif', nom: 'Stress comportemental', d: "Plus de fuites et plus de membres pressés.", t: 'bool' },
    { k: 'choc_fuite', nom: 'Choc de fuite', d: "Points de fuite ajoutés en stress comportemental.", t: 'range', min: 0, max: 0.15, step: 0.01, fmt: 'pct' },
    { k: 'macro_actif', nom: 'Stress macro', d: "Choc économique systémique (fuites corrélées).", t: 'bool' },
    { k: 'z_choc', nom: 'Sévérité du choc macro', d: "Ampleur du choc (négatif = mauvaise conjoncture).", t: 'range', min: -4, max: 0, step: 0.5 },
  ]},
];

function fmtParam(v, fmt) {
  if (fmt === 'k') return fmt0(v);
  if (fmt === 'pct') return (v * 100).toFixed(1).replace(/\.0$/, '') + '%';
  if (fmt === 'x') return v.toFixed(1) + '×';
  return (typeof v === 'number' && v % 1 !== 0) ? v.toFixed(2) : v;
}
const fmt0 = x => Math.round(x).toLocaleString('fr-FR');

function renderParametres() {
  $('paramGroupes').innerHTML = SCHEMA.map(g => `
    <div class="param-groupe">
      <h3>${g.grp}</h3>
      <div class="grp-desc">${g.desc}</div>
      ${g.items.map(it => paramRow(it)).join('')}
    </div>`).join('');
  // attacher les handlers
  SCHEMA.flatMap(g => g.items).forEach(it => attachParam(it));
}

function paramRow(it) {
  let ctrl = '';
  if (it.t === 'range') {
    ctrl = `<div class="p-ctrl"><input type="range" id="px_${it.k}" min="${it.min}" max="${it.max}" step="${it.step}" value="${PARAMS[it.k]}"><span class="p-val" id="pv_${it.k}">${fmtParam(PARAMS[it.k], it.fmt)}</span></div>`;
  } else if (it.t === 'bool') {
    ctrl = `<div class="p-toggle" id="px_${it.k}"><button data-v="1" class="${PARAMS[it.k] ? 'on' : ''}">Oui</button><button data-v="0" class="${!PARAMS[it.k] ? 'on' : ''}">Non</button></div>`;
  } else if (it.t === 'mode') {
    ctrl = `<div class="p-toggle" id="px_${it.k}"><button data-v="garantie" class="${PARAMS[it.k] === 'garantie' ? 'on' : ''}">Garantie</button><button data-v="nue" class="${PARAMS[it.k] === 'nue' ? 'on' : ''}">Nue</button></div>`;
  }
  return `<div class="param-row"><div><div class="p-nom">${it.nom}</div><div class="p-desc">${it.d}</div></div>${ctrl}<div></div></div>`;
}

function attachParam(it) {
  const el = $('px_' + it.k); if (!el) return;
  if (it.t === 'range') {
    el.addEventListener('input', e => { PARAMS[it.k] = +e.target.value; $('pv_' + it.k).textContent = fmtParam(PARAMS[it.k], it.fmt); $('paramStatus').textContent = '⟳ modifié — relancez pour voir l\'effet'; });
  } else {
    el.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
      el.querySelectorAll('button').forEach(b => b.classList.remove('on')); btn.classList.add('on');
      const v = btn.dataset.v; PARAMS[it.k] = (it.t === 'mode') ? v : (v === '1');
      $('paramStatus').textContent = '⟳ modifié — relancez pour voir l\'effet';
    }));
  }
}

$('btnRunParams') && $('btnRunParams').addEventListener('click', () => {
  // basculer vers l'onglet chiffres et lancer avec PARAMS
  document.querySelectorAll('.niv-btn').forEach(x => x.classList.toggle('active', x.dataset.niv === 'investisseur'));
  $('vue-public').hidden = true; $('vue-parametres').hidden = true; $('vue-doc').hidden = true; $('vue-investisseur').hidden = false;
  lancerSimAvec(PARAMS);
});
$('btnResetParams') && $('btnResetParams').addEventListener('click', () => { PARAMS = { ...DEFAUTS }; renderParametres(); $('paramStatus').textContent = 'valeurs par défaut restaurées'; });

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
  let p = SCEN[scen]({ ...DEFAUTS, n_pools: P.n_pools, m_membres: P.m_membres, c: P.c, p_fuite_base: P.p_fuite_base, prime_facteur_prudence: P.prime_facteur_prudence });
  lancerSimAvec(p, P._runs);
}
function lancerSimAvec(pBase, nRuns) {
  $('btnRun').textContent = 'Calcul…';
  setTimeout(() => {
    const p = { ...pBase };
    const a = monteCarlo(p, nRuns || P._runs, 12345);
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
    { l: 'P&L brut Opérateur', v: fmtM(a.pnlOp.moy), c: a.pnlOp.moy > 0 ? 'ok' : 'bad', s: `${p.n_pools} pools · hors coûts` },
    { l: 'Revenu brut / pool', v: fmtM(a.margePool.moy), c: 'brand', s: 'primes + surplus d\'enchère' },
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
