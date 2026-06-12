/* moteur.js — Port JavaScript du simulateur « tontine structurée avec SFD dépositaire ».
 *
 * Fidèle au moteur Python (config.py / risque.py / moteur.py / orchestrateur.py / pnl.py) :
 *   - facteur systémique unique Z + secteurs porteurs de charge rho (Vasicek mono-facteur)
 *   - composition contrainte-K réaliste (taux de respect mesuré)
 *   - split immédiat/différé, enchère valeur-temps endogène, bid = commission alpha + consignation
 *   - défauts conditionnels à Z, confiscations, garantie pot complet des disciplinés
 *   - cascade 4 niveaux : réserve pool -> méta-réserve inter-pools -> ligne SFD -> report
 *   - P&L Opérateur (break-even) + P&L SFD (rendement, perte bornée)
 *   - stress comportemental + macro, combinables
 *
 * Nommage neutre : Operateur / SFD / Membre / Pool.
 */

// ---------- RNG déterministe (mulberry32) + normale ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
// CDF normale standard (Abramowitz & Stegun)
function erf(x) {
  const s = Math.sign(x), ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return s * y;
}
function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
// Quantile normal (Acklam)
function PhiInv(p) {
  p = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p <= phigh) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}
function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q, base = Math.floor(pos), rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;

// ---------- Configuration par défaut (miroir de config.py) ----------
export const DEFAUTS = {
  // structure
  n_pools: 50, m_membres: 10, c: 10000, n_cycles: 2, k_max: 3,
  // mécanisme
  part_immediate: 0.50, alpha: 0.50, rho_mensuel: 0.02, bid_bruit_sigma: 0.25,
  // préférences (urgent / modéré / épargnant)
  part_urgent: 0.20, part_modere: 0.50, part_epargnant: 0.30,
  urg_urgent: 1.60, urg_modere: 1.00, urg_epargnant: 0.55,
  pdm_urgent: 1.35, pdm_modere: 1.00, pdm_epargnant: 0.70,
  // risque
  pd_base_annuel: 0.08, pd_base_sigma: 0.04,
  // secteurs : [nom, part, charge rho]
  secteurs: [["commerce", 0.30, 0.30], ["agriculture", 0.20, 0.45], ["transport", 0.15, 0.25], ["services", 0.20, 0.15], ["artisanat", 0.15, 0.20]],
  // SFD
  r_epargne_annuel: 0.04, r_credit_annuel: 0.24, r_replacement_annuel: 0.08, periodes_par_an: 12,
  ligne_defauts_pilote: 4,
  // bonus
  bonus_cycles_requis: 2, beta: 0.50,
  // cascade
  meta_reserve_active: true, ligne_sfd_active: true,
  // stress
  comportemental_actif: false, choc_pd: 0.0, bascule_urgents: 0.0,
  macro_actif: false, z_choc: 0.0, z_persistance: 0,
  // P&L Opérateur
  cout_acquisition_membre: 2000, cout_mm_taux: 0.01, cout_notif_membre: 20, cout_support_pool: 200,
  couts_fixes_mensuels: 500000, retrocession_sfd: 0.0,
};

// taux périodiques
const rPeriode = (annuel, ppa) => Math.pow(1 + annuel, 1 / ppa) - 1;

// ---------- Profils + composition K ----------
function tirerProfils(rng, n, p) {
  const ppa = p.periodes_par_an;
  const noms = p.secteurs.map(s => s[0]);
  let parts = p.secteurs.map(s => s[1]); const sp = parts.reduce((a, b) => a + b, 0); parts = parts.map(x => x / sp);
  const cum = []; let acc = 0; for (const x of parts) { acc += x; cum.push(acc); }
  const charges = {}; p.secteurs.forEach(s => charges[s[0]] = s[2]);
  const profils = [];
  for (let i = 0; i < n; i++) {
    const r = rng(); let si = 0; while (si < cum.length - 1 && r > cum[si]) si++;
    const sec = noms[si];
    let pdAnn = p.pd_base_annuel + gaussian(rng) * p.pd_base_sigma;
    pdAnn = Math.min(Math.max(pdAnn, 0.005), 0.60);
    const pdMens = 1 - Math.pow(1 - pdAnn, 1 / ppa);
    profils.push({ secteur: sec, rho: charges[sec], pdMens, seuil: PhiInv(pdMens) });
  }
  return profils;
}

function composerPools(rng, profils, m, kMax) {
  const n = profils.length, nPools = Math.floor(n / m);
  const pools = Array.from({ length: nPools }, () => []);
  const compteSec = Array.from({ length: nPools }, () => ({}));
  let violations = 0;
  // compter parts par secteur
  const parts = {}; profils.forEach(pr => parts[pr.secteur] = (parts[pr.secteur] || 0) + 1);
  const ordre = profils.map((_, i) => i).sort((i, j) => {
    const pi = parts[profils[i].secteur], pj = parts[profils[j].secteur];
    if (pi !== pj) return pj - pi;
    return profils[j].rho - profils[i].rho;
  });
  for (const i of ordre) {
    const sec = profils[i].secteur;
    const ouverts = []; for (let pp = 0; pp < nPools; pp++) if (pools[pp].length < m) ouverts.push(pp);
    if (!ouverts.length) continue;
    const conformes = ouverts.filter(pp => (compteSec[pp][sec] || 0) < kMax);
    let pCible;
    if (conformes.length) {
      pCible = conformes.reduce((best, pp) => (pools[pp].length < pools[best].length ? pp : best), conformes[0]);
    } else {
      pCible = ouverts.reduce((best, pp) => ((compteSec[pp][sec] || 0) < (compteSec[best][sec] || 0) ? pp : best), ouverts[0]);
      violations++;
    }
    pools[pCible].push(i);
    compteSec[pCible][sec] = (compteSec[pCible][sec] || 0) + 1;
  }
  // taux de respect
  let respectants = 0, nonVides = 0;
  for (const pool of pools) {
    if (!pool.length) continue; nonVides++;
    const cnt = {}; let mx = 0; pool.forEach(i => { const s = profils[i].secteur; cnt[s] = (cnt[s] || 0) + 1; mx = Math.max(mx, cnt[s]); });
    if (mx <= kMax) respectants++;
  }
  return { pools: pools.filter(p => p.length === m), tauxRespectK: nonVides ? respectants / nonVides : 1 };
}

function pdConditionnelle(seuil, rho, z) {
  return Phi((seuil - Math.sqrt(rho) * z) / Math.sqrt(Math.max(1e-9, 1 - rho)));
}

function tirerZ(rng, p, etat) {
  if (!p.macro_actif || p.z_choc === 0) return gaussian(rng);
  if (p.z_persistance > 0 && etat.restant > 0) { etat.restant--; return etat.courant; }
  const z = p.z_choc + gaussian(rng) * 0.6;
  if (p.z_persistance > 0) { etat.courant = z; etat.restant = p.z_persistance - 1; }
  return z;
}

// ---------- Simulation d'un portefeuille (un run) ----------
export function simulerPortefeuille(p, graine) {
  const rng = mulberry32(graine);
  const m = p.m_membres, totalTours = m * p.n_cycles;
  const poolMensuel = m * p.c, immediat = p.part_immediate * poolMensuel, differe = (1 - p.part_immediate) * poolMensuel;
  const rEpargne = rPeriode(p.r_epargne_annuel, p.periodes_par_an);
  const rCredit = rPeriode(p.r_credit_annuel, p.periodes_par_an);
  const rReplace = rPeriode(p.r_replacement_annuel, p.periodes_par_an);
  const beta = p.beta;

  // profils + pools
  const profils = tirerProfils(rng, p.n_pools * m, p);
  const { pools: poolsIdx, tauxRespectK } = composerPools(rng, profils, m, p.k_max);
  const nPools = poolsIdx.length;

  // assigner préférences
  function pref() {
    const r = rng();
    if (r < p.part_urgent) return { t: 'urgent', urg: p.urg_urgent, pdm: p.pdm_urgent };
    if (r < p.part_urgent + p.part_modere) return { t: 'modere', urg: p.urg_modere, pdm: p.pdm_modere };
    return { t: 'epargnant', urg: p.urg_epargnant, pdm: p.pdm_epargnant };
  }
  // états membres
  const etats = [];
  for (let pid = 0; pid < nPools; pid++) {
    const membres = poolsIdx[pid].map((gi, i) => {
      const pr = profils[gi], pf = pref();
      let urg = pf.urg, pdm = pf.pdm, type = pf.t;
      // stress comportemental : bascule
      if (p.comportemental_actif && (type === 'modere' || type === 'epargnant') && rng() < p.bascule_urgents) {
        type = 'urgent'; urg = p.urg_urgent; pdm = p.pdm_urgent;
      }
      return { i, seuil: pr.seuil, rho: pr.rho, urg, pdm, enDefaut: false, tourDefaut: null, aRecuPot: false, cyclesSansDefaut: 0, contribue: 0, recuImm: 0, recuDif: 0, consigne: 0, commission: 0, bonus: 0 };
    });
    etats.push(membres);
  }

  // Z par mois (commun)
  const etatZ = {}; const zMois = []; for (let t = 0; t < totalTours; t++) zMois.push(tirerZ(rng, p, etatZ));

  const reserves = new Array(nPools).fill(0);
  const provDiffere = new Array(nPools).fill(0);
  const ligneUtilisee = new Array(nPools).fill(0);
  // plafond ligne SFD par pool
  const expoUnit = p.part_immediate * poolMensuel / m;
  const plafondLigne = p.ligne_defauts_pilote * expoUnit;
  const chocPd = p.comportemental_actif ? p.choc_pd : 0;

  // trackers
  const tr = { n1: 0, n2: 0, n3: 0, n4: 0, m1: 0, m2: 0, m3: 0, m4: 0, perteSFD: 0 };
  let commissionTot = 0, totalBids = 0, nDefauts = 0, fondsBonus = 0;
  let interetsCredit = 0, sommeReserve = 0, sommeCollecte = 0, poolsActifsMois = 0;

  for (let t = 1; t <= totalTours; t++) {
    const z = zMois[t - 1], slot = (t - 1) % m;
    if (slot === 0) for (const membres of etats) for (const mb of membres) mb.aRecuPot = false;
    for (let pid = 0; pid < nPools; pid++) reserves[pid] *= (1 + rEpargne);

    const besoins = new Array(nPools).fill(0);
    const infos = new Array(nPools);

    for (let pid = 0; pid < nPools; pid++) {
      const membres = etats[pid];
      // défauts (vectorisé conceptuellement)
      for (const mb of membres) {
        if (mb.enDefaut) continue;
        const pd = Math.min(Math.max(pdConditionnelle(mb.seuil, mb.rho, z) * mb.pdm + chocPd, 0), 1);
        if (rng() < pd) {
          mb.enDefaut = true; mb.tourDefaut = t; nDefauts++;
          const conf = mb.consigne; mb.consigne = 0;
          reserves[pid] += (1 - beta) * conf; fondsBonus += beta * conf;
        }
      }
      const actifs = membres.filter(mb => !mb.enDefaut);
      let collecte = 0; for (const mb of actifs) { mb.contribue += p.c; collecte += p.c; }
      // bénéficiaire (ordre résiduel)
      const candidats = membres.filter(mb => !mb.enDefaut && !mb.aRecuPot).sort((a, b) => a.i - b.i);
      let benef = null;
      if (candidats.length) { benef = candidats[0]; benef.aRecuPot = true; }
      // enchère sur le différé
      const elig = membres.filter(mb => !mb.enDefaut && !mb.aRecuPot);
      let gagnant = null, bid = 0;
      if (elig.length) {
        const moisGagnes = Math.max(0, (m - 1) - slot);
        const baseFrac = p.rho_mensuel * moisGagnes;
        let best = -1;
        for (const mb of elig) {
          const bruit = Math.exp(gaussian(rng) * p.bid_bruit_sigma);
          const wtp = Math.min(Math.max(baseFrac * mb.urg * bruit, 0), 1) * differe;
          if (wtp > best) { best = wtp; gagnant = mb; }
        }
        bid = Math.max(0, best);
      }
      const commission = p.alpha * bid, consignation = (1 - p.alpha) * bid;
      totalBids += bid; commissionTot += commission;
      if (gagnant && bid > 0) { gagnant.commission += commission; gagnant.consigne += consignation; reserves[pid] += consignation; }

      let besoin = 0, decaisse = 0, niv = 0;
      if (benef) {
        decaisse = immediat; benef.recuImm += immediat;
        const surplus = collecte - immediat;
        if (surplus >= 0) provDiffere[pid] += surplus;
        else {
          const besoinImm = -surplus;
          const pris = Math.min(reserves[pid], besoinImm); reserves[pid] -= pris;
          if (pris > 0) { tr.m1 += pris; tr.n1++; niv = 1; }
          besoin = besoinImm - pris;
        }
      }
      besoins[pid] = besoin;
      infos[pid] = { collecte, decaisse, niv, nActifs: actifs.length };
    }

    // cascade besoins immédiats : niv2 méta, niv3 ligne, niv4 report
    appliquerCascade(besoins, reserves, ligneUtilisee, plafondLigne, tr, p, nPools, infos);

    // agrégats
    const dureeCredit = Math.max(1, m - slot);
    for (let pid = 0; pid < nPools; pid++) {
      if (infos[pid].decaisse > 0) interetsCredit += infos[pid].decaisse * rCredit * dureeCredit;
      sommeReserve += reserves[pid]; sommeCollecte += infos[pid].collecte;
      if (infos[pid].nActifs > 0) poolsActifsMois++;
    }

    // fin de cycle : restitution des différés (garantie pot complet)
    if (slot === m - 1) {
      const besoinsDif = new Array(nPools).fill(0);
      for (let pid = 0; pid < nPools; pid++) {
        const benefsDisc = etats[pid].filter(mb => !mb.enDefaut && mb.aRecuPot);
        const du = benefsDisc.length * differe;
        const dispo = provDiffere[pid] + reserves[pid];
        if (du <= dispo) {
          const reste = du - provDiffere[pid];
          provDiffere[pid] = Math.max(0, provDiffere[pid] - du);
          if (reste > 0) reserves[pid] -= reste;
          for (const mb of benefsDisc) mb.recuDif += differe;
        } else {
          provDiffere[pid] = 0; reserves[pid] = 0; besoinsDif[pid] = du - dispo;
          const couvert = dispo;
          for (const mb of benefsDisc) mb.recuDif += du > 0 ? differe * (couvert / du) : 0;
        }
      }
      appliquerCascade(besoinsDif, reserves, ligneUtilisee, plafondLigne, tr, p, nPools, null);
      for (let pid = 0; pid < nPools; pid++) for (const mb of etats[pid]) if (!mb.enDefaut) mb.cyclesSansDefaut++;
    }
  }

  // bonus inter-cycle
  const eligibles = [];
  for (let pid = 0; pid < nPools; pid++) for (const mb of etats[pid]) if (!mb.enDefaut && mb.cyclesSansDefaut >= p.bonus_cycles_requis) eligibles.push(mb);
  const totalConsElig = eligibles.reduce((s, mb) => s + mb.consigne, 0);
  let bonusDistribue = 0;
  if (eligibles.length && fondsBonus > 0) {
    for (const mb of eligibles) {
      const part = totalConsElig > 0 ? fondsBonus * (mb.consigne / totalConsElig) : fondsBonus / eligibles.length;
      mb.bonus += part; bonusDistribue += part;
    }
  }

  // agrégats finaux
  const nMembres = nPools * m;
  const reserveMoy = sommeReserve / totalTours, collecteMoy = sommeCollecte / totalTours;
  const depotsMoyens = reserveMoy + collecteMoy;

  // KPIs
  const nActifsFin = etats.flat().filter(mb => !mb.enDefaut).length;
  const tauxCompletion = nMembres ? nActifsFin / nMembres : 0;
  const actifsC1 = etats.flat().filter(mb => mb.tourDefaut === null || mb.tourDefaut > m).length;
  const retention = actifsC1 ? nActifsFin / actifsC1 : 1;
  const bonusMoyen = eligibles.length ? eligibles.reduce((s, mb) => s + mb.bonus, 0) / eligibles.length : 0;

  // P&L Opérateur
  const coutAcq = p.cout_acquisition_membre * nMembres;
  const fluxPool = poolMensuel + p.part_immediate * poolMensuel;
  const coutOpsUnit = p.cout_mm_taux * fluxPool + p.cout_notif_membre * m + p.cout_support_pool;
  const coutOps = coutOpsUnit * poolsActifsMois;
  const coutsFixes = p.couts_fixes_mensuels * totalTours;
  const retro = p.retrocession_sfd * interetsCredit;
  const revenusOp = commissionTot + retro;
  const pnlOp = revenusOp - (coutAcq + coutOps + coutsFixes);
  const margePool = (commissionTot - coutAcq - coutOps) / nPools;
  const breakEven = margePool > 1e-9 ? coutsFixes / margePool : Infinity;

  // P&L SFD
  const spread = depotsMoyens * (rReplace - rEpargne) * totalTours;
  const revenusSFD = interetsCredit + spread;
  const remuEpargne = depotsMoyens * rEpargne * totalTours;
  const coutRisque = tr.perteSFD;
  const pnlSFD = revenusSFD - coutRisque - remuEpargne;
  const rendementSFD = depotsMoyens > 1e-9 ? pnlSFD / depotsMoyens : 0;

  return {
    nPools, nMembres, tauxRespectK, nDefauts, tauxCompletion, retention, bonusMoyen,
    freq: { n1: tr.n1, n2: tr.n2, n3: tr.n3, n4: tr.n4 },
    montant: { n1: tr.m1, n2: tr.m2, n3: tr.m3, n4: tr.m4 },
    perteSFD: tr.perteSFD,
    pnlOp, commissionTot, margePool, breakEven, coutOps, coutsFixes, coutAcq,
    pnlSFD, rendementSFD, interetsCredit, spread, depotsMoyens,
    fondsBonus, bonusDistribue,
  };
}

function appliquerCascade(besoins, reserves, ligneUtilisee, plafondLigne, tr, p, nPools, infos) {
  const total = besoins.reduce((a, b) => a + b, 0);
  if (total <= 1e-9) return;
  // niv2 méta-réserve
  if (p.meta_reserve_active) {
    for (let pid = 0; pid < nPools; pid++) {
      if (besoins[pid] <= 1e-9) continue;
      let dispoAutres = 0; for (let q = 0; q < nPools; q++) if (q !== pid && reserves[q] > 0) dispoAutres += reserves[q];
      const a = Math.min(besoins[pid], dispoAutres);
      if (a > 0) {
        for (let q = 0; q < nPools; q++) if (q !== pid && reserves[q] > 0) reserves[q] -= a * (reserves[q] / dispoAutres);
        besoins[pid] -= a; tr.m2 += a; tr.n2++; if (infos) infos[pid].niv = 2;
      }
    }
  }
  // niv3 ligne SFD
  if (p.ligne_sfd_active) {
    for (let pid = 0; pid < nPools; pid++) {
      if (besoins[pid] <= 1e-9) continue;
      const dispo = Math.max(0, plafondLigne - ligneUtilisee[pid]);
      const tire = Math.min(besoins[pid], dispo);
      if (tire > 0) { ligneUtilisee[pid] += tire; besoins[pid] -= tire; tr.m3 += tire; tr.n3++; tr.perteSFD += tire; if (infos) infos[pid].niv = 3; }
    }
  }
  // niv4 report
  for (let pid = 0; pid < nPools; pid++) {
    if (besoins[pid] > 1e-9) { tr.m4 += besoins[pid]; tr.n4++; if (infos) infos[pid].niv = 4; besoins[pid] = 0; }
  }
}

// ---------- Monte Carlo ----------
export function monteCarlo(p, nRuns, graineBase) {
  const cles = ['tauxCompletion', 'retention', 'nDefauts', 'tauxRespectK', 'perteSFD', 'pnlOp', 'breakEven', 'pnlSFD', 'rendementSFD', 'bonusMoyen', 'commissionTot'];
  const freqCles = ['n1', 'n2', 'n3', 'n4'];
  const acc = {}; cles.forEach(k => acc[k] = []); freqCles.forEach(k => acc['freq_' + k] = []);
  const pertes = [];
  for (let i = 0; i < nRuns; i++) {
    const r = simulerPortefeuille(p, graineBase + i);
    cles.forEach(k => { const v = r[k]; if (isFinite(v)) acc[k].push(v); });
    freqCles.forEach(k => acc['freq_' + k].push(r.freq[k]));
    pertes.push(r.perteSFD);
  }
  const agg = {};
  Object.keys(acc).forEach(k => {
    const s = acc[k].slice().sort((a, b) => a - b);
    agg[k] = { moyenne: mean(acc[k]), p5: quantile(s, 0.05), p95: quantile(s, 0.95) };
  });
  agg._pertes = pertes;
  return agg;
}
