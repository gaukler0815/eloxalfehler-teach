/*
 * leaderboard.js — local high score list.
 *
 * Rules from the Game Design Bible:
 *  - Name entry after finishing, max 12 chars, uppercase, no account/mail.
 *  - Mandatory name filter (block list); duplicate names get a running number.
 *  - Top 10 visible, the player's own entry always shown with its real rank.
 *  - Two tabs: today's list and the all-time list.
 * Stored in localStorage; a shared trade-show server can replace this later.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  const KEY = 'eloxal-rebels.scores.v1';
  // Small block list; extend as needed. Matched case-insensitively as substring.
  const BLOCKLIST = ['ARSCH', 'SCHEISS', 'FICK', 'FOTZE', 'NAZI', 'HITLER', 'SEX', 'HURE'];

  function load() {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || [];
    } catch (e) {
      return [];
    }
  }

  function save(list) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list));
    } catch (e) {
      /* storage full / disabled — leaderboard is best effort */
    }
  }

  // Clean a raw name to the allowed form. Returns '' if nothing usable remains.
  function sanitizeName(raw) {
    let n = (raw || '').toUpperCase();
    n = n.replace(/[^A-ZÄÖÜ0-9 ]/g, ''); // letters, digits, spaces only
    n = n.replace(/\s+/g, ' ').trim();
    n = n.slice(0, 12);
    return n;
  }

  function isBlocked(name) {
    const flat = name.replace(/\s/g, '');
    return BLOCKLIST.some((bad) => flat.indexOf(bad) !== -1);
  }

  // Ensure the name is unique by appending a running number if needed.
  function uniqueName(name, list) {
    const taken = new Set(list.map((e) => e.name));
    if (!taken.has(name)) return name;
    for (let i = 2; i < 1000; i++) {
      const suffix = ' ' + i;
      const base = name.slice(0, 12 - suffix.length);
      const candidate = base + suffix;
      if (!taken.has(candidate)) return candidate;
    }
    return name;
  }

  function todayKey(ts) {
    const d = new Date(ts);
    return d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate();
  }

  /**
   * Add a score. Returns { entry, rankTotal, rankToday } or { error }.
   */
  function submit(rawName, um) {
    const name = sanitizeName(rawName);
    if (!name) return { error: 'Bitte einen Namen eingeben.' };
    if (isBlocked(name)) return { error: 'Dieser Name ist nicht erlaubt.' };

    const list = load();
    const finalName = uniqueName(name, list);
    const entry = { name: finalName, um: um | 0, ts: Date.now() };
    list.push(entry);
    save(list);

    return {
      entry,
      rankTotal: rankOf(entry, table('total')),
      rankToday: rankOf(entry, table('today'))
    };
  }

  // Build a sorted table. scope: 'total' | 'today'.
  function table(scope) {
    let list = load();
    if (scope === 'today') {
      const tk = todayKey(Date.now());
      list = list.filter((e) => todayKey(e.ts) === tk);
    }
    list.sort((a, b) => b.um - a.um || a.ts - b.ts); // higher µm first, earlier wins ties
    return list;
  }

  function rankOf(entry, sorted) {
    const i = sorted.findIndex((e) => e.ts === entry.ts && e.name === entry.name);
    return i >= 0 ? i + 1 : null;
  }

  /**
   * A view for rendering: top N plus the player's own row with its real rank.
   */
  function view(scope, ownEntry, topN) {
    const sorted = table(scope);
    const top = sorted.slice(0, topN || 10).map((e, i) => ({ rank: i + 1, ...e }));
    let own = null;
    if (ownEntry) {
      const r = rankOf(ownEntry, sorted);
      if (r) own = { rank: r, ...ownEntry, isOwn: true };
    }
    return { top, own };
  }

  function clear() {
    save([]);
  }

  ER.leaderboard = { submit, table, view, sanitizeName, isBlocked, clear };
})(typeof window !== 'undefined' ? window : globalThis);
