import { DINO_LEVEL } from './dinos.js';
import { TIER_LEVEL } from './tiere.js';
import { NATUR_LEVEL } from './natur.js';
import { WELTRAUM_LEVEL } from './weltraum.js';

/**
 * Vier Welten mit je 25 Level-Sets zu 5 Fragen = 500 Fragen.
 *
 * Die 100 Level laufen abwechselnd durch die Welten:
 * Level 1 Dino, 2 Tier, 3 Natur, 4 Weltraum, 5 Dino ...
 * So kommt nie Langeweile auf, und jede Welt wird genau 25 Mal gespielt.
 */

export const WELTEN = [
  {
    id: 'dinos',
    name: 'Dino-Welt',
    icon: '🦕',
    farbe: '#1b7f4b',
    saetze: DINO_LEVEL,
    themen: [
      'Der Tyrannosaurus Rex', 'Pflanzenfresser', 'Fleischfresser', 'Panzer und Stacheln',
      'Die Langhälse', 'Flugsaurier', 'Saurier im Meer', 'Eier und Dino-Babys',
      'Rekorde', 'Zähne und Fressen', 'Fossilien', 'Forscher und Museen',
      'Die Zeit der Dinos', 'Raptoren', 'Hörner und Kämme', 'Dino-Namen',
      'Federn und Vögel', 'Fußspuren', 'Wo die Dinos lebten', 'Das Ende der Dinos',
      'Kleine Dinos', 'Berühmte Funde', 'Sinne und Gehirn', 'Irrtümer über Dinos',
      'Großes Dino-Finale',
    ],
  },
  {
    id: 'tiere',
    name: 'Tier-Welt',
    icon: '🦊',
    farbe: '#e07a2b',
    saetze: TIER_LEVEL,
    themen: [
      'Haustiere', 'Auf dem Bauernhof', 'Im heimischen Wald', 'Vögel',
      'Insekten', 'Bienen und Ameisen', 'Am Fluss und im See', 'Im Meer',
      'Wale und Delfine', 'Afrika', 'Im Dschungel', 'In der Wüste',
      'Eis und Schnee', 'Australien', 'Tierkinder', 'Tierfamilien und Gruppen',
      'Winter und Wanderung', 'Tarnung und Verstecken', 'Schnell und langsam',
      'Riesen und Zwerge', 'Reptilien', 'Amphibien', 'Spinnen und Krabbeltiere',
      'Sinne der Tiere', 'Großes Tier-Finale',
    ],
  },
  {
    id: 'natur',
    name: 'Natur-Welt',
    icon: '🌳',
    farbe: '#2f8f4f',
    saetze: NATUR_LEVEL,
    themen: [
      'Die Jahreszeiten', 'Wetter', 'Wolken und Regen', 'Bäume',
      'Blumen und Blüten', 'Pilze', 'Der Wald', 'Wiese und Garten',
      'Wasser', 'Flüsse und Meere', 'Berge', 'Vulkane',
      'Steine und Erde', 'Der Boden lebt', 'Obst und Gemüse', 'Vom Korn zum Brot',
      'Blätter und Farben', 'Umwelt schützen', 'Müll und Recycling',
      'Energie aus der Natur', 'Tag und Nacht', 'Licht und Regenbogen', 'Luft',
      'Lebensräume', 'Großes Natur-Finale',
    ],
  },
  {
    id: 'weltraum',
    name: 'Weltraum',
    icon: '🚀',
    farbe: '#2f7fd1',
    saetze: WELTRAUM_LEVEL,
    themen: [
      'Die Sonne', 'Der Mond', 'Unsere Erde', 'Merkur und Venus',
      'Der Mars', 'Der Jupiter', 'Der Saturn', 'Uranus und Neptun',
      'Sterne', 'Sternbilder', 'Die Milchstraße', 'Unser Sonnensystem',
      'Raketen', 'Astronauten', 'Die Raumstation ISS', 'Die Mondlandung',
      'Satelliten', 'Kometen und Sternschnuppen', 'Asteroiden', 'Teleskope',
      'Schwerkraft', 'Tag, Jahr und Jahreszeiten', 'Mondphasen und Finsternisse',
      'Roboter im Weltraum', 'Großes Weltraum-Finale',
    ],
  },
];

export const FRAGEN_PRO_LEVEL = 5;
export const ANZAHL_LEVEL = 100;
/** Ab dieser Quote ist ein Level bestanden (4 von 5 Fragen). */
export const BESTEHENSQUOTE = 0.8;

/** Die 100 Level, abwechselnd aus den vier Welten. */
export const LEVEL = Array.from({ length: ANZAHL_LEVEL }, (_, i) => {
  const welt = WELTEN[i % WELTEN.length];
  const index = Math.floor(i / WELTEN.length);
  return {
    nr: i + 1,
    welt,
    index,
    thema: welt.themen[index],
    fragen: welt.saetze[index],
  };
});

export function getLevel(nr) {
  return LEVEL[Math.min(ANZAHL_LEVEL, Math.max(1, nr)) - 1];
}

/** Alle Level einer Welt (25 Stück, in Spielreihenfolge). */
export function levelDerWelt(weltId) {
  return LEVEL.filter((l) => l.welt.id === weltId);
}
