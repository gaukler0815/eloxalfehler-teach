/**
 * Baustein fuer alle Fragen.
 *
 * WICHTIG: Die erste Antwort in der Liste ist immer die richtige.
 * Die App mischt die Reihenfolge vor dem Anzeigen - dadurch kann beim
 * Schreiben der 500 Fragen kein falscher Index passieren, und Linnea kann
 * sich auch nicht merken, dass "immer B" stimmt.
 *
 * @param {string} frage
 * @param {[string, string, string, string]} antworten  [richtig, falsch, falsch, falsch]
 * @param {string} info  Ein Satz, der nach dem Antworten erklaert.
 */
export function f(frage, antworten, info) {
  return { frage, antworten, info };
}
