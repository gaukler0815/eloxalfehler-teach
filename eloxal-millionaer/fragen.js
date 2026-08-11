/*
 * Fragenkatalog "Wer wird Eloxal-Millionär"
 * --------------------------------------------------
 * Jede Frage hat:
 *   f    = Frage (String)
 *   a    = Array mit 4 Antworten
 *   r    = Index der richtigen Antwort (0-3)
 *   e    = Erklärung (wird nach der Antwort gezeigt)
 *   k    = Kategorie
 *   s    = Schwierigkeit: 1 = leicht, 2 = mittel, 3 = schwer
 *
 * Der Katalog ist bewusst groß gehalten, damit pro Runde
 * zufällig gezogen werden kann und nicht immer dieselben
 * Fragen einer Kategorie/Stufe erscheinen.
 */

const FRAGENKATALOG = [

  /* ============================================================
   *  STUFE 1 – LEICHT (Grundlagen)
   * ============================================================ */
  {
    f: "Wofür steht der Begriff „Eloxal“?",
    a: ["Elektrolytische Oxidation von Aluminium", "Elektronische Legierung aus Aluminium",
        "Elastische Oberflächen-Lackierung", "Elektrisches Löten von Aluminium"],
    r: 0, k: "Grundlagen", s: 1,
    e: "Eloxal = ELektrolytische OXidation von ALuminium. Dabei wird eine schützende Oxidschicht erzeugt."
  },
  {
    f: "Welches Metall wird beim klassischen Eloxal-Verfahren behandelt?",
    a: ["Kupfer", "Aluminium", "Eisen", "Zink"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Eloxieren ist ein Verfahren speziell für Aluminium und seine Legierungen."
  },
  {
    f: "Die beim Eloxieren entstehende Schutzschicht besteht hauptsächlich aus …",
    a: ["Aluminiumnitrid", "Aluminiumhydroxid", "Aluminiumoxid", "Aluminiumcarbid"],
    r: 2, k: "Grundlagen", s: 1,
    e: "Es bildet sich Aluminiumoxid (Al₂O₃), eine sehr harte und korrosionsbeständige Schicht."
  },
  {
    f: "Wächst die Eloxalschicht auf dem Bauteil auf oder wandelt sie das Material um?",
    a: ["Sie wird wie Farbe aufgetragen", "Sie entsteht durch Umwandlung des Grundmaterials",
        "Sie wird aufgeklebt", "Sie wird aufgedampft"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Die Oxidschicht wächst aus dem Aluminium heraus – etwa zur Hälfte nach innen und zur Hälfte nach außen."
  },
  {
    f: "Welche Eigenschaft hat eine eloxierte Oberfläche typischerweise NICHT?",
    a: ["Höhere Härte", "Besserer Korrosionsschutz", "Elektrische Leitfähigkeit der Schicht", "Bessere Einfärbbarkeit"],
    r: 2, k: "Grundlagen", s: 1,
    e: "Die Oxidschicht ist ein elektrischer Isolator – deshalb muss vor dem Eloxieren gut kontaktiert werden."
  },
  {
    f: "In welchem Zustand liegt das Elektrolyt beim Standard-Eloxieren vor?",
    a: ["Fest", "Flüssig (Säurebad)", "Gasförmig", "Als Pulver"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Eloxiert wird in einem flüssigen Elektrolyten, üblicherweise verdünnter Schwefelsäure."
  },
  {
    f: "Welche Säure wird am häufigsten als Elektrolyt beim Eloxieren verwendet?",
    a: ["Salzsäure", "Schwefelsäure", "Salpetersäure", "Essigsäure"],
    r: 1, k: "Anodisation", s: 1,
    e: "Das GS-Verfahren (Gleichstrom-Schwefelsäure) mit Schwefelsäure ist das mit Abstand gebräuchlichste."
  },
  {
    f: "Das Werkstück wird beim Eloxieren als … geschaltet.",
    a: ["Kathode (Minuspol)", "Anode (Pluspol)", "Neutralleiter", "Erdung"],
    r: 1, k: "Anodisation", s: 1,
    e: "Das Aluminium wird als Anode (Pluspol) geschaltet – daher der Name „Anodisieren“."
  },
  {
    f: "Warum werden Aluminiumfelgen oder Fassaden oft eloxiert?",
    a: ["Um sie schwerer zu machen", "Für Korrosionsschutz und Optik",
        "Um sie magnetisch zu machen", "Um sie brennbar zu machen"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Eloxal schützt vor Korrosion, erhöht die Härte und ermöglicht dekorative Farben."
  },
  {
    f: "Kann man eloxiertes Aluminium einfärben?",
    a: ["Nein, niemals", "Ja, die poröse Schicht nimmt Farbstoff auf",
        "Nur mit Lack", "Nur in Schwarz"],
    r: 1, k: "Färben", s: 1,
    e: "Vor dem Verdichten ist die Schicht porös und kann Farbstoffe oder Metallsalze aufnehmen."
  },
  {
    f: "Welcher Schritt schließt die Poren der Eloxalschicht am Ende?",
    a: ["Beizen", "Verdichten (Sealing)", "Entfetten", "Schleifen"],
    r: 1, k: "Verdichten", s: 1,
    e: "Beim Verdichten (Sealing) werden die Poren geschlossen, wodurch Farbe und Schutz dauerhaft werden."
  },
  {
    f: "Womit wird das Aluminium vor dem Eloxieren gereinigt?",
    a: ["Mit Öl", "Durch Entfetten/Beizen", "Mit Klarlack", "Mit Sand"],
    r: 1, k: "Vorbehandlung", s: 1,
    e: "Entfetten und Beizen entfernen Öle, Schmutz und die natürliche Oxidhaut für ein gleichmäßiges Ergebnis."
  },
  {
    f: "Ist eine eloxierte Oberfläche härter oder weicher als das blanke Aluminium?",
    a: ["Weicher", "Genauso weich", "Härter", "Sie hat keine Härte"],
    r: 2, k: "Grundlagen", s: 1,
    e: "Aluminiumoxid ist deutlich härter als das Grundmetall und dadurch verschleißfester."
  },
  {
    f: "Welche Farbe hat eine unbehandelte, nur eloxierte (nicht gefärbte) Aluminiumoberfläche meist?",
    a: ["Knallrot", "Silbrig/naturfarben", "Tiefschwarz", "Leuchtend grün"],
    r: 1, k: "Färben", s: 1,
    e: "Ohne Färbung wirkt Eloxal silbrig-natur, teils leicht matt – der bekannte „Natur-Eloxal“-Look."
  },
  {
    f: "Was passiert grob gesagt an der Kathode während des Eloxierens?",
    a: ["Es bildet sich Wasserstoff", "Es bildet sich Gold", "Nichts", "Das Bad gefriert"],
    r: 0, k: "Chemie", s: 1,
    e: "An der Kathode entsteht Wasserstoffgas, während an der Anode die Oxidschicht wächst."
  },

  /* ============================================================
   *  STUFE 2 – MITTEL
   * ============================================================ */
  {
    f: "In welchem typischen Bereich liegt die Schichtdicke von dekorativem Eloxal?",
    a: ["0,1–1 µm", "5–25 µm", "100–200 µm", "1–3 mm"],
    r: 1, k: "Anodisation", s: 2,
    e: "Dekoratives/schützendes Eloxal liegt meist bei ca. 5–25 µm; Hartanodisation deutlich darüber."
  },
  {
    f: "Wie nennt man das Verfahren für besonders dicke, harte Schichten?",
    a: ["Weichanodisieren", "Hartanodisieren (Harteloxal)", "Kaltverzinken", "Brünieren"],
    r: 1, k: "Anodisation", s: 2,
    e: "Harteloxal erzeugt Schichten von ~25–100+ µm bei niedriger Temperatur und hoher Spannung."
  },
  {
    f: "Welche Badtemperatur ist für normales GS-Eloxal typisch?",
    a: ["ca. -5 °C", "ca. 18–20 °C", "ca. 60 °C", "ca. 95 °C"],
    r: 1, k: "Anodisation", s: 2,
    e: "GS-Schwefelsäure-Eloxal läuft meist bei etwa 18–20 °C. Harteloxal arbeitet deutlich kälter (~0 °C)."
  },
  {
    f: "Warum wird Harteloxal bei sehr niedriger Temperatur durchgeführt?",
    a: ["Damit das Bad nicht gefriert", "Um die Rücklösung der Schicht zu bremsen und sie dichter zu machen",
        "Um Strom zu sparen", "Damit die Farbe hält"],
    r: 1, k: "Anodisation", s: 2,
    e: "Kälte reduziert das chemische Rücklösen der Oxidschicht – so entstehen dichtere, härtere Schichten."
  },
  {
    f: "Welche Struktur hat die frisch gebildete Eloxalschicht vor dem Verdichten?",
    a: ["Glasklar geschlossen", "Porös mit vielen feinen Kanälen", "Faserig wie Holz", "Magnetisch"],
    r: 1, k: "Grundlagen", s: 2,
    e: "Es bildet sich eine wabenartige, poröse Struktur – ideal zur Aufnahme von Farbstoffen."
  },
  {
    f: "Was ist „Adsorptivfärben“ beim Eloxieren?",
    a: ["Färben mit Strom", "Einlagern von organischem Farbstoff in die Poren",
        "Lackieren nach dem Sealing", "Bedrucken der Oberfläche"],
    r: 1, k: "Färben", s: 2,
    e: "Beim Adsorptivfärben wird organischer Farbstoff in die offenen Poren eingelagert, bevor verdichtet wird."
  },
  {
    f: "Welches Färbeverfahren gilt als besonders lichtecht und witterungsbeständig?",
    a: ["Tauchfärben mit Lebensmittelfarbe", "Elektrolytisches Färben (Metallsalze)",
        "Filzstift", "Sprühlackierung"],
    r: 1, k: "Färben", s: 2,
    e: "Beim elektrolytischen Färben werden Metallsalze (z. B. Zinn) in die Poren abgeschieden – sehr lichtecht."
  },
  {
    f: "Womit wird häufig verdichtet (gesealt)?",
    a: ["Kaltem Öl", "Heißem, entmineralisiertem Wasser / Nickelsalzen",
        "Trockener Luft", "Flüssigstickstoff"],
    r: 1, k: "Verdichten", s: 2,
    e: "Heißwasser-Sealing (~96–100 °C) oder Kaltsealing mit Nickelsalzen schließen die Poren."
  },
  {
    f: "Was geschieht chemisch beim Heißwasser-Verdichten?",
    a: ["Das Oxid schmilzt", "Aluminiumoxid wandelt sich an der Oberfläche in Böhmit (Hydroxid) um und quillt zu",
        "Die Farbe verdampft", "Das Aluminium rostet"],
    r: 1, k: "Verdichten", s: 2,
    e: "Al₂O₃ reagiert mit Wasser zu Böhmit (AlOOH), das aufquillt und die Poren verschließt."
  },
  {
    f: "Welche Rolle spielt die Aluminiumlegierung für das Eloxal-Ergebnis?",
    a: ["Keine", "Legierungselemente wie Silizium oder Kupfer beeinflussen Farbe und Gleichmäßigkeit stark",
        "Nur das Gewicht ändert sich", "Sie ändert nur den Preis"],
    r: 1, k: "Werkstoff", s: 2,
    e: "Si führt oft zu grauen Schichten, Cu kann fleckig wirken – reine Al-Legierungen eloxieren am schönsten."
  },
  {
    f: "Welche Legierungsgruppe eignet sich besonders gut zum dekorativen Eloxieren?",
    a: ["AlSi-Gusslegierungen mit hohem Si", "AlMg-Legierungen (z. B. 5005)",
        "Kupferreiche 2xxx-Legierungen", "Bleilegierungen"],
    r: 1, k: "Werkstoff", s: 2,
    e: "AlMg-Legierungen wie 5005/6060 liefern gleichmäßige, klare, gut färbbare Schichten."
  },
  {
    f: "Was bewirkt eine E6/EV1-Bezeichnung in der Architektur?",
    a: ["Eine Schraubengröße", "Eine standardisierte Eloxal-Qualität (Vorbehandlung E6, Farbe EV1 natur)",
        "Eine Legierungsnorm", "Eine Lacknorm"],
    r: 1, k: "Qualität", s: 2,
    e: "E6/EV1 ist eine gängige Architektur-Kennzeichnung: E6 = Vorbehandlung, EV1 = naturfarben eloxiert."
  },
  {
    f: "Warum muss beim Eloxieren gut kontaktiert / aufgespannt werden?",
    a: ["Aus Dekogründen", "Weil die wachsende Oxidschicht isoliert und der Strom sonst abreißt",
        "Damit es schwerer wird", "Wegen der Farbe"],
    r: 1, k: "Anodisation", s: 2,
    e: "Da die Schicht isoliert, braucht es festen, ausreichend dimensionierten Kontakt (Gestell/Warenträger)."
  },
  {
    f: "Was versteht man unter „Kontaktstellen“ am fertigen Teil?",
    a: ["Farbfehler", "Unbeschichtete Stellen dort, wo das Gestell das Teil hielt",
        "Kratzer", "Verdichtungsflecken"],
    r: 1, k: "Fehler", s: 2,
    e: "An den Aufhängepunkten fließt der Strom ein – dort bleibt eine kleine unbeschichtete Kontaktstelle."
  },
  {
    f: "Welche Kennzahl beschreibt, wie gut eine Eloxalschicht verdichtet ist?",
    a: ["Der Farbindex", "Der Admittanz-/Leitwertwert oder Farbtropfentest",
        "Die Dichte in kg/m³", "Die Schuhgröße"],
    r: 1, k: "Qualität", s: 2,
    e: "Verdichtungsgüte prüft man z. B. per Admittanzmessung oder Farbtropfentest (DIN-Prüfungen)."
  },
  {
    f: "Was ist der Unterschied zwischen Beizen und Entfetten?",
    a: ["Kein Unterschied", "Entfetten entfernt Öle, Beizen trägt Metall/Oxid ab und mattiert",
        "Beizen ist trocken", "Entfetten färbt"],
    r: 1, k: "Vorbehandlung", s: 2,
    e: "Entfetten reinigt nur; Beizen (meist in Natronlauge) trägt Material ab und erzeugt eine matte Optik."
  },
  {
    f: "Welche Rolle spielt destilliertes/VE-Wasser beim Verdichten?",
    a: ["Keine", "Vermeidet Kalk-/Salzausblühungen und Sealingbelag",
        "Macht die Farbe dunkler", "Erhöht die Härte"],
    r: 1, k: "Verdichten", s: 2,
    e: "Hartes Wasser hinterlässt Beläge; VE-Wasser sorgt für saubere, fleckenfreie Verdichtung."
  },

  /* ============================================================
   *  STUFE 3 – SCHWER (Experten)
   * ============================================================ */
  {
    f: "Aus wie vielen Teilschichten besteht die Eloxalschicht grundsätzlich?",
    a: ["Nur eine homogene Schicht", "Zwei: dünne Sperrschicht (Barrier Layer) + poröse Deckschicht",
        "Drei metallische Lagen", "Fünf Farbschichten"],
    r: 1, k: "Chemie", s: 3,
    e: "Direkt am Metall liegt die dünne, dichte Sperrschicht, darüber die dicke poröse Schicht."
  },
  {
    f: "Welcher Faraday-Zusammenhang bestimmt maßgeblich die Schichtdicke?",
    a: ["Nur die Zeit", "Stromdichte × Zeit (Ladungsmenge)",
        "Nur die Temperatur", "Der Luftdruck"],
    r: 1, k: "Chemie", s: 3,
    e: "Die abgeschiedene/gebildete Menge ist proportional zur Ladungsmenge – Stromdichte und Zeit sind entscheidend."
  },
  {
    f: "Welche typische Stromdichte wird beim GS-Verfahren oft gefahren?",
    a: ["ca. 0,01 A/dm²", "ca. 1,5 A/dm²", "ca. 50 A/dm²", "ca. 500 A/dm²"],
    r: 1, k: "Anodisation", s: 3,
    e: "Übliche Werte liegen bei etwa 1–2 A/dm² (häufig ~1,5 A/dm²) im Schwefelsäure-Standardprozess."
  },
  {
    f: "Was beschreibt das „Burning“ (Verbrennen) beim Eloxieren?",
    a: ["Bad kocht über", "Lokale Überhitzung durch zu hohe Stromdichte → Schichtzerstörung/pulvrige Stellen",
        "Farbstoff verbrennt", "Kathode schmilzt"],
    r: 1, k: "Fehler", s: 3,
    e: "Bei zu hoher lokaler Stromdichte/schlechter Kühlung überhitzt die Schicht und wird pulvrig/zerstört („Burning“)."
  },
  {
    f: "Welche Rolle spielt gelöstes Aluminium im Schwefelsäurebad?",
    a: ["Ist immer schädlich und muss auf 0 gehalten werden", "Ein gewisser Gehalt ist normal; zu hoch verschlechtert Qualität/Leitfähigkeit",
        "Erhöht immer die Härte", "Hat keinen Einfluss"],
    r: 1, k: "Anodisation", s: 3,
    e: "Etwas gelöstes Al (~5–15 g/l) ist normal; zu viel führt zu trüben, schlecht färbbaren Schichten."
  },
  {
    f: "Warum kann Kupfer in der Legierung zu Problemen führen?",
    a: ["Es macht die Schicht magnetisch", "Cu löst sich bevorzugt, führt zu lokalen Fehlstellen und geringerer Korrosionsbeständigkeit",
        "Es erhöht die Härte zu stark", "Es färbt alles grün"],
    r: 1, k: "Werkstoff", s: 3,
    e: "Kupferreiche Legierungen (2xxx) eloxieren ungleichmäßig, die Schicht wird poröser und weniger schützend."
  },
  {
    f: "Was ist der „Interferenzeffekt“ beim elektrolytischen Färben (Zweistufen-/Farbanodisation)?",
    a: ["Ein Stromausfall", "Farbwirkung durch Lichtinterferenz an der modifizierten Porenstruktur",
        "Reflexion an der Kathode", "Ein Messfehler"],
    r: 1, k: "Färben", s: 3,
    e: "Durch gezielte Porenaufweitung entstehen Interferenzfarben – unabhängig von organischen Farbstoffen."
  },
  {
    f: "Welche Wechselstrom-Färbung nutzt z. B. Zinnsalze für Bronze-/Schwarztöne?",
    a: ["Sanodal-Trockenfärbung", "Elektrolytisches Färben (z. B. nach dem Sn-Verfahren)",
        "Heißsublimation", "Pulverbeschichtung"],
    r: 1, k: "Färben", s: 3,
    e: "Beim elektrolytischen Färben mit Zinnsalzen (Wechselstrom) entstehen sehr beständige Bronze- bis Schwarztöne."
  },
  {
    f: "Was versteht man unter „Craze Cracking“ / Rissbildung bei Harteloxal?",
    a: ["Farbrisse durch UV", "Feine Risse durch Eigenspannungen bei dicken Schichten / Temperaturwechsel",
        "Risse im Bad", "Risse in der Anode"],
    r: 1, k: "Fehler", s: 3,
    e: "Dicke, harte, spröde Schichten neigen bei Biegung oder Thermoschock zu feinen Rissnetzwerken."
  },
  {
    f: "Welche Kennzahl (nach DIN EN ISO) prüft die Abriebfestigkeit von Eloxal?",
    a: ["Brinellhärte", "Abrasionstest / Verschleißprüfung (z. B. Taber-Abraser)",
        "Zugversuch", "Kerbschlagbiegeversuch"],
    r: 1, k: "Qualität", s: 3,
    e: "Der Verschleißwiderstand wird u. a. mit dem Taber-Abraser oder Strahlverschleißtests bestimmt."
  },
  {
    f: "Warum ist die Sperrschicht (Barrier Layer) für die Isolation entscheidend?",
    a: ["Sie ist besonders dick", "Sie ist dünn, aber dicht und bestimmt die Durchbruchspannung",
        "Sie leitet Strom gut", "Sie ist porös"],
    r: 1, k: "Chemie", s: 3,
    e: "Die dichte Sperrschicht (nm-Bereich) begrenzt den Stromfluss; ihre Dicke skaliert mit der Spannung."
  },
  {
    f: "Welche Näherung gilt oft für die Sperrschichtdicke in Abhängigkeit der Spannung?",
    a: ["ca. 1 µm pro Volt", "ca. 1–1,4 nm pro Volt", "ca. 1 mm pro Volt", "unabhängig von der Spannung"],
    r: 1, k: "Chemie", s: 3,
    e: "Die Sperrschicht wächst etwa mit ~1,2–1,4 nm/V – ein zentraler Zusammenhang der Anodisation."
  },
  {
    f: "Was bewirkt ein zu hoher Chloridgehalt (z. B. durch Schleppwasser) im Bad?",
    a: ["Bessere Farbe", "Lochfraß/Pitting und Angriff auf die Schicht",
        "Höhere Härte", "Nichts"],
    r: 1, k: "Fehler", s: 3,
    e: "Chloride sind aggressiv und fördern Lochfraß (Pitting) – Bäder werden daher chloridarm gehalten."
  },
  {
    f: "Welche Bedeutung hat der pH-Wert bzw. die Säurekonzentration für die Porenbildung?",
    a: ["Keine", "Sie steuert das Gleichgewicht aus Schichtaufbau und chemischer Rücklösung",
        "Nur die Farbe", "Nur den Stromverbrauch"],
    r: 1, k: "Chemie", s: 3,
    e: "Die Säure löst die Schicht teilweise zurück – dieses Gleichgewicht erzeugt überhaupt erst die Poren."
  },
  {
    f: "Was ist ein typischer Grund für „Sealing Smut“ (Verdichtungsbelag)?",
    a: ["Zu kaltes Bad", "Ausgefällte Reaktionsprodukte/Nickel auf der Oberfläche beim Verdichten",
        "Zu viel Strom", "Falsche Legierung"],
    r: 1, k: "Fehler", s: 3,
    e: "Beim (Nickel-)Sealing können sich Beläge abscheiden, die als grauer Schleier („Smut“) sichtbar werden."
  },
  {
    f: "Warum altert unverdichtetes Eloxal in feuchter Luft optisch nach?",
    a: ["Wegen UV", "Weil Luftfeuchte langsam ein unkontrolliertes Selbst-Sealing bewirkt und Farbe/Aufnahme stört",
        "Wegen Magnetismus", "Gar nicht"],
    r: 1, k: "Verdichten", s: 3,
    e: "Offene Poren reagieren mit Luftfeuchtigkeit – deshalb sollte zeitnah gefärbt/verdichtet werden."
  },
  {
    f: "Welche Aussage zum GS-, GX- und Harteloxal-Verfahren stimmt?",
    a: ["Alle nutzen Salzsäure", "GS = Gleichstrom/Schwefelsäure, GX = Gleichstrom/Schwefel-Oxalsäure, Harteloxal = kalt & dick",
        "Alle laufen bei 90 °C", "Es gibt nur ein Verfahren"],
    r: 1, k: "Anodisation", s: 3,
    e: "GS nutzt reine Schwefelsäure, GX zusätzlich Oxalsäure (härter/farbig), Harteloxal arbeitet kalt für dicke Schichten."
  }

];

// Für Modulsysteme (optional); im Browser als globale Variable nutzbar.
if (typeof module !== "undefined" && module.exports) {
  module.exports = FRAGENKATALOG;
}
