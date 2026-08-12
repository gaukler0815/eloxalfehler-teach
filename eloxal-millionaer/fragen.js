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
  },

  /* ============================================================
   *  STUFE 1 – LEICHT (Erweiterung)
   * ============================================================ */
  {
    f: "Ist Eloxieren dasselbe wie Verchromen oder Lackieren?",
    a: ["Ja, völlig gleich", "Nein – Eloxal wandelt das Metall selbst in Oxid um, es wird nichts Fremdes aufgetragen",
        "Ja, nur der Name ist anders", "Nein, Eloxal ist eine Klebefolie"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Beim Lackieren/Verchromen liegt Material obenauf. Eloxal ist eine Umwandlungsschicht aus dem Alu selbst."
  },
  {
    f: "Kann man normalen Stahl oder Edelstahl klassisch eloxieren?",
    a: ["Ja, genauso wie Alu", "Nein, das Eloxalverfahren funktioniert nur mit Aluminium",
        "Nur Edelstahl", "Nur verzinkten Stahl"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Eloxal ist Aluminium vorbehalten. Stahl würde im Säurebad rosten/sich auflösen statt eine Schutzschicht zu bilden."
  },
  {
    f: "Wofür steht die Abkürzung „GS“ beim GS-Verfahren?",
    a: ["Gute Schicht", "Gleichstrom-Schwefelsäure", "Glanz-Sealing", "Grün-Silber"],
    r: 1, k: "Anodisation", s: 1,
    e: "GS = Gleichstrom-Schwefelsäure – das Standard-Eloxalverfahren."
  },
  {
    f: "Welche dieser Alltagsdinge sind häufig eloxiert?",
    a: ["Holzstühle", "Alu-Kochtöpfe, Taschenlampen und Smartphone-Rahmen",
        "Glasflaschen", "Autoreifen"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Viele Aluminiumprodukte – von Kochgeschirr bis Gehäuse – werden eloxiert für Schutz und Optik."
  },
  {
    f: "Warum wird zwischen den Prozessbädern gespült?",
    a: ["Damit es glänzt", "Um Verschleppung von Chemikalien ins nächste Bad zu vermeiden",
        "Um das Teil zu kühlen", "Aus Dekogründen"],
    r: 1, k: "Vorbehandlung", s: 1,
    e: "Spülen verhindert, dass Säure/Lauge verschleppt wird und das folgende Bad verunreinigt."
  },
  {
    f: "Wie wirkt eine frisch gebeizte (mattierte) Eloxaloberfläche optisch?",
    a: ["Hochglänzend wie ein Spiegel", "Seidenmatt / gleichmäßig matt",
        "Durchsichtig", "Rau wie Schmirgelpapier"],
    r: 1, k: "Vorbehandlung", s: 1,
    e: "Alkalisches Beizen (Natronlauge) erzeugt die typische, gleichmäßig seidenmatte Eloxaloptik."
  },
  {
    f: "Welche Rolle spielt Strom beim Eloxieren?",
    a: ["Keine", "Er treibt die elektrolytische Bildung der Oxidschicht an",
        "Er heizt nur das Bad", "Er färbt das Teil"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Ohne den elektrischen Strom (Anodisation) wächst keine Oxidschicht – er ist der Motor des Verfahrens."
  },
  {
    f: "Welche Farben sind bei Architektur-Eloxal besonders klassisch?",
    a: ["Natur (silber), Schwarz und Bronzetöne", "Neonpink und Neongelb",
        "Nur Weiß", "Gold und Rosa"],
    r: 0, k: "Färben", s: 1,
    e: "Silber (natur), Schwarz und diverse Bronzetöne sind die typischen, sehr beständigen Architekturfarben."
  },
  {
    f: "Was passiert mit dem Aluminium, wenn man es gar nicht schützt?",
    a: ["Es bleibt ewig blank", "Es bildet von selbst eine dünne, aber ungleichmäßige Oxidhaut und kann korrodieren",
        "Es rostet rot wie Eisen", "Es wird magnetisch"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Alu bildet natürlich eine dünne Oxidhaut; Eloxal macht diese dick, gleichmäßig und dauerhaft schützend."
  },
  {
    f: "Ist die eloxierte Schicht dünner oder dicker als ein menschliches Haar?",
    a: ["Viel dicker als ein Finger", "Meist dünner – wenige Tausendstel Millimeter (µm)",
        "Genau 1 cm", "Immer 1 mm"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Dekoratives Eloxal liegt bei ca. 5–25 µm – deutlich dünner als ein Haar (~50–70 µm)."
  },
  {
    f: "Welches Bad wird zum Aufhellen/Glänzen vor dem Eloxieren manchmal genutzt?",
    a: ["Ein Glänzbad (chemisch/elektrolytisch Polieren)", "Ein Ölbad",
        "Ein Salzwasserbad", "Ein Zuckerbad"],
    r: 0, k: "Vorbehandlung", s: 1,
    e: "Chemisches oder elektrolytisches Glänzen erzeugt vor dem Eloxieren eine glänzende, glatte Oberfläche."
  },
  {
    f: "Warum ist eloxiertes Aluminium bei Fensterrahmen so beliebt?",
    a: ["Weil es leuchtet", "Wegen Korrosionsschutz, Kratzfestigkeit und Farbstabilität über Jahrzehnte",
        "Weil es billiger als Farbe ist", "Weil es weich ist"],
    r: 1, k: "Grundlagen", s: 1,
    e: "Eloxal ist extrem witterungs- und UV-beständig – ideal für langlebige Fassaden und Fenster."
  },

  /* ============================================================
   *  STUFE 2 – MITTEL (Erweiterung)
   * ============================================================ */
  {
    f: "Warum wird der Elektrolyt beim Eloxieren umgewälzt/bewegt?",
    a: ["Nur der Optik wegen", "Für gleichmäßige Temperatur und Schichtdicke und um lokale Überhitzung zu vermeiden",
        "Um Strom zu sparen", "Damit es schäumt"],
    r: 1, k: "Anodisation", s: 2,
    e: "Bad- und Warenbewegung (oft Luft-/Umwälzung) sorgt für gleichmäßige Kühlung und homogene Schichten."
  },
  {
    f: "Welche Schichtdicke fordert man für bewitterte Außen-Architektur meist mindestens?",
    a: ["ca. 2 µm", "ca. 20–25 µm (z. B. Klasse AA25)", "ca. 0,5 µm", "ca. 200 µm"],
    r: 1, k: "Qualität", s: 2,
    e: "Für Außenanwendungen sind i. d. R. ≥20 µm (oft 25 µm) gefordert, damit die Schicht dauerhaft schützt."
  },
  {
    f: "Was beschreibt der Farbtropfentest (nach ISO 2143)?",
    a: ["Die Schichtdicke", "Die Verdichtungsqualität – schlecht verdichtete Schicht nimmt Farbstoff auf",
        "Die Härte", "Den Stromverbrauch"],
    r: 1, k: "Qualität", s: 2,
    e: "Ein Farbstofftropfen auf schlecht verdichteter Schicht hinterlässt einen Fleck – gut verdichtet perlt er ab."
  },
  {
    f: "Was ist Chromsäure-Anodisieren (CAA) typischerweise?",
    a: ["Ein Färbeverfahren", "Ein dünnes, sehr korrosionsschützendes Verfahren u. a. für Luftfahrt/Klebeflächen",
        "Ein Verdichtungsschritt", "Eine Beize"],
    r: 1, k: "Anodisation", s: 2,
    e: "CAA erzeugt sehr dünne, aber gut haftende und korrosionsschützende Schichten – klassisch in der Luftfahrt."
  },
  {
    f: "Warum wird Harteloxal für Verschleißteile oft nur teil- oder gar nicht verdichtet?",
    a: ["Aus Kostengründen", "Weil Verdichten die Härte etwas senkt – für maximale Verschleißfestigkeit lässt man Poren teils offen",
        "Weil es sonst rostet", "Weil Farbe sonst nicht hält"],
    r: 1, k: "Verdichten", s: 2,
    e: "Sealing reduziert die Oberflächenhärte geringfügig; bei reinen Verschleißteilen verzichtet man oft bewusst darauf."
  },
  {
    f: "Um wie viel ändert sich das Maß eines Teils grob durch die Eloxalschicht?",
    a: ["Gar nicht", "Etwa um die halbe Schichtdicke pro Fläche (die Schicht wächst rund zur Hälfte nach außen)",
        "Um die doppelte Schichtdicke", "Um mehrere Millimeter"],
    r: 1, k: "Werkstoff", s: 2,
    e: "Da die Schicht etwa zur Hälfte nach außen wächst, nimmt das Maß pro Seite um ca. die halbe Schichtdicke zu."
  },
  {
    f: "Was ist beim zweistufigen elektrolytischen Färben der Ablauf?",
    a: ["Erst färben, dann anodisieren", "Erst anodisieren, dann in einem Metallsalzbad mit Wechselstrom einfärben",
        "Nur lackieren", "Nur beizen"],
    r: 1, k: "Färben", s: 2,
    e: "Zuerst wird die Oxidschicht erzeugt, dann werden mit Wechselstrom Metallsalze (z. B. Zinn) in die Poren gefällt."
  },
  {
    f: "Welche typische Schwefelsäure-Konzentration hat ein GS-Standardbad ungefähr?",
    a: ["ca. 5 g/l", "ca. 180–200 g/l", "ca. 1000 g/l", "ca. 50 g/l"],
    r: 1, k: "Anodisation", s: 2,
    e: "Übliche GS-Bäder arbeiten mit rund 180–200 g/l Schwefelsäure (plus etwas gelöstem Aluminium)."
  },
  {
    f: "Was bewirkt eine zu lange Beizzeit in Natronlauge?",
    a: ["Nichts", "Zu starken Materialabtrag, Maßänderung und ggf. zu raue Oberfläche",
        "Höhere Härte", "Bessere Farbe"],
    r: 1, k: "Vorbehandlung", s: 2,
    e: "Zu langes Beizen trägt zu viel Alu ab – Maße und scharfe Kanten leiden, die Oberfläche wird überbeizt."
  },
  {
    f: "Wozu dient das „Dekapieren“/Absäuren nach dem alkalischen Beizen?",
    a: ["Zum Färben", "Zum Entfernen von Beizschlamm/Belägen (z. B. in Salpetersäure) vor dem Eloxieren",
        "Zum Verdichten", "Zum Trocknen"],
    r: 1, k: "Vorbehandlung", s: 2,
    e: "Nach dem Beizen bleibt oft ein Belag (Legierungsbestandteile); Dekapieren in Säure entfernt ihn und klärt die Oberfläche."
  },
  {
    f: "Warum liefert eine siliziumreiche Gusslegierung oft graue, unschöne Eloxalschichten?",
    a: ["Wegen des Kupfers", "Silizium lässt sich nicht mit-anodisieren und bleibt als graue Partikel in der Schicht",
        "Wegen zu viel Strom", "Wegen der Farbe des Bades"],
    r: 1, k: "Werkstoff", s: 2,
    e: "Freies Silizium oxidiert nicht mit und erscheint als graue Einlagerung – Gusslegierungen eloxieren daher oft grau."
  },
  {
    f: "Welche Kennzeichnung steht in der Architektur oft für naturfarbenes Eloxal?",
    a: ["C34", "EV1 / E6-EV1", "RAL 9010", "AA5"],
    r: 1, k: "Qualität", s: 2,
    e: "EV1 bezeichnet naturfarben (silber) eloxiert; E6 steht für die zugehörige Vorbehandlung (mattgebeizt)."
  },
  {
    f: "Was ist ein Grund für ungleichmäßige Schichtdicke auf einem großen Bauteil?",
    a: ["Zu viele Fragen", "Ungleichmäßige Stromverteilung/Kontaktierung und Abschattung im Bad",
        "Zu sauberes Teil", "Zu kaltes Wasser beim Spülen"],
    r: 1, k: "Fehler", s: 2,
    e: "Kanten ziehen mehr Strom, abgeschattete Flächen weniger – gute Kontaktierung und Anordnung sind entscheidend."
  },
  {
    f: "Welche Farbe erzeugt elektrolytisches Färben mit Zinnsalzen je nach Dauer?",
    a: ["Nur Rot", "Champagner über Bronze bis Schwarz", "Nur Blau", "Nur Grün"],
    r: 1, k: "Färben", s: 2,
    e: "Mit der Abscheidedauer im Zinnsalzbad geht der Ton von hellem Champagner über Bronze bis Tiefschwarz."
  },
  {
    f: "Warum ist die Konstanz der Badtemperatur für die Farbe wichtig?",
    a: ["Gar nicht", "Temperaturschwankungen ändern Porenstruktur und Schichtaufbau → Farbabweichungen",
        "Nur wegen der Härte", "Nur wegen des Stroms"],
    r: 1, k: "Anodisation", s: 2,
    e: "Schon kleine Temperaturunterschiede verändern die Poren und damit Farbannahme und Farbton spürbar."
  },

  /* ============================================================
   *  STUFE 3 – SCHWER (Erweiterung)
   * ============================================================ */
  {
    f: "Wie ist die poröse Eloxalschicht im idealisierten Modell (nach Keller) aufgebaut?",
    a: ["Als glatte Platte", "Als dichte Packung sechseckiger Zellen mit je einer zentralen Pore",
        "Als Fasergeflecht", "Als Kügelchen"],
    r: 1, k: "Chemie", s: 3,
    e: "Das klassische Modell beschreibt hexagonale Oxidzellen mit einer zentralen Pore und der Sperrschicht am Boden."
  },
  {
    f: "Wovon hängt der Porendurchmesser bzw. Zellabstand maßgeblich ab?",
    a: ["Von der Farbe", "Von Spannung und Elektrolyt (Säureart/-konzentration, Temperatur)",
        "Von der Teilegröße", "Vom Wetter"],
    r: 1, k: "Chemie", s: 3,
    e: "Zellgröße/Porendurchmesser skalieren mit der Formierspannung und dem verwendeten Elektrolyten."
  },
  {
    f: "Welche Norm regelt die Prüfung der Verdichtungsqualität über den Massenverlust im Säuretest?",
    a: ["ISO 2360", "ISO 3210", "ISO 9001", "ISO 14001"],
    r: 1, k: "Qualität", s: 3,
    e: "ISO 3210 prüft die Verdichtung über den Massenverlust nach einem Säuretauchtest (Phosphor-/Chromsäure)."
  },
  {
    f: "Mit welchem Verfahren misst man die Eloxal-Schichtdicke zerstörungsfrei?",
    a: ["Mit dem Zollstock", "Wirbelstromverfahren (Eddy Current) nach ISO 2360",
        "Durch Wiegen", "Mit dem Magneten"],
    r: 1, k: "Qualität", s: 3,
    e: "Auf NE-Metallen misst das Wirbelstromverfahren (ISO 2360) die nichtleitende Oxidschichtdicke berührungslos."
  },
  {
    f: "Warum muss beim Eloxieren gekühlt werden?",
    a: ["Damit die Farbe hält", "Die Schichtbildung ist exotherm und lokale Wärme würde die Schicht rücklösen/verbrennen",
        "Damit das Bad nicht gefriert", "Wegen des Stroms an der Kathode"],
    r: 1, k: "Anodisation", s: 3,
    e: "Der Prozess erzeugt Wärme; ohne Kühlung steigt die Rücklösung, die Schicht wird weich oder „verbrennt“."
  },
  {
    f: "Welches Salz wird beim klassischen Wechselstrom-Farbeloxal (Sn-Verfahren) verwendet?",
    a: ["Kochsalz", "Zinn(II)-sulfat", "Kupfersulfat", "Natriumhydroxid"],
    r: 1, k: "Färben", s: 3,
    e: "Zinn(II)-sulfat ist das gängige Metallsalz für sehr licht- und witterungsechte Bronze-/Schwarztöne."
  },
  {
    f: "Wie entstehen echte Interferenzfarben beim Eloxieren (ohne Farbstoff)?",
    a: ["Durch Lackieren", "Durch gezielte Modifikation des Porenbodens, sodass Licht interferiert",
        "Durch Erhitzen", "Durch Magnetfelder"],
    r: 1, k: "Färben", s: 3,
    e: "Wird der Porenboden definiert aufgeweitet/modifiziert, entstehen durch Lichtinterferenz stabile Farbtöne."
  },
  {
    f: "Wie wird Eloxal-Abwasser (Al-haltig) typischerweise aufbereitet?",
    a: ["In den Gully", "Neutralisation und Ausfällung als Aluminiumhydroxid-Schlamm",
        "Verbrennen", "Eindampfen zu Gold"],
    r: 1, k: "Umwelt", s: 3,
    e: "Saure/alkalische Abwässer werden neutralisiert; Aluminium fällt als Al(OH)₃-Schlamm aus und wird abgetrennt."
  },
  {
    f: "Welche Skala beschreibt die Lichtechtheit von Eloxalfarben?",
    a: ["Mohshärte", "Blauwollskala (Lichtechtheitsstufen)", "pH-Skala", "Beaufortskala"],
    r: 1, k: "Qualität", s: 3,
    e: "Die Blauwollskala (1–8) bewertet die Lichtechtheit – elektrolytische Farben liegen hier meist sehr hoch."
  },
  {
    f: "Warum wird vor dem elektrolytischen Färben oft ein „Pore Widening“ gemacht?",
    a: ["Zum Reinigen", "Um die Poren definiert aufzuweiten und die Farbannahme/Abscheidung zu steuern",
        "Zum Verdichten", "Zum Trocknen"],
    r: 1, k: "Färben", s: 3,
    e: "Ein kontrolliertes Aufweiten der Poren verbessert und steuert die Einlagerung der Farbmetalle."
  },
  {
    f: "Welche Aussage zur Sperrschichtdicke stimmt?",
    a: ["Sie ist unabhängig von der Spannung", "Sie wächst näherungsweise proportional zur angelegten Spannung (~1,2–1,4 nm/V)",
        "Sie ist immer 1 µm", "Sie schrumpft mit der Spannung"],
    r: 1, k: "Chemie", s: 3,
    e: "Die dichte Barriereschicht am Porenboden wächst etwa mit ~1,2–1,4 nm pro Volt Formierspannung."
  },
  {
    f: "Was ist „Powdering“/Kreiden bei Eloxal?",
    a: ["Ein Reinigungsschritt", "Pulvrige, nicht haftende Schicht durch zu hohe Temperatur/Stromdichte",
        "Ein Farbeffekt", "Ein Verdichtungsbelag"],
    r: 1, k: "Fehler", s: 3,
    e: "Bei Überhitzung wird die Schicht mürbe und kreidet ab („Powdering“) – ein klassischer Prozessfehler."
  },
  {
    f: "Welche typischen Prozessdaten hat ein Harteloxal-Bad in etwa?",
    a: ["~60 °C, 5 g/l Säure, 0,1 A/dm²", "~0 °C, ~200 g/l Schwefelsäure, ~2–3 A/dm², Spannung teils bis ~60 V",
        "~20 °C, 1000 g/l, 0,5 A/dm²", "~95 °C, reines Wasser"],
    r: 1, k: "Anodisation", s: 3,
    e: "Harteloxal läuft nahe 0 °C bei hoher Stromdichte und Spannung, um dicke, dichte, harte Schichten zu erzeugen."
  },
  {
    f: "Warum steigt die Anodenspannung beim Eloxieren im Verlauf meist an?",
    a: ["Weil das Bad kälter wird", "Weil die wachsende, isolierende Schicht den Widerstand erhöht",
        "Wegen der Farbe", "Zufällig"],
    r: 1, k: "Chemie", s: 3,
    e: "Je dicker die isolierende Oxidschicht, desto höher die nötige Spannung für konstante Stromdichte."
  },
  {
    f: "Welche Prüfung nutzt die Admittanz-/Impedanzmessung?",
    a: ["Härte", "Verdichtungsgrad (gut verdichtete Schicht hat geringe Admittanz)",
        "Farbe", "Schichtdicke"],
    r: 1, k: "Qualität", s: 3,
    e: "Die elektrische Admittanz sinkt mit besserer Verdichtung – eine schnelle, zerstörungsfreie Sealing-Prüfung."
  }

];

// Für Modulsysteme (optional); im Browser als globale Variable nutzbar.
if (typeof module !== "undefined" && module.exports) {
  module.exports = FRAGENKATALOG;
}
