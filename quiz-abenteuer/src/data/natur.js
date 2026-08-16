import { f } from './frage.js';

/** Natur-Welt: 25 Level mit je 5 Fragen. Erste Antwort = richtig. */
export const NATUR_LEVEL = [
  // 1 - Die Jahreszeiten
  [
    f('Wie viele Jahreszeiten gibt es bei uns?', ['Vier', 'Zwei', 'Drei', 'Zwölf'], 'Frühling, Sommer, Herbst und Winter.'),
    f('In welcher Jahreszeit blühen die meisten Bäume?', ['Im Frühling', 'Im Winter', 'Im Herbst', 'Nie'], 'Erst die Blüte, dann die Frucht.'),
    f('Was passiert im Herbst mit den Laubbäumen?', ['Sie verlieren ihre Blätter', 'Sie bekommen Blüten', 'Sie werden größer', 'Nichts'], 'So sparen sie im Winter Wasser.'),
    f('Welche Jahreszeit ist am kältesten?', ['Der Winter', 'Der Sommer', 'Der Frühling', 'Der Herbst'], 'Die Sonne steht dann tief und wärmt wenig.'),
    f('Wann sind die Tage am längsten?', ['Im Sommer', 'Im Winter', 'Im Herbst', 'Immer gleich'], 'Am 21. Juni ist der längste Tag.'),
  ],
  // 2 - Wetter
  [
    f('Womit misst man die Temperatur?', ['Mit einem Thermometer', 'Mit einem Lineal', 'Mit einer Waage', 'Mit einer Uhr'], 'Angegeben wird sie in Grad Celsius.'),
    f('Was ist Wind?', ['Bewegte Luft', 'Kaltes Wasser', 'Staub', 'Licht'], 'Luft strömt von kalt nach warm.'),
    f('Warum sieht man den Blitz vor dem Donner?', ['Licht ist schneller als Schall', 'Der Donner kommt später an die Reihe', 'Blitze sind lauter', 'Zufall'], 'Aus den Sekunden kann man die Entfernung schätzen.'),
    f('Was zeigt eine Wetterfahne an?', ['Aus welcher Richtung der Wind weht', 'Wie warm es ist', 'Wie viel es regnet', 'Wie spät es ist'], 'Sie dreht sich immer in den Wind.'),
    f('Was ist Nebel?', ['Eine Wolke direkt am Boden', 'Rauch', 'Staub aus der Wüste', 'Kalte Luft'], 'Er besteht aus winzigen Wassertröpfchen.'),
  ],
  // 3 - Wolken und Regen
  [
    f('Woraus bestehen Wolken?', ['Aus winzigen Wassertröpfchen', 'Aus Watte', 'Aus Rauch', 'Aus Luft allein'], 'Werden die Tropfen zu schwer, regnet es.'),
    f('Wie kommt Wasser in die Wolken?', ['Es verdunstet und steigt auf', 'Es wird hochgepumpt', 'Regen fällt nach oben', 'Es wächst dort'], 'Sonne erwärmt Seen und Meere.'),
    f('Wie nennt man den Weg des Wassers von Wolke zu Meer und zurück?', ['Wasserkreislauf', 'Wasserstraße', 'Wasserleitung', 'Wasserfall'], 'Er wiederholt sich ohne Ende.'),
    f('Woraus besteht eine Schneeflocke?', ['Aus Eiskristallen', 'Aus Zucker', 'Aus Watte', 'Aus Sand'], 'Fast jede Flocke sieht anders aus.'),
    f('Wie viele Ecken hat ein Schneekristall meistens?', ['Sechs', 'Vier', 'Fünf', 'Acht'], 'Alle Kristalle wachsen sechseckig.'),
  ],
  // 4 - Bäume
  [
    f('Woran erkennt man das Alter eines Baumes?', ['An den Jahresringen im Stamm', 'An der Höhe', 'An der Blattfarbe', 'An der Rinde allein'], 'Jedes Jahr kommt ein Ring dazu.'),
    f('Wozu hat ein Baum Wurzeln?', ['Zum Halt und für Wasser', 'Zum Atmen der Blüten', 'Für die Farbe', 'Zum Schlafen'], 'Die Wurzeln reichen oft sehr weit.'),
    f('Was macht ein Baum mit Sonnenlicht?', ['Er stellt daraus Nahrung her', 'Er wird braun', 'Er speichert es als Licht', 'Nichts'], 'Dabei entsteht Sauerstoff für uns.'),
    f('Welcher Baum behält im Winter seine Nadeln?', ['Die Tanne', 'Die Buche', 'Die Eiche', 'Der Ahorn'], 'Nadeln verdunsten kaum Wasser.'),
    f('Was wächst aus einer Eichel?', ['Eine Eiche', 'Eine Tanne', 'Eine Birke', 'Ein Pilz'], 'Eichhörnchen pflanzen sie ganz nebenbei.'),
  ],
  // 5 - Blumen und Blüten
  [
    f('Wozu haben Blumen bunte Blüten?', ['Um Insekten anzulocken', 'Damit sie hübsch sind', 'Zum Schutz vor Regen', 'Zum Wärmen'], 'Insekten holen Nektar und bestäuben dabei.'),
    f('Wie heißt der feine Staub in einer Blüte?', ['Blütenstaub oder Pollen', 'Mehl', 'Zucker', 'Sand'], 'Er muss zur nächsten Blüte gelangen.'),
    f('Wie fliegen die Samen des Löwenzahns davon?', ['Mit kleinen Fallschirmen im Wind', 'Mit Flügeln', 'Sie hüpfen', 'Vögel tragen sie'], 'Deshalb heißt er auch Pusteblume.'),
    f('Was braucht eine Pflanze zum Wachsen?', ['Licht, Wasser und Erde', 'Nur Dunkelheit', 'Nur Wind', 'Gar nichts'], 'Aus der Erde holt sie Nährstoffe.'),
    f('Welche Blume dreht sich in Richtung Sonne?', ['Die junge Sonnenblume', 'Die Rose', 'Die Tulpe', 'Das Gänseblümchen'], 'Später bleibt die Blüte nach Osten gerichtet.'),
  ],
  // 6 - Pilze
  [
    f('Sind Pilze Pflanzen?', ['Nein, sie sind etwas Eigenes', 'Ja', 'Ja, kleine Bäume', 'Sie sind Tiere'], 'Pilze bilden ein eigenes Reich in der Natur.'),
    f('Welcher Pilz ist giftig?', ['Der Fliegenpilz', 'Der Champignon', 'Der Pfifferling', 'Der Steinpilz'], 'Rot mit weißen Punkten heißt: Finger weg.'),
    f('Was ist der größte Teil eines Pilzes?', ['Das feine Geflecht in der Erde', 'Der Hut', 'Der Stiel', 'Die Lamellen'], 'Es heißt Myzel und kann riesig sein.'),
    f('Warum darf man Pilze nur mit Erwachsenen sammeln?', ['Manche sind sehr giftig', 'Sie sind zu schwer', 'Sie beißen', 'Sie sind zu teuer'], 'Giftige und essbare sehen sich oft ähnlich.'),
    f('Wo wachsen Pilze am liebsten?', ['Im feuchten, schattigen Wald', 'In der prallen Sonne', 'Im trockenen Sand', 'Auf dem Wasser'], 'Nach Regen schießen sie aus dem Boden.'),
  ],
  // 7 - Der Wald
  [
    f('Was ist ganz oben im Wald?', ['Die Baumkronen', 'Das Moos', 'Die Wurzeln', 'Die Pilze'], 'Der Wald hat Stockwerke wie ein Haus.'),
    f('Was geben Bäume an die Luft ab?', ['Sauerstoff', 'Regen', 'Sand', 'Rauch'], 'Den atmen Menschen und Tiere ein.'),
    f('Was passiert mit altem Laub am Waldboden?', ['Es wird zu neuer Erde', 'Es verschwindet einfach', 'Es wird zu Stein', 'Es fliegt weg'], 'Winzige Lebewesen zersetzen es.'),
    f('Warum ist es im Wald oft kühler?', ['Bäume spenden Schatten und speichern Wasser', 'Dort weht immer Wind', 'Dort liegt Schnee', 'Die Erde ist kalt'], 'Der Waldboden hält Feuchtigkeit fest.'),
    f('Was gehört nicht in den Wald?', ['Müll', 'Moos', 'Pilze', 'Käfer'], 'Eine Plastikflasche bleibt hunderte Jahre liegen.'),
  ],
  // 8 - Wiese und Garten
  [
    f('Welches Tier hüpft im Sommer durch die Wiese?', ['Die Heuschrecke', 'Der Hai', 'Der Pinguin', 'Der Wal'], 'Ihre Musik macht sie mit den Beinen.'),
    f('Was ist Klee?', ['Eine Wiesenpflanze', 'Ein Insekt', 'Ein Pilz', 'Ein Stein'], 'Vierblättriger Klee gilt als Glücksbringer.'),
    f('Warum ist eine bunte Wiese besser als kurzer Rasen?', ['Dort finden Insekten Futter', 'Sie sieht ordentlicher aus', 'Sie wächst schneller', 'Sie braucht mehr Wasser'], 'Blühende Wiesen sind ein Insektenparadies.'),
    f('Was macht man mit Küchenabfällen im Garten?', ['Man macht Kompost daraus', 'Man verbrennt sie', 'Man wirft sie in den Fluss', 'Man vergräbt sie im Sand'], 'Aus Abfall wird wertvolle Erde.'),
    f('Warum sollte man Blumen im Garten stehen lassen?', ['Sie sind Futter für Bienen und Schmetterlinge', 'Sie riechen', 'Sie kosten Geld', 'Sie brauchen Platz'], 'Ohne Insekten gäbe es weniger Obst.'),
  ],
  // 9 - Wasser
  [
    f('Bei wie viel Grad gefriert Wasser?', ['Bei 0 Grad', 'Bei 10 Grad', 'Bei 100 Grad', 'Bei 50 Grad'], 'Dann wird es fest und heißt Eis.'),
    f('Bei wie viel Grad kocht Wasser?', ['Bei 100 Grad', 'Bei 40 Grad', 'Bei 0 Grad', 'Bei 200 Grad'], 'Dabei wird es zu Wasserdampf.'),
    f('In welchen drei Formen gibt es Wasser?', ['Fest, flüssig und als Dampf', 'Nur flüssig', 'Nur fest', 'Rot, grün und blau'], 'Eis, Wasser und Wasserdampf.'),
    f('Woher kommt unser Trinkwasser meistens?', ['Aus dem Grundwasser', 'Aus dem Meer', 'Aus Wolken direkt', 'Aus dem Schwimmbad'], 'Es wird aus tiefen Brunnen geholt.'),
    f('Warum kann man Meerwasser nicht trinken?', ['Es ist zu salzig', 'Es ist zu kalt', 'Es ist zu blau', 'Es ist zu tief'], 'Salz entzieht dem Körper Wasser.'),
  ],
  // 10 - Flüsse und Meere
  [
    f('Wohin fließen die meisten Flüsse am Ende?', ['Ins Meer', 'In den Himmel', 'In die Wüste', 'In den Wald'], 'Manche enden auch in einem See.'),
    f('Wo beginnt ein Fluss?', ['An einer Quelle', 'Am Meer', 'In einer Wolke', 'In einem Haus'], 'Oft entspringt sie in den Bergen.'),
    f('Was ist Ebbe und Flut?', ['Das Steigen und Fallen des Meeres', 'Wellen im Sturm', 'Regen am Meer', 'Ein Fisch'], 'Der Mond zieht das Wasser an.'),
    f('Welcher Teil der Erde ist mit Wasser bedeckt?', ['Etwa zwei Drittel', 'Ein Zehntel', 'Alles', 'Fast nichts'], 'Deshalb heißt sie auch der blaue Planet.'),
    f('Was macht ein Wasserfall?', ['Wasser stürzt eine Stufe hinunter', 'Wasser fließt bergauf', 'Wasser bleibt stehen', 'Wasser verschwindet'], 'Über viele Jahre gräbt er sich tiefer.'),
  ],
  // 11 - Berge
  [
    f('Wie heißt der höchste Berg der Erde?', ['Mount Everest', 'Zugspitze', 'Brocken', 'Ätna'], 'Er ist 8848 Meter hoch.'),
    f('Wie heißt der höchste Berg in Deutschland?', ['Zugspitze', 'Mount Everest', 'Matterhorn', 'Feldberg'], 'Sie liegt in den Alpen in Bayern.'),
    f('Warum ist es auf Bergen kälter?', ['Je höher, desto kühler wird die Luft', 'Dort ist mehr Schatten', 'Dort weht kein Wind', 'Dort ist Nacht'], 'Pro 100 Meter wird es etwa ein halbes Grad kälter.'),
    f('Was ist ein Gletscher?', ['Ein riesiger Eisstrom im Gebirge', 'Ein Bergsee', 'Ein Felsen', 'Ein Sturm'], 'Er bewegt sich ganz langsam talwärts.'),
    f('Wie entstehen Gebirge?', ['Erdplatten schieben sich zusammen', 'Menschen bauen sie', 'Regen türmt sie auf', 'Über Nacht'], 'Das dauert viele Millionen Jahre.'),
  ],
  // 12 - Vulkane
  [
    f('Wie heißt das flüssige Gestein, das aus einem Vulkan fließt?', ['Lava', 'Wasser', 'Sand', 'Öl'], 'Unter der Erde heißt es Magma.'),
    f('Wie heiß ist Lava ungefähr?', ['Über 1000 Grad', '50 Grad', '100 Grad', '10 Grad'], 'Sie glüht hellrot bis orange.'),
    f('Was steigt bei einem Ausbruch in den Himmel?', ['Eine Aschewolke', 'Regen', 'Schnee', 'Nebel'], 'Asche kann sogar Flugzeuge stoppen.'),
    f('Wie nennt man die Öffnung oben am Vulkan?', ['Krater', 'Tür', 'Fenster', 'Loch im Meer'], 'Dort tritt das Magma aus.'),
    f('Warum bauen Menschen trotzdem an Vulkanen?', ['Die Böden sind sehr fruchtbar', 'Es ist dort kühl', 'Es gibt dort Gold', 'Es ist ungefährlich'], 'Vulkanasche düngt die Felder.'),
  ],
  // 13 - Steine und Erde
  [
    f('Woraus besteht Sand?', ['Aus winzigen Gesteinskörnchen', 'Aus Zucker', 'Aus Holz', 'Aus Salz'], 'Wellen und Wind zerreiben das Gestein.'),
    f('Was ist Kohle?', ['Uraltes zusammengepresstes Pflanzenmaterial', 'Gepresste Erde', 'Verbranntes Holz', 'Schwarzer Sand'], 'Sie entstand vor Millionen Jahren.'),
    f('Was ist ein Kiesel?', ['Ein rund geschliffener Stein', 'Ein Tier', 'Eine Pflanze', 'Ein Pilz'], 'Wasser hat ihn glatt gerieben.'),
    f('Wie tief geht die Erde bis zum Mittelpunkt?', ['Etwa 6400 Kilometer', '10 Kilometer', '100 Kilometer', 'Eine Million Kilometer'], 'Innen ist es sehr heiß.'),
    f('Was findet man manchmal in Steinen?', ['Fossilien', 'Wasser', 'Luftballons', 'Kekse'], 'Zum Beispiel Abdrücke von Muscheln.'),
  ],
  // 14 - Der Boden lebt
  [
    f('Wer lockert den Boden im Garten?', ['Der Regenwurm', 'Die Fliege', 'Der Vogel', 'Die Schnecke'], 'Seine Gänge lassen Luft und Wasser durch.'),
    f('Wie heißt die fruchtbare dunkle Erde?', ['Humus', 'Sand', 'Lehm allein', 'Kies'], 'Sie entsteht aus zersetzten Pflanzenresten.'),
    f('Was passiert auf einem Komposthaufen?', ['Abfälle werden zu Erde', 'Abfälle verschwinden', 'Abfälle werden zu Stein', 'Nichts'], 'Würmer und Bakterien machen die Arbeit.'),
    f('Wie lange dauert es, bis ein Zentimeter Boden entsteht?', ['Viele Jahrzehnte', 'Einen Tag', 'Eine Woche', 'Ein Jahr'], 'Deshalb ist Boden so kostbar.'),
    f('Was lebt alles in einer Handvoll Erde?', ['Millionen kleinster Lebewesen', 'Gar nichts', 'Nur Steine', 'Nur Wasser'], 'Die meisten sieht man nur mit dem Mikroskop.'),
  ],
  // 15 - Obst und Gemüse
  [
    f('Wo wächst die Kartoffel?', ['Unter der Erde', 'Am Baum', 'Am Strauch', 'Im Wasser'], 'Gegessen wird die verdickte Knolle.'),
    f('Woran wächst der Apfel?', ['Am Baum', 'Unter der Erde', 'An einer Ranke', 'Im Wasser'], 'Vorher war dort eine Blüte.'),
    f('Was ist eine Möhre?', ['Eine Wurzel', 'Eine Blüte', 'Ein Blatt', 'Eine Frucht'], 'Die grünen Blätter schauen oben heraus.'),
    f('Warum ist Obst gesund?', ['Es enthält viele Vitamine', 'Es ist süß', 'Es ist bunt', 'Es ist billig'], 'Vitamine halten den Körper fit.'),
    f('Welches Gemüse wächst am Strunk in Blättern?', ['Der Kohl', 'Die Kartoffel', 'Die Möhre', 'Die Zwiebel'], 'Kohlköpfe können sehr schwer werden.'),
  ],
  // 16 - Vom Korn zum Brot
  [
    f('Woraus wird Mehl gemacht?', ['Aus Getreidekörnern', 'Aus Kartoffeln', 'Aus Milch', 'Aus Zucker'], 'In der Mühle werden die Körner zermahlen.'),
    f('Welche Pflanze ist ein Getreide?', ['Weizen', 'Tomate', 'Apfelbaum', 'Rose'], 'Auch Roggen, Hafer und Gerste gehören dazu.'),
    f('Womit erntet der Bauer Getreide?', ['Mit dem Mähdrescher', 'Mit der Schere', 'Mit dem Bagger', 'Mit dem Traktoranhänger allein'], 'Er schneidet und drischt in einem Schritt.'),
    f('Was macht den Brotteig locker?', ['Die Hefe', 'Das Salz', 'Das Wasser', 'Die Butter'], 'Hefe bildet kleine Gasbläschen.'),
    f('Was ist Stroh?', ['Die trockenen Halme nach der Ernte', 'Frisches Gras', 'Getrocknete Blumen', 'Heu aus Klee'], 'Es dient als Einstreu im Stall.'),
  ],
  // 17 - Blätter und Farben
  [
    f('Warum sind Blätter grün?', ['Wegen des Blattgrüns', 'Weil sie nass sind', 'Wegen der Sonne', 'Weil sie jung sind'], 'Der Farbstoff heißt Chlorophyll.'),
    f('Warum werden Blätter im Herbst bunt?', ['Das Blattgrün wird abgebaut', 'Sie werden angemalt', 'Sie frieren', 'Sie trocknen aus'], 'Übrig bleiben gelbe und rote Farbstoffe.'),
    f('Wozu braucht der Baum seine Blätter?', ['Zum Herstellen von Nahrung', 'Zum Schmücken', 'Zum Atmen der Wurzeln', 'Zum Wasserspeichern'], 'Sie fangen das Sonnenlicht ein.'),
    f('Welches Blatt hat fünf Finger wie eine Hand?', ['Das Ahornblatt', 'Das Eichenblatt', 'Das Birkenblatt', 'Das Lindenblatt'], 'Sein Samen dreht sich wie ein Hubschrauber.'),
    f('Was passiert mit den Blättern nach dem Fallen?', ['Sie werden zu Erde', 'Sie fliegen weg', 'Sie werden zu Holz', 'Sie verschwinden'], 'Regenwürmer ziehen sie in den Boden.'),
  ],
  // 18 - Umwelt schützen
  [
    f('Wie lange braucht eine Plastikflasche zum Zerfallen?', ['Hunderte Jahre', 'Eine Woche', 'Ein Jahr', 'Einen Tag'], 'Deshalb gehört Plastik in die Wertstofftonne.'),
    f('Was tut man mit Müll in der Natur?', ['Man nimmt ihn mit', 'Man vergräbt ihn', 'Man lässt ihn liegen', 'Man verbrennt ihn'], 'Tiere verletzen sich sonst daran.'),
    f('Wie spart man zu Hause Wasser?', ['Duschen statt baden', 'Den Hahn laufen lassen', 'Öfter baden', 'Mehr gießen'], 'Beim Zähneputzen den Hahn zudrehen hilft auch.'),
    f('Wie hilft man Vögeln im Winter?', ['Mit Futterhaus und Wasser', 'Mit Süßigkeiten', 'Mit Milch', 'Gar nicht'], 'Körner und Fettfutter sind ideal.'),
    f('Warum ist ein Insektenhotel gut?', ['Wildbienen finden dort einen Unterschlupf', 'Es sieht schön aus', 'Es hält Insekten fern', 'Es macht Honig'], 'Wildbienen bestäuben unsere Obstbäume.'),
  ],
  // 19 - Müll und Recycling
  [
    f('In welche Tonne kommt Altpapier?', ['In die blaue Tonne', 'In die gelbe Tonne', 'In den Biomüll', 'In den Restmüll'], 'Aus Altpapier wird neues Papier.'),
    f('Was kommt in die Biotonne?', ['Obst- und Gemüsereste', 'Batterien', 'Glasflaschen', 'Plastiktüten'], 'Daraus wird Kompost oder Energie.'),
    f('Wo gibt man alte Batterien ab?', ['An einer Sammelstelle im Laden', 'Im Restmüll', 'Im Garten', 'In der Biotonne'], 'Batterien enthalten Schadstoffe.'),
    f('Was passiert mit Altglas?', ['Es wird eingeschmolzen und neu geformt', 'Es wird vergraben', 'Es wird verbrannt', 'Es wird weggeworfen'], 'Glas kann man immer wieder verwenden.'),
    f('Was bedeutet Recycling?', ['Aus Altem wird Neues gemacht', 'Alles wird verbrannt', 'Müll wird versteckt', 'Man kauft mehr'], 'Das spart Rohstoffe und Energie.'),
  ],
  // 20 - Energie aus der Natur
  [
    f('Was macht ein Windrad?', ['Es macht Strom aus Wind', 'Es macht Wind', 'Es kühlt die Luft', 'Es pumpt Wasser hoch'], 'Der Wind dreht die großen Flügel.'),
    f('Was gewinnen Solarzellen aus der Sonne?', ['Strom', 'Wasser', 'Wind', 'Erde'], 'Sie sitzen oft auf Hausdächern.'),
    f('Wie macht ein Wasserkraftwerk Strom?', ['Fließendes Wasser treibt Turbinen an', 'Es kocht Wasser', 'Es friert Wasser ein', 'Es filtert Wasser'], 'Meist steht es an einem Stausee.'),
    f('Warum sind Sonne und Wind gute Energiequellen?', ['Sie gehen nie aus', 'Sie sind laut', 'Sie sind selten', 'Sie kosten nichts'], 'Man nennt sie erneuerbare Energien.'),
    f('Wie spart man zu Hause Strom?', ['Licht ausmachen, wenn man geht', 'Alles anlassen', 'Mehr Lampen kaufen', 'Fenster offen lassen'], 'Auch Geräte ganz ausschalten hilft.'),
  ],
  // 21 - Tag und Nacht
  [
    f('Warum wird es abends dunkel?', ['Die Erde dreht sich von der Sonne weg', 'Die Sonne geht schlafen', 'Wolken decken alles zu', 'Die Sonne fällt herunter'], 'Eine Drehung dauert 24 Stunden.'),
    f('In welche Richtung geht die Sonne auf?', ['Im Osten', 'Im Westen', 'Im Norden', 'Im Süden'], 'Am Abend geht sie im Westen unter.'),
    f('Welche Tiere sind nachts wach?', ['Eule und Fledermaus', 'Biene und Schmetterling', 'Huhn und Kuh', 'Ameise und Käfer'], 'Man nennt sie nachtaktiv.'),
    f('Warum blühen manche Blumen nur nachts?', ['Sie locken Nachtfalter an', 'Sie mögen keinen Regen', 'Sie schlafen tagsüber', 'Sie sind kaputt'], 'Ihr Duft ist nachts am stärksten.'),
    f('Wann ist am Tag der Schatten am kürzesten?', ['Mittags', 'Morgens', 'Abends', 'Nachts'], 'Dann steht die Sonne am höchsten.'),
  ],
  // 22 - Licht, Schatten, Regenbogen
  [
    f('Wie entsteht ein Schatten?', ['Etwas hält das Licht auf', 'Licht wird bunt', 'Es wird kalt', 'Wolken malen ihn'], 'Hinter dem Gegenstand fehlt das Licht.'),
    f('Wann sieht man einen Regenbogen?', ['Wenn Sonne und Regen zusammenkommen', 'Nur nachts', 'Nur im Winter', 'Bei Nebel'], 'Die Sonne muss hinter dir stehen.'),
    f('Wie viele Farben zeigt der Regenbogen?', ['Sieben', 'Drei', 'Zwei', 'Zwanzig'], 'Von Rot außen bis Violett innen.'),
    f('Was passiert mit Licht in einem Wassertropfen?', ['Es wird in Farben zerlegt', 'Es verschwindet', 'Es wird lauter', 'Es wird kalt'], 'Ein Prisma macht das genauso.'),
    f('Warum ist der Himmel blau?', ['Die Luft streut blaues Licht besonders stark', 'Das Meer spiegelt sich', 'Er ist angemalt', 'Wegen der Wolken'], 'Abends wird er rot, weil das Licht weiter reist.'),
  ],
  // 23 - Luft
  [
    f('Welches Gas aus der Luft brauchen wir zum Atmen?', ['Sauerstoff', 'Rauch', 'Helium', 'Kohlendioxid'], 'Pflanzen stellen ihn für uns her.'),
    f('Kann man Luft sehen?', ['Nein, aber man spürt sie als Wind', 'Ja, sie ist blau', 'Ja, sie ist weiß', 'Nur nachts'], 'Sie ist unsichtbar, aber überall.'),
    f('Was passiert mit warmer Luft?', ['Sie steigt nach oben', 'Sie sinkt nach unten', 'Sie bleibt stehen', 'Sie wird fest'], 'Deshalb steigen Heißluftballons.'),
    f('Was macht ein Drachen am Himmel möglich?', ['Der Wind', 'Die Sonne', 'Der Regen', 'Die Schwerkraft'], 'Die Luft drückt ihn nach oben.'),
    f('Was macht die Luft schlecht?', ['Abgase und Rauch', 'Regen', 'Wind', 'Bäume'], 'Bäume filtern Staub aus der Luft.'),
  ],
  // 24 - Lebensräume
  [
    f('Was ist ein Lebensraum?', ['Der Ort, an dem ein Tier lebt', 'Ein Zimmer im Haus', 'Ein Zoo', 'Ein Zelt'], 'Wald, Wiese und Teich sind Lebensräume.'),
    f('Welche Tiere leben am Teich?', ['Frosch und Libelle', 'Kamel und Löwe', 'Eisbär und Pinguin', 'Koala und Känguru'], 'Der Teich ist Kinderstube für viele Tiere.'),
    f('Warum sind Hecken für Tiere wichtig?', ['Sie bieten Schutz und Nahrung', 'Sie sehen schön aus', 'Sie halten Wind ab', 'Sie sind hoch'], 'Vögel bauen dort ihre Nester.'),
    f('Was passiert, wenn ein Lebensraum verschwindet?', ['Die Tiere verlieren ihr Zuhause', 'Nichts', 'Es wird schöner', 'Die Tiere werden größer'], 'Deshalb schützt man Moore und Wälder.'),
    f('Was ist ein Naturschutzgebiet?', ['Ein geschützter Bereich für Tiere und Pflanzen', 'Ein Spielplatz', 'Ein Parkplatz', 'Ein Museum'], 'Dort darf man die Wege nicht verlassen.'),
  ],
  // 25 - Großes Natur-Finale
  [
    f('Was entsteht, wenn Wasser verdunstet und aufsteigt?', ['Wolken', 'Schnee am Boden', 'Steine', 'Wind'], 'So beginnt der Wasserkreislauf.'),
    f('Woran erkennt man das Alter eines Baumes?', ['An den Jahresringen', 'An den Blättern', 'An der Rinde', 'An den Früchten'], 'Ein Ring pro Jahr.'),
    f('Wie heißt der höchste Berg der Welt?', ['Mount Everest', 'Zugspitze', 'Ätna', 'Brocken'], '8848 Meter hoch.'),
    f('Was kommt in die blaue Tonne?', ['Papier', 'Glas', 'Bioabfall', 'Batterien'], 'Daraus wird neues Papier.'),
    f('Warum ist der Regenwurm nützlich?', ['Er macht den Boden locker und fruchtbar', 'Er frisst Schädlinge', 'Er singt', 'Er bestäubt Blumen'], 'Ohne ihn wäre die Erde hart wie Beton.'),
  ],
];
