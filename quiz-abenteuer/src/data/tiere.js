import { f } from './frage.js';

/** Tier-Welt: 25 Level mit je 5 Fragen. Erste Antwort = richtig. */
export const TIER_LEVEL = [
  // 1 - Haustiere
  [
    f('Wie heißt ein junger Hund?', ['Welpe', 'Kalb', 'Fohlen', 'Küken'], 'Welpen kommen blind zur Welt.'),
    f('Womit fühlt eine Katze im Dunkeln?', ['Mit ihren Schnurrhaaren', 'Mit dem Schwanz', 'Mit den Ohren', 'Mit den Krallen'], 'Die Tasthaare messen, ob eine Lücke passt.'),
    f('Was frisst ein Kaninchen am liebsten?', ['Heu und Gemüse', 'Fleisch', 'Fisch', 'Käse'], 'Heu hält die Zähne kurz, die immer nachwachsen.'),
    f('Warum hecheln Hunde bei Hitze?', ['So kühlen sie sich ab', 'Sie haben Hunger', 'Sie sind traurig', 'Sie zählen'], 'Hunde schwitzen fast nur an den Pfoten.'),
    f('Wie zeigt eine Katze, dass es ihr gut geht?', ['Sie schnurrt', 'Sie faucht', 'Sie kratzt', 'Sie versteckt sich'], 'Schnurren beruhigt Katzen sogar selbst.'),
  ],
  // 2 - Auf dem Bauernhof
  [
    f('Wie viele Magenteile hat eine Kuh?', ['Vier', 'Einen', 'Zwei', 'Zehn'], 'Deshalb kann sie Gras wiederkäuen.'),
    f('Wie heißt ein junges Pferd?', ['Fohlen', 'Lamm', 'Ferkel', 'Welpe'], 'Ein Fohlen kann fast sofort laufen.'),
    f('Warum suhlen sich Schweine im Schlamm?', ['Zum Abkühlen', 'Weil sie schmutzig sein wollen', 'Zum Essen', 'Zum Schlafen'], 'Schweine können kaum schwitzen.'),
    f('Was gibt uns das Schaf?', ['Wolle', 'Honig', 'Eier', 'Federn'], 'Einmal im Jahr wird es geschoren.'),
    f('Wie heißt das weibliche Huhn?', ['Henne', 'Hahn', 'Gans', 'Ente'], 'Nur die Henne legt Eier.'),
  ],
  // 3 - Im heimischen Wald
  [
    f('Was macht ein Eichhörnchen im Herbst?', ['Es versteckt Nüsse', 'Es fliegt weg', 'Es baut ein Nest aus Stein', 'Es schläft im Wasser'], 'Vergessene Nüsse werden später zu Bäumen.'),
    f('Warum klopft ein Specht an Bäume?', ['Er sucht Insekten und baut Höhlen', 'Er macht Musik', 'Er schärft den Schnabel', 'Er weckt andere Vögel'], 'Sein Schädel federt die Stöße ab.'),
    f('Welches Tier gräbt Gänge unter der Wiese?', ['Der Maulwurf', 'Der Specht', 'Der Fuchs', 'Die Eule'], 'Die Erdhaufen sind der Aushub.'),
    f('Was frisst ein Fuchs?', ['Fast alles, zum Beispiel Mäuse und Beeren', 'Nur Gras', 'Nur Fisch', 'Nur Honig'], 'Füchse sind Allesfresser.'),
    f('Was macht ein Reh bei Gefahr?', ['Es flieht mit großen Sprüngen', 'Es greift an', 'Es klettert auf Bäume', 'Es gräbt sich ein'], 'Der weiße Fleck am Po warnt die anderen.'),
  ],
  // 4 - Vögel
  [
    f('Womit fliegen Vögel?', ['Mit Flügeln aus Federn', 'Mit Flughaut', 'Mit dem Schwanz', 'Mit Ohren'], 'Federn sind leicht und sehr stabil.'),
    f('Was macht die Feder eines Vogels so praktisch?', ['Sie ist leicht und hält warm', 'Sie ist schwer', 'Sie ist aus Metall', 'Sie leuchtet'], 'Vögel putzen Federn jeden Tag.'),
    f('Wie heißt der kleinste Vogel der Welt?', ['Die Bienenelfe, ein Kolibri', 'Der Spatz', 'Die Meise', 'Der Adler'], 'Sie ist kaum größer als eine Hummel.'),
    f('Welcher Vogel kann nicht fliegen?', ['Der Pinguin', 'Die Amsel', 'Die Schwalbe', 'Der Storch'], 'Dafür schwimmt er blitzschnell.'),
    f('Warum haben Eulen so weiche Federn?', ['Damit sie lautlos fliegen', 'Damit sie schwimmen können', 'Zum Kuscheln', 'Damit sie schneller sind'], 'Die Beute hört sie nicht kommen.'),
  ],
  // 5 - Insekten
  [
    f('Wie viele Beine hat ein Insekt?', ['Sechs', 'Vier', 'Acht', 'Zehn'], 'Daran erkennt man Insekten sofort.'),
    f('Was wird aus einer Raupe?', ['Ein Schmetterling', 'Ein Käfer', 'Eine Biene', 'Ein Wurm'], 'Dazwischen liegt die Zeit als Puppe.'),
    f('Was frisst ein Marienkäfer am liebsten?', ['Blattläuse', 'Blätter', 'Honig', 'Holz'], 'Deshalb freuen sich Gärtner über ihn.'),
    f('Welches Insekt fliegt blitzschnell über Teiche?', ['Die Libelle', 'Die Ameise', 'Der Floh', 'Die Laus'], 'Sie kann sogar rückwärts fliegen.'),
    f('Wie atmen Insekten?', ['Durch kleine Löcher am Körper', 'Mit einer Nase', 'Mit Kiemen', 'Gar nicht'], 'Die Löcher heißen Tracheen.'),
  ],
  // 6 - Bienen und Ameisen
  [
    f('Wer legt in einem Bienenvolk die Eier?', ['Nur die Königin', 'Alle Bienen', 'Die Arbeiterinnen', 'Die Drohnen'], 'Bis zu 2000 Eier an einem Tag.'),
    f('Woraus machen Bienen Honig?', ['Aus Nektar', 'Aus Wasser', 'Aus Erde', 'Aus Blättern'], 'Nektar ist der süße Saft der Blüten.'),
    f('Welche Form haben die Waben im Bienenstock?', ['Sechsecke', 'Kreise', 'Dreiecke', 'Quadrate'], 'Sechsecke sparen Wachs und Platz.'),
    f('Wie zeigen Bienen den Weg zu guten Blüten?', ['Mit einem Tanz', 'Mit Rufen', 'Mit einem Brief', 'Mit Duftwasser'], 'Der Schwänzeltanz verrät Richtung und Entfernung.'),
    f('Wie viel kann eine Ameise tragen?', ['Ein Vielfaches ihres eigenen Gewichts', 'Nur ein Blatt', 'Gar nichts', 'So viel wie ein Mensch'], 'Für ihre Größe sind Ameisen unglaublich stark.'),
  ],
  // 7 - Am Fluss und im See
  [
    f('Womit atmen Fische unter Wasser?', ['Mit Kiemen', 'Mit der Nase', 'Mit der Haut', 'Mit einer Lunge'], 'Kiemen holen Sauerstoff aus dem Wasser.'),
    f('Welches Tier baut Dämme aus Ästen?', ['Der Biber', 'Der Fischotter', 'Die Ente', 'Der Frosch'], 'Er fällt Bäume mit seinen Zähnen.'),
    f('Wie heißt ein junger Frosch am Anfang?', ['Kaulquappe', 'Fohlen', 'Küken', 'Larvenkäfer'], 'Erst später wachsen ihm Beine.'),
    f('Warum sind Enten im Wasser nicht nass?', ['Ihre Federn sind eingefettet', 'Sie haben Gummihaut', 'Sie trocknen sofort', 'Sie tauchen nicht'], 'Sie fetten die Federn mit dem Schnabel ein.'),
    f('Welches Tier hat eine Schale und lebt im Wasser?', ['Die Muschel', 'Die Maus', 'Der Specht', 'Die Biene'], 'Muscheln filtern das Wasser sauber.'),
  ],
  // 8 - Im Meer
  [
    f('Wie viele Arme hat ein Krake?', ['Acht', 'Vier', 'Sechs', 'Zwölf'], 'Er kann sie einzeln bewegen.'),
    f('Was ist ein Hai?', ['Ein Fisch', 'Ein Säugetier', 'Ein Vogel', 'Ein Reptil'], 'Sein Skelett ist aus biegsamem Knorpel.'),
    f('Welches Tier hat meistens fünf Arme?', ['Der Seestern', 'Die Qualle', 'Der Hering', 'Der Krebs'], 'Verliert er einen Arm, wächst er nach.'),
    f('Woraus besteht eine Qualle fast ganz?', ['Aus Wasser', 'Aus Luft', 'Aus Fett', 'Aus Knochen'], 'Über 95 Prozent sind Wasser.'),
    f('Wie bewegen sich Fische vorwärts?', ['Mit Flossen und Schwanz', 'Mit Beinen', 'Mit Flügeln', 'Mit Rädern'], 'Die Schwanzflosse gibt den Schub.'),
  ],
  // 9 - Wale und Delfine
  [
    f('Zu welcher Tiergruppe gehören Wale?', ['Zu den Säugetieren', 'Zu den Fischen', 'Zu den Vögeln', 'Zu den Reptilien'], 'Sie atmen Luft und säugen ihre Jungen.'),
    f('Wie lang wird ein Blauwal?', ['Bis zu 30 Meter', '3 Meter', '100 Meter', '300 Meter'], 'Er ist das größte Tier der Erde.'),
    f('Was frisst der Blauwal?', ['Winzige Krebse namens Krill', 'Große Fische', 'Seetang', 'Muscheln'], 'Er filtert sie mit seinen Barten aus dem Wasser.'),
    f('Wo atmen Wale?', ['Durch ein Loch auf dem Kopf', 'Durch den Mund', 'Durch die Haut', 'Durch die Flossen'], 'Beim Ausatmen entsteht die Fontäne.'),
    f('Wie finden Delfine ihre Beute?', ['Mit Klicklauten und Echo', 'Mit der Nase', 'Mit Licht', 'Durch Zufall'], 'Das Echo verrät ihnen Größe und Entfernung.'),
  ],
  // 10 - Afrika
  [
    f('Welches ist das größte Landtier der Welt?', ['Der Afrikanische Elefant', 'Das Nilpferd', 'Die Giraffe', 'Der Löwe'], 'Er wird über sechs Tonnen schwer.'),
    f('Wozu benutzt ein Elefant seinen Rüssel?', ['Zum Greifen, Trinken und Duschen', 'Nur zum Atmen', 'Zum Laufen', 'Zum Hören'], 'Im Rüssel stecken tausende Muskeln.'),
    f('Welches Tier hat den längsten Hals?', ['Die Giraffe', 'Das Zebra', 'Der Löwe', 'Das Krokodil'], 'Trotzdem hat sie nur sieben Halswirbel, wie du.'),
    f('Wie leben Löwen?', ['In einem Rudel', 'Immer allein', 'In einem Schwarm', 'In einem Stock'], 'Meist jagen die Löwinnen gemeinsam.'),
    f('Was ist bei jedem Zebra einmalig?', ['Sein Streifenmuster', 'Seine Größe', 'Seine Stimme', 'Seine Hufe'], 'Wie ein Fingerabdruck bei Menschen.'),
  ],
  // 11 - Im Dschungel
  [
    f('Was fressen die meisten Affen im Regenwald?', ['Früchte und Blätter', 'Nur Fleisch', 'Nur Fisch', 'Steine'], 'Viele Affen sind Allesfresser.'),
    f('Welcher Vogel kann Wörter nachsprechen?', ['Der Papagei', 'Die Taube', 'Die Möwe', 'Der Adler'], 'Er ahmt Geräusche nach, versteht sie aber nicht.'),
    f('Welche Katze ist die größte der Welt?', ['Der Tiger', 'Der Luchs', 'Der Leopard', 'Die Hauskatze'], 'Ein Sibirischer Tiger wiegt über 200 Kilo.'),
    f('Wie bewegen sich Gibbons durch die Bäume?', ['Sie schwingen sich an den Armen', 'Sie fliegen', 'Sie graben', 'Sie rollen'], 'Ihre Arme sind länger als die Beine.'),
    f('Warum sind viele Frösche im Regenwald so bunt?', ['Die Farbe warnt vor Gift', 'Damit sie hübsch aussehen', 'Damit man sie findet', 'Sie sind angemalt'], 'Auffällige Farben heißen: Finger weg!'),
  ],
  // 12 - In der Wüste
  [
    f('Was ist im Höcker eines Kamels?', ['Fett als Vorrat', 'Wasser', 'Luft', 'Knochen'], 'Aus dem Fett gewinnt es Energie.'),
    f('Warum hat der Wüstenfuchs Fennek so große Ohren?', ['Damit gibt er Wärme ab', 'Zum Fliegen', 'Zum Graben', 'Um besser zu riechen'], 'Die Ohren wirken wie eine Kühlung.'),
    f('Wann sind viele Wüstentiere unterwegs?', ['Nachts, wenn es kühl ist', 'Mittags in der Hitze', 'Nur im Winter', 'Nie'], 'Tagsüber verstecken sie sich im Schatten.'),
    f('Wie kommen Wüstentiere an Wasser?', ['Oft über ihre Nahrung', 'Aus dem Wasserhahn', 'Aus dem Meer', 'Sie brauchen keins'], 'Pflanzen und Beute enthalten Feuchtigkeit.'),
    f('Welches Tier hat einen Giftstachel am Schwanz?', ['Der Skorpion', 'Das Kamel', 'Die Wüstenmaus', 'Der Käfer'], 'Er jagt vor allem nachts.'),
  ],
  // 13 - Eis und Schnee
  [
    f('Wo leben Eisbären?', ['In der Arktis am Nordpol', 'Am Südpol', 'In Afrika', 'In Australien'], 'Pinguine und Eisbären treffen sich nie.'),
    f('Was hält Pinguine im Eis warm?', ['Eine dicke Fettschicht und dichte Federn', 'Ein Pullover', 'Warmes Wasser', 'Die Sonne'], 'Sie rücken zum Wärmen eng zusammen.'),
    f('Welche Farbe hat die Haut eines Eisbären?', ['Schwarz', 'Weiß', 'Rosa', 'Blau'], 'Die schwarze Haut speichert Wärme gut.'),
    f('Was fressen Robben am liebsten?', ['Fisch', 'Gras', 'Beeren', 'Blätter'], 'Sie tauchen dafür tief und lange.'),
    f('Welches Tier zieht im Norden große Herden?', ['Das Rentier', 'Der Löwe', 'Das Zebra', 'Der Tiger'], 'Rentiere wandern weite Strecken.'),
  ],
  // 14 - Australien
  [
    f('Was trägt ein Känguru in seinem Beutel?', ['Sein Junges', 'Futter', 'Steine', 'Wasser'], 'Ein Känguru-Baby ist bei der Geburt winzig.'),
    f('Was frisst ein Koala fast ausschließlich?', ['Eukalyptusblätter', 'Fleisch', 'Bananen', 'Gras'], 'Deshalb schläft er bis zu 20 Stunden am Tag.'),
    f('Wie bewegen sich Kängurus fort?', ['Sie hüpfen', 'Sie fliegen', 'Sie schwimmen', 'Sie rollen'], 'Der Schwanz hilft beim Gleichgewicht.'),
    f('Welches Tier legt Eier, ist aber ein Säugetier?', ['Das Schnabeltier', 'Der Koala', 'Das Känguru', 'Der Dingo'], 'Es lebt in Australien und schwimmt gerne.'),
    f('Wie heißt der wilde Hund Australiens?', ['Dingo', 'Wombat', 'Emu', 'Koala'], 'Dingos leben oft in kleinen Rudeln.'),
  ],
  // 15 - Tierkinder
  [
    f('Wie heißt ein junges Schaf?', ['Lamm', 'Ferkel', 'Kalb', 'Welpe'], 'Lämmer kommen meist im Frühling zur Welt.'),
    f('Wie heißt ein junges Schwein?', ['Ferkel', 'Fohlen', 'Küken', 'Kitz'], 'Ferkel werden oft zu acht oder mehr geboren.'),
    f('Wie heißt ein junges Rind?', ['Kalb', 'Lamm', 'Welpe', 'Küken'], 'Ein Kalb trinkt Milch bei der Mutter.'),
    f('Wie heißt ein junges Reh?', ['Kitz', 'Ferkel', 'Fohlen', 'Lamm'], 'Kitze liegen zum Schutz reglos im Gras.'),
    f('Wie heißt ein junger Vogel im Nest?', ['Küken', 'Kitz', 'Kalb', 'Welpe'], 'Am Anfang sind viele Küken nackt und blind.'),
  ],
  // 16 - Tierfamilien und Gruppen
  [
    f('Wie heißt eine Gruppe von Wölfen?', ['Rudel', 'Schwarm', 'Volk', 'Herde'], 'Im Rudel jagen und spielen sie gemeinsam.'),
    f('Wie heißt eine große Gruppe Fische?', ['Schwarm', 'Rudel', 'Herde', 'Volk'], 'Im Schwarm ist jeder Einzelne sicherer.'),
    f('Wie heißt die Gruppe aller Bienen in einem Stock?', ['Volk', 'Rudel', 'Herde', 'Schwarm'], 'Ein Volk kann 50000 Tiere haben.'),
    f('Wie heißt eine Gruppe von Kühen auf der Weide?', ['Herde', 'Rudel', 'Volk', 'Schwarm'], 'Herdentiere fühlen sich allein unwohl.'),
    f('Wie nennt man das männliche Pferd?', ['Hengst', 'Stute', 'Fohlen', 'Bulle'], 'Die weiblichen heißen Stuten.'),
  ],
  // 17 - Winter und Wanderung
  [
    f('Welches Tier hält einen echten Winterschlaf?', ['Der Igel', 'Das Reh', 'Der Fuchs', 'Die Amsel'], 'Sein Herz schlägt dabei sehr langsam.'),
    f('Warum ziehen Zugvögel im Herbst fort?', ['Weil hier das Futter fehlt', 'Weil sie sich langweilen', 'Weil es zu hell ist', 'Sie verirren sich'], 'Im Winter gibt es kaum Insekten.'),
    f('Wohin fliegen viele Störche im Winter?', ['Nach Afrika', 'Nach Grönland', 'Zum Nordpol', 'Nach Sibirien'], 'Das sind rund 10000 Kilometer.'),
    f('Wie finden Zugvögel den weiten Weg?', ['Mit dem Magnetfeld der Erde und der Sonne', 'Mit einer Landkarte', 'Sie fragen andere', 'Mit dem Handy'], 'Ein eingebauter Kompass hilft ihnen.'),
    f('Wie heißen Vögel, die den Winter hier bleiben?', ['Standvögel', 'Zugvögel', 'Wintervögel', 'Nachtvögel'], 'Meisen und Amseln gehören dazu.'),
  ],
  // 18 - Tarnung und Verstecken
  [
    f('Warum ist der Schneehase im Winter weiß?', ['Damit er im Schnee nicht auffällt', 'Weil ihm kalt ist', 'Weil er alt wird', 'Er wäscht sich'], 'Im Sommer wird sein Fell wieder braun.'),
    f('Was macht ein Chamäleon besonders gut?', ['Es kann seine Farbe ändern', 'Es kann fliegen', 'Es kann schwimmen', 'Es kann sprechen'], 'Die Farbe zeigt auch die Stimmung.'),
    f('Wie schützt sich ein Igel bei Gefahr?', ['Er rollt sich zur Stachelkugel', 'Er rennt weg', 'Er beißt', 'Er klettert'], 'Er hat etwa 8000 Stacheln.'),
    f('Was macht ein Tintenfisch bei Gefahr?', ['Er stößt eine Tintenwolke aus', 'Er wird rot', 'Er schläft ein', 'Er ruft laut'], 'In der Wolke verschwindet er blitzschnell.'),
    f('Warum haben viele Raupen Warnfarben?', ['Damit Vögel sie in Ruhe lassen', 'Damit man sie sieht', 'Weil sie krank sind', 'Damit sie wärmer sind'], 'Grelle Farben bedeuten oft: ungenießbar.'),
  ],
  // 19 - Schnell und langsam
  [
    f('Welches ist das schnellste Landtier?', ['Der Gepard', 'Der Elefant', 'Das Nilpferd', 'Die Maus'], 'Er schafft kurz über 100 Kilometer pro Stunde.'),
    f('Welches Tier ist besonders langsam?', ['Das Faultier', 'Der Hase', 'Das Pferd', 'Der Hund'], 'Es bewegt sich nur wenige Meter pro Minute.'),
    f('Welches Tier trägt sein Haus mit sich?', ['Die Schnecke', 'Der Hase', 'Der Frosch', 'Die Maus'], 'Bei Gefahr zieht sie sich hinein zurück.'),
    f('Wie schnell schlägt ein Kolibri mit den Flügeln?', ['Über 50 Mal pro Sekunde', 'Zwei Mal pro Sekunde', 'Ein Mal pro Minute', 'Gar nicht'], 'Deshalb kann er in der Luft stehen bleiben.'),
    f('Welcher Vogel ist der schnellste im Sturzflug?', ['Der Wanderfalke', 'Die Taube', 'Die Ente', 'Der Spatz'], 'Er erreicht über 300 Kilometer pro Stunde.'),
  ],
  // 20 - Riesen und Zwerge
  [
    f('Welches Tier ist das größte der Welt?', ['Der Blauwal', 'Der Elefant', 'Die Giraffe', 'Der Hai'], 'Sein Herz ist so groß wie ein kleines Auto.'),
    f('Welches Tier ist am höchsten gewachsen?', ['Die Giraffe', 'Der Elefant', 'Das Kamel', 'Der Bär'], 'Bis zu sechs Meter hoch.'),
    f('Wie schwer wird ein Afrikanischer Elefant?', ['Über sechs Tonnen', '200 Kilo', '50 Kilo', '100 Tonnen'], 'Das ist so viel wie vier Autos.'),
    f('Welches Tier ist winzig und lebt in Betten?', ['Die Hausstaubmilbe', 'Die Maus', 'Der Käfer', 'Die Spinne'], 'Sie ist ohne Lupe nicht zu sehen.'),
    f('Welcher Fisch ist der größte im Meer?', ['Der Walhai', 'Der Hering', 'Der Lachs', 'Der Goldfisch'], 'Er wird über zwölf Meter lang und frisst Plankton.'),
  ],
  // 21 - Reptilien
  [
    f('Warum liegen Reptilien gern in der Sonne?', ['Sie brauchen Wärme von außen', 'Sie mögen die Farbe', 'Sie schlafen dabei', 'Sie werden braun'], 'Sie können ihre Körperwärme nicht selbst machen.'),
    f('Was machen Schlangen, wenn sie wachsen?', ['Sie häuten sich', 'Sie werden bunt', 'Sie verlieren Zähne', 'Sie schrumpfen'], 'Die alte Haut wird komplett abgestreift.'),
    f('Womit fängt ein Chamäleon Insekten?', ['Mit seiner langen Zunge', 'Mit den Füßen', 'Mit dem Schwanz', 'Mit den Augen'], 'Die Zunge schnellt blitzschnell hervor.'),
    f('Wie lange kann ein Krokodil unter Wasser bleiben?', ['Sehr lange, oft über eine Stunde', 'Zehn Sekunden', 'Eine Minute', 'Gar nicht'], 'Beim Ruhen braucht es kaum Sauerstoff.'),
    f('Was schützt eine Schildkröte?', ['Ihr harter Panzer', 'Ihre Zähne', 'Ihre Krallen', 'Ihre Schnelligkeit'], 'Der Panzer ist mit dem Skelett verwachsen.'),
  ],
  // 22 - Amphibien
  [
    f('Wo legen Frösche ihre Eier ab?', ['Im Wasser', 'Auf Bäumen', 'Im Sand', 'In der Luft'], 'Der Laich sieht aus wie Gelee mit Punkten.'),
    f('Was wächst einer Kaulquappe zuerst?', ['Die Hinterbeine', 'Flügel', 'Federn', 'Zähne'], 'Danach kommen die Vorderbeine.'),
    f('Womit atmen erwachsene Frösche auch?', ['Über die Haut', 'Über die Augen', 'Über den Schwanz', 'Gar nicht'], 'Deshalb muss ihre Haut feucht bleiben.'),
    f('Wie fängt ein Frosch eine Fliege?', ['Mit seiner klebrigen Zunge', 'Mit den Händen', 'Mit einem Netz', 'Mit dem Schwanz'], 'Die Zunge schnellt in Sekundenbruchteilen vor.'),
    f('Welches Tier ist eine Amphibie?', ['Der Salamander', 'Die Eidechse', 'Die Schlange', 'Die Maus'], 'Amphibien leben im Wasser und an Land.'),
  ],
  // 23 - Spinnen und Krabbeltiere
  [
    f('Wie viele Beine hat eine Spinne?', ['Acht', 'Sechs', 'Vier', 'Zehn'], 'Deshalb ist sie kein Insekt.'),
    f('Wozu baut eine Spinne ihr Netz?', ['Um Beute zu fangen', 'Zum Schlafen', 'Als Schmuck', 'Zum Fliegen'], 'Spinnenseide ist erstaunlich reißfest.'),
    f('Welches Tier hat besonders viele Beinpaare?', ['Der Tausendfüßer', 'Die Fliege', 'Die Biene', 'Der Frosch'], 'Tausend sind es aber nie.'),
    f('Was macht ein Regenwurm für den Boden?', ['Er lockert ihn und macht ihn fruchtbar', 'Er frisst Wurzeln', 'Er trocknet ihn aus', 'Nichts'], 'Seine Gänge lassen Luft und Wasser hinein.'),
    f('Warum kommen Regenwürmer bei Regen hoch?', ['Ihre Gänge laufen voll Wasser', 'Sie wollen duschen', 'Sie suchen Sonne', 'Sie spielen'], 'An der Oberfläche bekommen sie besser Luft.'),
  ],
  // 24 - Sinne der Tiere
  [
    f('Wie finden Fledermäuse im Dunkeln ihren Weg?', ['Mit Echo aus ihren Rufen', 'Mit den Augen', 'Mit dem Geruch', 'Mit Licht'], 'Sie rufen und hören, was zurückkommt.'),
    f('Welches Tier riecht besonders gut?', ['Der Hund', 'Der Goldfisch', 'Die Schnecke', 'Der Frosch'], 'Ein Hund riecht millionenfach besser als wir.'),
    f('Wo hat eine Schnecke ihre Augen?', ['An den Spitzen der langen Fühler', 'Am Bauch', 'Am Haus', 'Am Fuß'], 'Bei Berührung zieht sie die Fühler ein.'),
    f('Was hören Elefanten, das wir nicht hören?', ['Sehr tiefe Töne', 'Sehr hohe Töne', 'Farben', 'Gedanken'], 'So verständigen sie sich über weite Strecken.'),
    f('Wie schmecken Schmetterlinge ihr Futter?', ['Mit den Füßen', 'Mit den Flügeln', 'Mit den Augen', 'Mit den Fühlern allein'], 'Sie treten auf eine Blüte und schmecken sofort.'),
  ],
  // 25 - Großes Tier-Finale
  [
    f('Welches Tier ist ein Säugetier?', ['Der Delfin', 'Der Hai', 'Der Frosch', 'Die Schildkröte'], 'Er atmet Luft und trinkt Milch als Baby.'),
    f('Wie viele Beine hat ein Käfer?', ['Sechs', 'Acht', 'Vier', 'Zwei'], 'Käfer sind Insekten.'),
    f('Welches Tier baut Dämme?', ['Der Biber', 'Der Dachs', 'Der Igel', 'Der Storch'], 'So staut er Wasser für seine Burg.'),
    f('Welches Tier hält Winterschlaf?', ['Der Igel', 'Die Kuh', 'Das Huhn', 'Die Katze'], 'Dafür frisst er sich im Herbst dick.'),
    f('Was fressen Blauwale?', ['Krill', 'Haie', 'Seetang', 'Vögel'], 'Winzige Krebse, dafür viele Millionen davon.'),
  ],
];
