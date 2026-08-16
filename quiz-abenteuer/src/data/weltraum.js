import { f } from './frage.js';

/** Weltraum-Welt: 25 Level mit je 5 Fragen. Erste Antwort = richtig. */
export const WELTRAUM_LEVEL = [
  // 1 - Die Sonne
  [
    f('Was ist die Sonne?', ['Ein Stern', 'Ein Planet', 'Ein Mond', 'Eine Wolke'], 'Sie ist der Stern, der uns am nächsten ist.'),
    f('Wie lange braucht das Sonnenlicht bis zur Erde?', ['Etwa 8 Minuten', 'Eine Sekunde', 'Einen Tag', 'Ein Jahr'], 'Licht ist das Schnellste, was es gibt.'),
    f('Wie viele Erden würden in die Sonne passen?', ['Über eine Million', 'Zehn', 'Hundert', 'Zwei'], 'Die Sonne ist gewaltig groß.'),
    f('Darf man in die Sonne schauen?', ['Nein, das schadet den Augen', 'Ja, kurz', 'Ja, mit der Sonnenbrille', 'Nur mittags'], 'Auch nicht mit einem Fernglas.'),
    f('Was schenkt uns die Sonne?', ['Licht und Wärme', 'Regen', 'Wind', 'Schatten'], 'Ohne sie gäbe es kein Leben auf der Erde.'),
  ],
  // 2 - Der Mond
  [
    f('Was ist der Mond?', ['Der Begleiter der Erde', 'Ein Stern', 'Ein Planet', 'Eine Wolke'], 'Er umkreist die Erde.'),
    f('Warum leuchtet der Mond?', ['Die Sonne bescheint ihn', 'Er brennt', 'Er hat Lampen', 'Er ist heiß'], 'Er wirft das Sonnenlicht zurück.'),
    f('Woher kommen die Krater auf dem Mond?', ['Von eingeschlagenen Gesteinsbrocken', 'Von Vulkanen der Erde', 'Von Regen', 'Von Astronauten'], 'Ohne Luft wird nichts abgebremst.'),
    f('Gibt es Luft auf dem Mond?', ['Nein', 'Ja, wie bei uns', 'Nur nachts', 'Nur an den Polen'], 'Deshalb braucht man dort einen Raumanzug.'),
    f('Was passiert mit Fußspuren auf dem Mond?', ['Sie bleiben sehr lange erhalten', 'Sie verwehen sofort', 'Sie werden weggeregnet', 'Sie leuchten'], 'Es gibt dort keinen Wind und keinen Regen.'),
  ],
  // 3 - Unsere Erde
  [
    f('Der wievielte Planet von der Sonne ist die Erde?', ['Der dritte', 'Der erste', 'Der fünfte', 'Der achte'], 'Vor ihr liegen Merkur und Venus.'),
    f('Warum heißt die Erde der blaue Planet?', ['Weil so viel Wasser sie bedeckt', 'Wegen des Himmels', 'Wegen der Wale', 'Wegen der Kälte'], 'Etwa zwei Drittel sind Ozean.'),
    f('Was schützt uns wie eine Hülle?', ['Die Lufthülle der Erde', 'Ein Glasdach', 'Die Wolken', 'Der Mond'], 'Sie heißt Atmosphäre.'),
    f('Wie lange braucht die Erde für eine Drehung?', ['24 Stunden', 'Eine Stunde', 'Ein Jahr', 'Eine Woche'], 'Dadurch entstehen Tag und Nacht.'),
    f('Was macht die Erde in einem Jahr?', ['Sie umrundet die Sonne einmal', 'Sie dreht sich einmal', 'Sie steht still', 'Sie wächst'], 'Dafür braucht sie 365 Tage.'),
  ],
  // 4 - Merkur und Venus
  [
    f('Welcher Planet ist der Sonne am nächsten?', ['Merkur', 'Venus', 'Erde', 'Mars'], 'Er ist auch der kleinste Planet.'),
    f('Welcher Planet ist der heißeste?', ['Venus', 'Merkur', 'Mars', 'Neptun'], 'Ihre dicken Wolken stauen die Hitze.'),
    f('Warum sieht man die Venus abends so hell?', ['Ihre Wolken spiegeln viel Sonnenlicht', 'Sie brennt', 'Sie ist ganz nah', 'Sie hat Lampen'], 'Man nennt sie auch Abendstern.'),
    f('Wie viele Monde hat der Merkur?', ['Keinen', 'Einen', 'Zwei', 'Zwanzig'], 'Auch die Venus hat keinen Mond.'),
    f('Wie ist die Oberfläche des Merkur?', ['Voller Krater', 'Voller Wasser', 'Voller Pflanzen', 'Voller Eis'], 'Sie sieht dem Mond sehr ähnlich.'),
  ],
  // 5 - Der Mars
  [
    f('Wie nennt man den Mars auch?', ['Der rote Planet', 'Der blaue Planet', 'Der grüne Planet', 'Der Ringplanet'], 'Sein Boden enthält viel Rost.'),
    f('Wie viele Monde hat der Mars?', ['Zwei', 'Keinen', 'Einen', 'Zehn'], 'Sie heißen Phobos und Deimos.'),
    f('Was steht auf dem Mars und ist riesig?', ['Der größte Vulkan im Sonnensystem', 'Ein Wald', 'Ein Ozean', 'Eine Stadt'], 'Der Olympus Mons ist fast 22 Kilometer hoch.'),
    f('Was suchen Forscher auf dem Mars?', ['Spuren von Wasser und Leben', 'Gold', 'Öl', 'Menschen'], 'Früher floss dort wahrscheinlich Wasser.'),
    f('Wie erforscht man den Mars heute?', ['Mit Robotern, die herumfahren', 'Mit Hunden', 'Mit Booten', 'Gar nicht'], 'Diese Roboter heißen Rover.'),
  ],
  // 6 - Der Jupiter
  [
    f('Welcher Planet ist der größte?', ['Jupiter', 'Saturn', 'Erde', 'Mars'], 'Alle anderen Planeten passen in ihn hinein.'),
    f('Was ist der Große Rote Fleck auf dem Jupiter?', ['Ein riesiger Sturm', 'Ein See', 'Ein Berg', 'Ein Wald'], 'Er tobt schon seit hunderten Jahren.'),
    f('Woraus besteht der Jupiter vor allem?', ['Aus Gas', 'Aus Fels', 'Aus Eis', 'Aus Wasser'], 'Man könnte auf ihm nicht landen.'),
    f('Wie viele Monde hat der Jupiter etwa?', ['Über 90', 'Einen', 'Fünf', 'Keinen'], 'Die vier größten sah schon Galileo Galilei.'),
    f('Wie heißt ein bekannter Mond des Jupiter?', ['Europa', 'Titan', 'Phobos', 'Charon'], 'Unter seinem Eis vermutet man einen Ozean.'),
  ],
  // 7 - Der Saturn
  [
    f('Wofür ist der Saturn berühmt?', ['Für seine Ringe', 'Für seine Farbe', 'Für seine Hitze', 'Für sein Wasser'], 'Die Ringe sind schon im Fernrohr zu sehen.'),
    f('Woraus bestehen die Ringe des Saturn?', ['Aus Eis- und Gesteinsbrocken', 'Aus festem Metall', 'Aus Gas', 'Aus Staubfarbe'], 'Manche Brocken sind winzig, andere hausgroß.'),
    f('Was für ein Planet ist der Saturn?', ['Ein Gasplanet', 'Ein Gesteinsplanet', 'Ein Eisplanet', 'Ein Zwergplanet'], 'Wie Jupiter hat er keine feste Oberfläche.'),
    f('Wie heißt der größte Mond des Saturn?', ['Titan', 'Europa', 'Luna', 'Deimos'], 'Er hat sogar eine dichte Lufthülle.'),
    f('Der wievielte Planet von der Sonne ist Saturn?', ['Der sechste', 'Der dritte', 'Der achte', 'Der erste'], 'Direkt nach dem Jupiter.'),
  ],
  // 8 - Uranus und Neptun
  [
    f('Was ist beim Uranus besonders?', ['Er liegt gekippt auf der Seite', 'Er ist rot', 'Er hat keine Monde', 'Er ist der heißeste'], 'Er rollt regelrecht um die Sonne.'),
    f('Welcher Planet ist der äußerste?', ['Neptun', 'Uranus', 'Saturn', 'Pluto'], 'Er ist sehr weit von der Sonne entfernt.'),
    f('Welche Farbe hat der Neptun?', ['Blau', 'Rot', 'Gelb', 'Grün'], 'Das liegt am Gas Methan in seiner Lufthülle.'),
    f('Was gibt es auf dem Neptun besonders heftig?', ['Stürme mit über 1000 km/h', 'Regen aus Wasser', 'Vulkane', 'Wälder'], 'Es sind die schnellsten Winde im Sonnensystem.'),
    f('Wie ist es auf Uranus und Neptun?', ['Eisig kalt', 'Angenehm warm', 'Heiß wie ein Ofen', 'Wie auf der Erde'], 'Die Sonne ist von dort nur ein heller Punkt.'),
  ],
  // 9 - Sterne
  [
    f('Was sind Sterne?', ['Riesige leuchtende Gaskugeln', 'Kleine Lampen', 'Löcher im Himmel', 'Spiegel'], 'Sie erzeugen ihr Licht selbst.'),
    f('Welcher Stern ist uns am nächsten?', ['Die Sonne', 'Der Polarstern', 'Sirius', 'Der Mond'], 'Alle anderen sind unvorstellbar weit weg.'),
    f('Warum funkeln Sterne?', ['Die Luft über uns flimmert', 'Sie blinken selbst', 'Sie drehen sich', 'Sie sind kaputt'], 'Im Weltraum funkeln sie nicht.'),
    f('Was verrät die Farbe eines Sterns?', ['Wie heiß er ist', 'Wie alt der Himmel ist', 'Wie weit er weg ist', 'Wie groß das All ist'], 'Blaue Sterne sind heißer als rote.'),
    f('Wann sieht man die meisten Sterne?', ['In einer dunklen, klaren Nacht', 'Am Mittag', 'Bei Regen', 'Im Nebel'], 'Stadtlicht überstrahlt viele Sterne.'),
  ],
  // 10 - Sternbilder
  [
    f('Was ist ein Sternbild?', ['Eine Figur aus Sternen am Himmel', 'Ein Foto vom Himmel', 'Ein Planet', 'Eine Rakete'], 'Die Sterne stehen in Wirklichkeit weit auseinander.'),
    f('Welches Sternbild sieht aus wie ein Wagen?', ['Der Große Wagen', 'Der Orion', 'Der Stier', 'Der Schwan'], 'Er gehört zum Sternbild Großer Bär.'),
    f('Welcher Stern zeigt immer nach Norden?', ['Der Polarstern', 'Die Sonne', 'Der Mond', 'Die Venus'], 'Früher navigierten Seefahrer nach ihm.'),
    f('Wie findet man den Polarstern?', ['Über die hinteren Sterne des Großen Wagens', 'Er ist der hellste', 'Er blinkt rot', 'Er steht im Süden'], 'Fünfmal der Abstand zeigt genau hin.'),
    f('Welches Sternbild hat drei Sterne als Gürtel?', ['Orion', 'Großer Wagen', 'Kleiner Bär', 'Krebs'], 'Im Winter ist es gut zu sehen.'),
  ],
  // 11 - Die Milchstraße
  [
    f('Wie heißt unsere Galaxie?', ['Milchstraße', 'Sonnensystem', 'Andromeda', 'Universum'], 'Unsere Sonne ist einer ihrer Sterne.'),
    f('Wie viele Sterne hat die Milchstraße etwa?', ['Hunderte Milliarden', 'Tausend', 'Eine Million', 'Zehn'], 'So viele kann niemand einzeln zählen.'),
    f('Wie sieht die Milchstraße am Nachthimmel aus?', ['Wie ein helles Band', 'Wie ein Punkt', 'Wie ein Kreis', 'Wie ein Stern'], 'Man sieht sie nur weit weg von Stadtlichtern.'),
    f('Was ist eine Galaxie?', ['Eine riesige Ansammlung von Sternen', 'Ein großer Planet', 'Ein heller Mond', 'Eine Rakete'], 'Im All gibt es Milliarden davon.'),
    f('Was ist das Universum?', ['Alles, was es gibt', 'Nur unsere Sonne', 'Nur die Erde', 'Ein Sternbild'], 'Es ist unvorstellbar groß.'),
  ],
  // 12 - Unser Sonnensystem
  [
    f('Wie viele Planeten hat unser Sonnensystem?', ['Acht', 'Sechs', 'Zehn', 'Zwölf'], 'Von Merkur bis Neptun.'),
    f('Welcher Planet kommt direkt nach der Erde?', ['Mars', 'Venus', 'Jupiter', 'Merkur'], 'Die Reihenfolge nach außen ist Erde, Mars, Jupiter.'),
    f('Was ist Pluto heute?', ['Ein Zwergplanet', 'Der neunte Planet', 'Ein Mond', 'Ein Stern'], 'Seit 2006 zählt er nicht mehr als Planet.'),
    f('Was steht im Mittelpunkt unseres Sonnensystems?', ['Die Sonne', 'Die Erde', 'Der Mond', 'Der Jupiter'], 'Alle Planeten kreisen um sie.'),
    f('Welche Planeten sind Gasriesen?', ['Jupiter und Saturn', 'Merkur und Venus', 'Erde und Mars', 'Erde und Mond'], 'Auch Uranus und Neptun haben keine feste Oberfläche.'),
  ],
  // 13 - Raketen
  [
    f('Warum fliegt eine Rakete nach oben?', ['Sie stößt heiße Gase nach unten aus', 'Sie hat Flügel', 'Der Wind schiebt sie', 'Sie ist leicht'], 'Rückstoß nennt man dieses Prinzip.'),
    f('Warum haben Raketen mehrere Stufen?', ['Leere Teile werden abgeworfen', 'Damit sie hübsch aussehen', 'Für mehr Sitzplätze', 'Damit sie leiser sind'], 'Ohne Ballast fliegt der Rest leichter weiter.'),
    f('Wo startet eine Rakete?', ['Von einer Startrampe', 'Vom Flughafen', 'Vom Schiff im Hafen', 'Vom Berg'], 'Startplätze liegen oft nahe am Äquator.'),
    f('Was passiert beim Start mit den Astronauten?', ['Sie werden stark in den Sitz gedrückt', 'Sie schweben sofort', 'Sie schlafen', 'Sie stehen auf'], 'Die Beschleunigung ist enorm.'),
    f('Wie schnell muss eine Rakete sein, um die Erde zu umkreisen?', ['Etwa 28000 Kilometer pro Stunde', '100 Kilometer pro Stunde', '1000 Kilometer pro Stunde', 'So schnell wie ein Auto'], 'Sonst fällt sie wieder herunter.'),
  ],
  // 14 - Astronauten
  [
    f('Warum tragen Astronauten einen Raumanzug?', ['Im All gibt es keine Luft', 'Damit sie schön aussehen', 'Gegen Regen', 'Gegen Insekten'], 'Der Anzug liefert Luft und schützt vor Kälte.'),
    f('Wie bewegen sich Astronauten in der Raumstation?', ['Sie schweben', 'Sie gehen normal', 'Sie fahren Rad', 'Sie kriechen'], 'Alles muss festgeschnallt werden.'),
    f('Wie schlafen Astronauten?', ['In einem festgemachten Schlafsack', 'Auf einer Matratze', 'Im Sitzen am Fenster', 'Gar nicht'], 'Sonst würden sie durch die Station treiben.'),
    f('Warum müssen Astronauten viel Sport machen?', ['Ohne Schwerkraft werden Muskeln schwach', 'Aus Langeweile', 'Um abzunehmen', 'Für den Wettkampf'], 'Sie trainieren mehrere Stunden am Tag.'),
    f('Wie trinken Astronauten im All?', ['Aus Beuteln mit Strohhalm', 'Aus offenen Bechern', 'Aus dem Wasserhahn', 'Gar nicht'], 'Wasser würde sonst als Kugel davonschweben.'),
  ],
  // 15 - Die Raumstation ISS
  [
    f('Wofür steht die Abkürzung ISS?', ['Internationale Raumstation', 'Italienische Sternwarte', 'Innere Sonnenstation', 'Interstellares Schiff'], 'Viele Länder bauten sie gemeinsam.'),
    f('Wie hoch fliegt die ISS über der Erde?', ['Etwa 400 Kilometer', '10 Kilometer', '4000 Kilometer', 'Eine Million Kilometer'], 'Man kann sie abends von der Erde sehen.'),
    f('Wie lange braucht die ISS für eine Erdumrundung?', ['Etwa 90 Minuten', 'Einen Tag', 'Eine Woche', 'Ein Jahr'], 'Sie rast mit 28000 Kilometern pro Stunde.'),
    f('Wie oft sehen Astronauten dort einen Sonnenaufgang?', ['Etwa 16 Mal am Tag', 'Einmal am Tag', 'Nie', 'Einmal pro Woche'], 'Weil sie die Erde so schnell umrunden.'),
    f('Was machen Astronauten auf der ISS?', ['Forschen und Experimente machen', 'Nur schlafen', 'Urlaub machen', 'Autos bauen'], 'Schwerelosigkeit ermöglicht besondere Versuche.'),
  ],
  // 16 - Die Mondlandung
  [
    f('In welchem Jahr landeten zum ersten Mal Menschen auf dem Mond?', ['1969', '1999', '1929', '2009'], 'Millionen Menschen sahen es im Fernsehen.'),
    f('Wie hieß der erste Mensch auf dem Mond?', ['Neil Armstrong', 'Juri Gagarin', 'Alexander Gerst', 'Galileo Galilei'], 'Er sagte einen berühmten Satz beim Aussteigen.'),
    f('Wie hieß die Mission der ersten Mondlandung?', ['Apollo 11', 'Sputnik 1', 'Voyager 2', 'Artemis'], 'Die Landefähre hieß Eagle, also Adler.'),
    f('Wie viele Menschen waren bisher auf dem Mond?', ['Zwölf', 'Zwei', 'Hundert', 'Tausend'], 'Alle kamen zwischen 1969 und 1972 dorthin.'),
    f('Wie bewegten sich die Astronauten auf dem Mond?', ['Sie hüpften', 'Sie rannten', 'Sie schwammen', 'Sie flogen'], 'Dort zieht die Schwerkraft sechsmal schwächer.'),
  ],
  // 17 - Satelliten
  [
    f('Was ist ein Satellit?', ['Ein Gerät, das die Erde umkreist', 'Ein Stern', 'Eine Rakete', 'Ein Planet'], 'Auch der Mond ist ein natürlicher Satellit.'),
    f('Wofür brauchen wir Wettersatelliten?', ['Für die Wettervorhersage', 'Für das Internet allein', 'Für Musik', 'Für Strom'], 'Sie fotografieren Wolken von oben.'),
    f('Was hilft dem Navi im Auto?', ['Satelliten im All', 'Das Radio', 'Die Straßenschilder', 'Der Motor'], 'Mehrere Satelliten bestimmen den Ort.'),
    f('Wie bleiben Satelliten oben?', ['Sie fliegen sehr schnell um die Erde', 'Sie hängen an Seilen', 'Sie schweben von selbst', 'Ballons halten sie'], 'Ihr Fall geht ständig an der Erde vorbei.'),
    f('Was ist Weltraumschrott?', ['Alte Teile, die im All herumfliegen', 'Müll auf der Erde', 'Staub auf dem Mond', 'Ein Komet'], 'Er ist gefährlich für Raumfahrzeuge.'),
  ],
  // 18 - Kometen und Sternschnuppen
  [
    f('Woraus besteht ein Komet?', ['Aus Eis und Staub', 'Aus Metall', 'Aus Gas allein', 'Aus Wasser'], 'Man nennt ihn auch schmutzigen Schneeball.'),
    f('Wann bekommt ein Komet einen Schweif?', ['Wenn er nahe an die Sonne kommt', 'Immer', 'Nur nachts', 'Wenn er landet'], 'Die Sonnenwärme lässt das Eis verdampfen.'),
    f('Was ist eine Sternschnuppe wirklich?', ['Ein verglühendes Staubkorn', 'Ein fallender Stern', 'Ein Flugzeug', 'Ein Satellit'], 'Sie leuchtet in der Lufthülle auf.'),
    f('Wie heißt ein Brocken, der auf der Erde landet?', ['Meteorit', 'Komet', 'Planet', 'Satellit'], 'Viele findet man in der Wüste oder im Eis.'),
    f('Wann sieht man besonders viele Sternschnuppen?', ['Wenn die Erde durch eine Staubspur fliegt', 'Bei Vollmond', 'Im Regen', 'Am Mittag'], 'Im August sind es die Perseiden.'),
  ],
  // 19 - Asteroiden
  [
    f('Wo liegt der Asteroidengürtel?', ['Zwischen Mars und Jupiter', 'Zwischen Erde und Mond', 'Hinter Neptun allein', 'Um die Sonne herum ganz innen'], 'Dort kreisen unzählige Gesteinsbrocken.'),
    f('Was sind Asteroiden?', ['Gesteinsbrocken im Weltraum', 'Kleine Sterne', 'Monde der Erde', 'Wolken'], 'Manche sind nur metergroß, andere hunderte Kilometer.'),
    f('Was passierte vor 66 Millionen Jahren?', ['Ein großer Asteroid traf die Erde', 'Der Mond entstand', 'Die Sonne erlosch', 'Ein Vulkan flog ins All'], 'Danach starben die großen Dinosaurier aus.'),
    f('Wie schützt man die Erde vor Asteroiden?', ['Man sucht sie mit Teleskopen', 'Mit einem Zaun', 'Mit Regenschirmen', 'Gar nicht'], 'Forscher haben sogar schon einen abgelenkt.'),
    f('Woraus bestehen Asteroiden?', ['Aus Gestein und Metall', 'Aus Eis allein', 'Aus Gas', 'Aus Sand'], 'Manche enthalten viel Eisen.'),
  ],
  // 20 - Teleskope
  [
    f('Wozu dient ein Teleskop?', ['Um weit entfernte Dinge zu vergrößern', 'Zum Fliegen', 'Zum Messen der Zeit', 'Zum Funken'], 'Damit sieht man Krater auf dem Mond.'),
    f('Wer schaute als einer der Ersten mit einem Fernrohr zum Himmel?', ['Galileo Galilei', 'Neil Armstrong', 'Albert Einstein', 'Isaac Newton'], 'Er entdeckte vier Monde des Jupiter.'),
    f('Warum stehen große Teleskope oft auf Bergen?', ['Dort ist die Luft klarer', 'Dort ist es wärmer', 'Wegen der Aussicht', 'Dort ist mehr Platz'], 'Weniger Luft bedeutet ein schärferes Bild.'),
    f('Wie heißt das berühmte Teleskop im Weltall?', ['Hubble', 'Sputnik', 'Apollo', 'Ariane'], 'Es umkreist die Erde seit 1990.'),
    f('Warum ist ein Teleskop im All besser?', ['Keine Luft stört das Bild', 'Es ist näher an den Sternen', 'Es ist billiger', 'Es ist größer'], 'Dort funkeln die Sterne nicht.'),
  ],
  // 21 - Schwerkraft
  [
    f('Was hält uns auf dem Boden?', ['Die Schwerkraft', 'Der Luftdruck', 'Der Wind', 'Unsere Schuhe'], 'Die Erde zieht alles zu sich.'),
    f('Wie stark zieht der Mond im Vergleich zur Erde?', ['Etwa sechsmal schwächer', 'Genauso stark', 'Doppelt so stark', 'Gar nicht'], 'Deshalb hüpften die Astronauten dort.'),
    f('Warum schweben Astronauten in der Raumstation?', ['Sie fallen ständig um die Erde herum', 'Dort gibt es keine Schwerkraft', 'Sie sind sehr leicht', 'Sie tragen Ballons'], 'Man nennt es Schwerelosigkeit.'),
    f('Was hält die Planeten auf ihrer Bahn?', ['Die Anziehungskraft der Sonne', 'Ein Seil', 'Der Wind im All', 'Nichts'], 'Sonst würden sie geradeaus davonfliegen.'),
    f('Was passiert mit Wasser in der Schwerelosigkeit?', ['Es bildet schwebende Kugeln', 'Es fällt nach unten', 'Es wird fest', 'Es verschwindet'], 'Die Oberflächenspannung formt Kugeln.'),
  ],
  // 22 - Tag, Jahr und Jahreszeiten
  [
    f('Wodurch entstehen Tag und Nacht?', ['Durch die Drehung der Erde', 'Durch Wolken', 'Weil die Sonne ausgeht', 'Durch den Mond'], 'Eine Umdrehung dauert 24 Stunden.'),
    f('Wie lange braucht die Erde um die Sonne?', ['365 Tage', '24 Stunden', '30 Tage', '10 Jahre'], 'Das ist genau ein Jahr.'),
    f('Warum gibt es Jahreszeiten?', ['Die Erdachse ist geneigt', 'Die Erde kommt der Sonne näher', 'Der Mond schiebt', 'Wegen der Wolken'], 'Mal trifft die Sonne steiler, mal flacher auf.'),
    f('Warum hat ein Schaltjahr einen Tag mehr?', ['Ein Jahr dauert etwas länger als 365 Tage', 'Weil der Februar kurz ist', 'Zur Erinnerung', 'Wegen des Mondes'], 'Alle vier Jahre gleicht man das aus.'),
    f('Wo geht die Sonne auf?', ['Im Osten', 'Im Westen', 'Im Norden', 'Im Süden'], 'Weil sich die Erde nach Osten dreht.'),
  ],
  // 23 - Mondphasen und Finsternisse
  [
    f('Wie heißt der ganz runde volle Mond?', ['Vollmond', 'Neumond', 'Halbmond', 'Sichelmond'], 'Dann steht die Erde zwischen Sonne und Mond.'),
    f('Was ist Neumond?', ['Der Mond ist nicht zu sehen', 'Der Mond ist ganz rund', 'Der Mond ist rot', 'Der Mond ist weg'], 'Seine beleuchtete Seite zeigt von uns weg.'),
    f('Wie lange dauert ein Mondzyklus etwa?', ['Etwa 29 Tage', 'Eine Woche', 'Ein Jahr', 'Einen Tag'], 'Von Vollmond zu Vollmond.'),
    f('Was passiert bei einer Sonnenfinsternis?', ['Der Mond schiebt sich vor die Sonne', 'Die Sonne geht aus', 'Die Erde dreht sich rückwärts', 'Wolken decken die Sonne zu'], 'Es wird am Tag für kurze Zeit dunkel.'),
    f('Was passiert bei einer Mondfinsternis?', ['Der Schatten der Erde fällt auf den Mond', 'Der Mond zerbricht', 'Der Mond fliegt weg', 'Die Sonne verdeckt den Mond'], 'Der Mond leuchtet dabei oft rötlich.'),
  ],
  // 24 - Roboter im Weltraum
  [
    f('Was ist ein Rover?', ['Ein Fahrzeug, das fremde Planeten erkundet', 'Eine Rakete', 'Ein Teleskop', 'Ein Raumanzug'], 'Auf dem Mars fahren gleich mehrere.'),
    f('Warum schickt man erst Roboter statt Menschen?', ['Es ist sicherer und günstiger', 'Roboter sind klüger', 'Menschen wollen nicht', 'Roboter sind schneller'], 'Roboter brauchen weder Luft noch Essen.'),
    f('Wie bekommen viele Raumsonden ihre Energie?', ['Von Solarzellen', 'Von Benzin', 'Von Batterien aus dem Laden', 'Vom Wind'], 'Weit draußen reicht das Sonnenlicht kaum.'),
    f('Wie lange braucht ein Funkspruch zum Mars?', ['Mehrere Minuten', 'Eine Sekunde', 'Einen Tag', 'Ein Jahr'], 'Deshalb müssen Rover vieles allein entscheiden.'),
    f('Was macht eine Raumsonde?', ['Sie fliegt zu fernen Zielen und funkt Daten', 'Sie bringt Menschen zum Mond', 'Sie putzt Satelliten', 'Sie bleibt auf der Erde'], 'Voyager fliegt schon seit 1977.'),
  ],
  // 25 - Großes Weltraum-Finale
  [
    f('Wie viele Planeten hat unser Sonnensystem?', ['Acht', 'Neun', 'Sieben', 'Zwölf'], 'Pluto zählt als Zwergplanet.'),
    f('Welcher Planet hat die auffälligen Ringe?', ['Saturn', 'Mars', 'Merkur', 'Erde'], 'Sie bestehen aus Eis und Gestein.'),
    f('Was ist die Sonne?', ['Ein Stern', 'Ein Planet', 'Ein Komet', 'Ein Mond'], 'Sie ist die Energiequelle für alles Leben.'),
    f('Wer betrat 1969 als Erster den Mond?', ['Neil Armstrong', 'Juri Gagarin', 'Galileo Galilei', 'Alexander Gerst'], 'Mit der Mission Apollo 11.'),
    f('Was ist eine Sternschnuppe?', ['Ein verglühendes Staubkorn', 'Ein herabfallender Stern', 'Ein Satellit', 'Ein Komet'], 'Sie leuchtet hoch oben in der Luft auf.'),
  ],
];
