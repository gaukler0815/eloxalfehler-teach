import { f } from './frage.js';

/**
 * Dino-Welt: 25 Level mit je 5 Fragen.
 * Erste Antwort = richtige Antwort (wird beim Anzeigen gemischt).
 */
export const DINO_LEVEL = [
  // 1 - Der Tyrannosaurus Rex
  [
    f('Was bedeutet der Name "Tyrannosaurus Rex"?', ['König der Tyrannen-Echsen', 'Schneller Jäger', 'Großer Zahn', 'Alter Riese'], 'Rex ist das lateinische Wort für König.'),
    f('Wie lang war ein T-Rex ungefähr?', ['12 Meter', '2 Meter', '30 Meter', '80 Meter'], 'So lang wie ein großer Bus.'),
    f('Was war am T-Rex besonders klein?', ['Seine Arme', 'Seine Beine', 'Sein Kopf', 'Sein Schwanz'], 'Die Arme waren kaum länger als deine.'),
    f('Wie viele Zähne hatte ein T-Rex ungefähr?', ['60', '6', '600', '12'], 'Manche Zähne waren so lang wie eine Banane.'),
    f('Was konnte der T-Rex besonders gut?', ['Riechen', 'Schwimmen', 'Fliegen', 'Klettern'], 'Sein Geruchssinn war besser als der von einem Hund.'),
  ],
  // 2 - Pflanzenfresser
  [
    f('Was fressen Pflanzenfresser?', ['Blätter und Pflanzen', 'Fische', 'Andere Dinos', 'Steine'], 'Manche fraßen jeden Tag hunderte Kilo Grünzeug.'),
    f('Welcher Dino hatte einen sehr langen Hals?', ['Brachiosaurus', 'T-Rex', 'Velociraptor', 'Ankylosaurus'], 'Sein Hals war allein etwa 9 Meter lang.'),
    f('Warum schluckten manche Dinos Steine?', ['Zum Zerkleinern des Futters im Magen', 'Weil sie Hunger hatten', 'Um schwerer zu werden', 'Zum Spielen'], 'Diese Magensteine heißen Gastrolithen.'),
    f('Welche Zähne haben Pflanzenfresser?', ['Flache Mahlzähne', 'Lange spitze Zähne', 'Gar keine Zähne', 'Zähne aus Gold'], 'Damit zermahlen sie harte Blätter.'),
    f('Warum hatten viele Pflanzenfresser lange Hälse?', ['Um an hohe Blätter zu kommen', 'Zum Schwimmen', 'Um zu fliegen', 'Um schneller zu rennen'], 'So kamen sie an Futter, das kein anderer erreichte.'),
  ],
  // 3 - Fleischfresser
  [
    f('Wie sehen die Zähne von Fleischfressern aus?', ['Spitz und scharf', 'Flach und breit', 'Rund wie Kugeln', 'Weich wie Gummi'], 'Mit spitzen Zähnen hält man Beute gut fest.'),
    f('Welcher Dino war ein großer Räuber?', ['Allosaurus', 'Brachiosaurus', 'Stegosaurus', 'Triceratops'], 'Der Allosaurus hatte Hörnchen über den Augen.'),
    f('Auf wie vielen Beinen liefen die großen Räuber?', ['Auf zwei', 'Auf vier', 'Auf sechs', 'Auf acht'], 'So hatten sie die Vorderbeine zum Zupacken frei.'),
    f('Welcher Räuber fraß am liebsten Fisch?', ['Spinosaurus', 'Triceratops', 'Stegosaurus', 'Diplodocus'], 'Sein Maul sah aus wie das eines Krokodils.'),
    f('Was fraßen Räuber, wenn sie nichts jagen konnten?', ['Tiere, die schon tot waren', 'Blätter', 'Steine', 'Nichts, sie schliefen'], 'Auch Aasfressen war ganz normal.'),
  ],
  // 4 - Panzer und Stacheln
  [
    f('Was hatte der Ankylosaurus am Schwanz?', ['Eine Knochenkeule', 'Eine Flosse', 'Ein Horn', 'Eine Feder'], 'Damit konnte er kräftig zuschlagen.'),
    f('Was stand auf dem Rücken des Stegosaurus?', ['Große Knochenplatten', 'Weiches Fell', 'Bunte Federn', 'Ein Segel aus Haut'], 'Er trug etwa 17 davon.'),
    f('Welche Körperstelle war beim Ankylosaurus weich?', ['Der Bauch', 'Der Rücken', 'Der Kopf', 'Der Schwanz'], 'Deshalb blieb er immer dicht am Boden.'),
    f('Woraus bestand der Panzer der Dinos?', ['Aus Knochenplatten', 'Aus Holz', 'Aus Stein', 'Aus Metall'], 'Die Platten wuchsen direkt in der Haut.'),
    f('Wozu waren die Rückenplatten wahrscheinlich noch gut?', ['Zum Wärmen und Abkühlen', 'Zum Fliegen', 'Zum Graben', 'Zum Schwimmen'], 'Wie ein Heizkörper an der Wand.'),
  ],
  // 5 - Die Langhälse
  [
    f('Wie hoch war ein Brachiosaurus ungefähr?', ['13 Meter', '3 Meter', '40 Meter', '90 Meter'], 'So hoch wie ein Haus mit vier Stockwerken.'),
    f('Welcher Dino war einer der schwersten überhaupt?', ['Argentinosaurus', 'Velociraptor', 'Compsognathus', 'Archaeopteryx'], 'Er wog so viel wie zwölf Elefanten.'),
    f('Was war beim Diplodocus besonders lang?', ['Der Schwanz', 'Die Ohren', 'Die Arme', 'Die Zunge'], 'Der Schwanz knallte wie eine Peitsche.'),
    f('Was mussten Langhälse jeden Tag tun?', ['Sehr viel fressen', 'Sehr weit fliegen', 'Tief tauchen', 'Bäume fällen'], 'Ein großer Körper braucht viel Futter.'),
    f('Wie sah der Kopf eines Langhalses aus?', ['Erstaunlich klein', 'Riesengroß', 'Rund wie ein Ball', 'Flach wie ein Teller'], 'Der Kopf war winzig im Vergleich zum Körper.'),
  ],
  // 6 - Flugsaurier
  [
    f('Waren Flugsaurier echte Dinosaurier?', ['Nein, nur Verwandte', 'Ja, alle', 'Ja, die kleinen', 'Sie waren Vögel'], 'Sie lebten aber zur gleichen Zeit.'),
    f('Woraus bestanden die Flügel der Flugsaurier?', ['Aus einer Flughaut', 'Aus Federn', 'Aus Fell', 'Aus Schuppen'], 'Die Haut spannte sich über einen langen Finger.'),
    f('Wie breit waren die Flügel des Quetzalcoatlus?', ['Bis zu 11 Meter', '1 Meter', '30 Meter', '100 Meter'], 'So breit wie ein kleines Flugzeug.'),
    f('Was fingen viele Flugsaurier über dem Meer?', ['Fische', 'Blätter', 'Steine', 'Schmetterlinge'], 'Sie flogen dicht über dem Wasser.'),
    f('Welcher Flugsaurier hatte einen großen Kopfkamm?', ['Pteranodon', 'Diplodocus', 'Ankylosaurus', 'Triceratops'], 'Vielleicht half der Kamm beim Steuern.'),
  ],
  // 7 - Saurier im Meer
  [
    f('Was war beim Plesiosaurus besonders lang?', ['Der Hals', 'Der Schnabel', 'Die Beine', 'Die Ohren'], 'Er hatte vier Flossen und einen langen Hals.'),
    f('Wie sah der Ichthyosaurus aus?', ['Wie ein Delfin', 'Wie ein Krebs', 'Wie eine Qualle', 'Wie ein Vogel'], 'Er war aber ein Reptil, kein Fisch.'),
    f('Wie atmeten die Meeressaurier?', ['Sie atmeten Luft', 'Mit Kiemen', 'Gar nicht', 'Durch die Haut'], 'Sie mussten zum Atmen auftauchen.'),
    f('Wie hieß der größte Jäger der Urzeit-Meere?', ['Mosasaurus', 'Stegosaurus', 'Pteranodon', 'Iguanodon'], 'Er wurde über 15 Meter lang.'),
    f('Womit schwammen die Meeressaurier?', ['Mit Flossen', 'Mit Flügeln', 'Mit Hufen', 'Mit Rädern'], 'Ihre Beine waren zu Flossen geworden.'),
  ],
  // 8 - Eier und Dino-Babys
  [
    f('Wie kamen junge Dinos auf die Welt?', ['Sie schlüpften aus Eiern', 'Sie wurden geboren', 'Sie wuchsen an Bäumen', 'Aus dem Wasser'], 'Alle Dinosaurier legten Eier.'),
    f('Wie lang ist das größte gefundene Dino-Ei?', ['60 Zentimeter', '6 Zentimeter', '2 Meter', '5 Meter'], 'Für so einen großen Körper ist das erstaunlich klein.'),
    f('Warum konnten Dino-Eier nicht riesig sein?', ['Sonst kommt keine Luft durch die Schale', 'Sie wären zu teuer', 'Sie wären zu bunt', 'Sie würden schwimmen'], 'Das Baby im Ei braucht Luft.'),
    f('Was bedeutet der Name Maiasaura?', ['Gute Mutter-Echse', 'Große Echse', 'Schnelle Echse', 'Nest-Räuber'], 'Bei ihren Nestern fand man Jungtiere.'),
    f('Wo legten viele Dinos ihre Eier ab?', ['In einer Mulde im Boden', 'Auf Bäumen', 'Im tiefen Wasser', 'In Höhlen aus Eis'], 'Oft deckten sie das Nest mit Pflanzen zu.'),
  ],
  // 9 - Rekorde
  [
    f('Welcher Dino war so groß wie ein Huhn?', ['Compsognathus', 'Brachiosaurus', 'Triceratops', 'Spinosaurus'], 'Er war einer der kleinsten Dinos.'),
    f('Welcher Räuber war noch länger als der T-Rex?', ['Spinosaurus', 'Velociraptor', 'Stegosaurus', 'Protoceratops'], 'Er wurde bis zu 15 Meter lang.'),
    f('Was war am Diplodocus besonders?', ['Er war sehr lang', 'Er war sehr klein', 'Er konnte fliegen', 'Er lebte im Eis'], 'Über 25 Meter, meist Hals und Schwanz.'),
    f('Wie schwer war ein Argentinosaurus etwa?', ['Wie zwölf Elefanten', 'Wie ein Auto', 'Wie ein Fahrrad', 'Wie ein Pferd'], 'Bei jedem Schritt zitterte der Boden.'),
    f('Welcher Dino hatte die längsten Krallen?', ['Therizinosaurus', 'Triceratops', 'Stegosaurus', 'Pteranodon'], 'Seine Krallen waren so lang wie dein Unterarm.'),
  ],
  // 10 - Zähne und Fressen
  [
    f('Was passierte, wenn einem Dino ein Zahn abbrach?', ['Ein neuer wuchs nach', 'Er blieb ohne Zahn', 'Er starb', 'Er bekam einen Goldzahn'], 'Dinos wechselten ihr Leben lang Zähne.'),
    f('Wie kauten die großen Langhälse ihr Futter?', ['Fast gar nicht, sie schluckten es', 'Sehr lange', 'Mit den Füßen', 'Mit dem Schwanz'], 'Im Magen wurde es dann zerkleinert.'),
    f('Wozu hatte der Triceratops einen Schnabel?', ['Zum Abreißen von Pflanzen', 'Zum Fischen', 'Zum Graben von Höhlen', 'Zum Singen'], 'Dahinter saßen hunderte Mahlzähne.'),
    f('Welcher Dino hatte über tausend Zähne?', ['Der Hadrosaurier', 'Der T-Rex', 'Der Pteranodon', 'Der Velociraptor'], 'Die Zähne standen dicht wie eine Raspel.'),
    f('Woran erkennen Forscher, was ein Dino gefressen hat?', ['An der Form der Zähne', 'An der Farbe der Knochen', 'An der Größe der Augen', 'Am Namen'], 'Spitze Zähne heißt Fleisch, flache heißt Pflanzen.'),
  ],
  // 11 - Fossilien
  [
    f('Was ist ein Fossil?', ['Ein versteinerter Rest aus alter Zeit', 'Ein moderner Knochen', 'Ein Stück Holz', 'Ein Vogelnest'], 'Meistens sind es Knochen oder Zähne.'),
    f('Was muss passieren, damit ein Fossil entsteht?', ['Das Tier wird schnell zugedeckt', 'Es muss schneien', 'Ein Forscher muss zusehen', 'Es muss brennen'], 'Am besten mit Sand oder Schlamm.'),
    f('Was bringt das Wasser in die Knochen?', ['Mineralien', 'Luft', 'Farbe', 'Sand'], 'Die Mineralien machen den Knochen hart wie Stein.'),
    f('Wie lange dauert es, bis ein Fossil entsteht?', ['Viele Millionen Jahre', 'Eine Woche', 'Ein Jahr', 'Hundert Jahre'], 'Fossilien sind unvorstellbar alt.'),
    f('Können auch Fußspuren Fossilien sein?', ['Ja', 'Nein, nur Knochen', 'Nur bei Vögeln', 'Nur im Wasser'], 'Spuren verraten sogar, wie schnell ein Dino lief.'),
  ],
  // 12 - Forscher und Museen
  [
    f('Wie heißen Forscher, die Dinos ausgraben?', ['Paläontologen', 'Astronauten', 'Archäologen der Ägypter', 'Meteorologen'], 'Sie erforschen Leben aus der Urzeit.'),
    f('Womit legen Forscher ein Fossil vorsichtig frei?', ['Mit Pinseln', 'Mit dem Bagger', 'Mit Wasser aus dem Schlauch', 'Mit Feuer'], 'Grobe Werkzeuge würden die Knochen zerbrechen.'),
    f('Wie heißt das bekannteste T-Rex-Skelett?', ['Sue', 'Rexi', 'Anna', 'Bruno'], 'Es steht in einem Museum in Chicago.'),
    f('Was machen Forscher, wenn Knochen fehlen?', ['Sie ergänzen sie als Nachbildung', 'Sie lassen das Skelett weg', 'Sie erfinden neue Tiere', 'Sie warten 100 Jahre'], 'Nachbildungen werden extra gekennzeichnet.'),
    f('Warum sind Museen für Dino-Forschung wichtig?', ['Dort werden Fossilien aufbewahrt und erforscht', 'Dort leben echte Dinos', 'Dort werden Dinos gezüchtet', 'Dort schlafen die Forscher'], 'Sammlungen wachsen über viele Jahrzehnte.'),
  ],
  // 13 - Die Zeit der Dinos
  [
    f('Wie lange lebten die Dinosaurier auf der Erde?', ['Über 160 Millionen Jahre', '100 Jahre', '2000 Jahre', 'Eine Million Tage'], 'Viel länger als es Menschen gibt.'),
    f('Wann starben die großen Dinos aus?', ['Vor 66 Millionen Jahren', 'Vor 200 Jahren', 'Vor 5000 Jahren', 'Letztes Jahr'], 'Das ist das Ende der Kreidezeit.'),
    f('Lebten Menschen zusammen mit den Dinos?', ['Nein, nie', 'Ja, immer', 'Nur in Afrika', 'Nur die Kinder'], 'Zwischen ihnen liegen viele Millionen Jahre.'),
    f('Wie heißt der letzte Zeitabschnitt der Dinos?', ['Kreidezeit', 'Eiszeit', 'Steinzeit', 'Bronzezeit'], 'Davor kamen Jura und Trias.'),
    f('Welcher Dino lebte in der Jurazeit?', ['Stegosaurus', 'Triceratops', 'T-Rex', 'Ankylosaurus'], 'T-Rex und Triceratops kamen erst viel später.'),
  ],
  // 14 - Raptoren
  [
    f('Wie groß war ein Velociraptor wirklich?', ['Etwa so groß wie ein Truthahn', 'So groß wie ein Elefant', 'So groß wie ein Bus', 'So groß wie ein Haus'], 'In Filmen wird er viel größer gezeigt.'),
    f('Wo saß die große Sichelkralle der Raptoren?', ['Am Hinterfuß', 'Am Kopf', 'Am Schwanz', 'Am Rücken'], 'Beim Laufen hielten sie die Kralle hoch.'),
    f('Was hatten Raptoren am Körper?', ['Federn', 'Schuppenpanzer', 'Fell', 'Stacheln'], 'Fliegen konnten sie damit trotzdem nicht.'),
    f('Wofür brauchte ein Raptor seinen langen Schwanz?', ['Für das Gleichgewicht', 'Zum Graben', 'Zum Schwimmen', 'Zum Klettern'], 'Beim schnellen Rennen hielt er ihn steif.'),
    f('Wie schnell konnte ein Velociraptor etwa rennen?', ['40 Kilometer pro Stunde', '5 Kilometer pro Stunde', '200 Kilometer pro Stunde', '300 Kilometer pro Stunde'], 'Schneller als ein Fahrrad in der Stadt.'),
  ],
  // 15 - Hörner und Kämme
  [
    f('Wie viele Hörner hatte der Triceratops im Gesicht?', ['Drei', 'Eins', 'Fünf', 'Zehn'], 'Zwei über den Augen, eins auf der Nase.'),
    f('Was konnte der Parasaurolophus mit seinem Kamm?', ['Laute Töne machen', 'Fliegen', 'Graben', 'Wasser speichern'], 'Der Kamm war innen hohl wie eine Trompete.'),
    f('Was war beim Pachycephalosaurus besonders dick?', ['Das Schädeldach', 'Der Schwanz', 'Die Zunge', 'Die Ohren'], 'So dick wie ein Ziegelstein.'),
    f('Was hatte der Styracosaurus am Nackenschild?', ['Lange Stacheln', 'Weiche Federn', 'Kleine Augen', 'Blätter'], 'Sechs große Stacheln standen wie eine Krone ab.'),
    f('Wozu dienten Hörner und Kämme wahrscheinlich?', ['Zum Angeben und zur Verteidigung', 'Zum Fliegen', 'Zum Schwimmen', 'Zum Schlafen'], 'Auffällige Köpfe beeindrucken Artgenossen.'),
  ],
  // 16 - Dino-Namen
  [
    f('Was bedeutet die Endung "-saurus"?', ['Echse', 'Zahn', 'Groß', 'Schnell'], 'Fast alle Dino-Namen enden darauf.'),
    f('Was bedeutet "Triceratops"?', ['Dreihorn-Gesicht', 'Großer Panzer', 'Schneller Läufer', 'Langer Hals'], 'Tri heißt drei.'),
    f('Was bedeutet "Stegosaurus"?', ['Dach-Echse', 'Feuer-Echse', 'Wasser-Echse', 'Berg-Echse'], 'Man dachte, die Platten lägen wie Dachziegel.'),
    f('Was bedeutet "Velociraptor"?', ['Schneller Räuber', 'Kleiner Freund', 'Gefiederter König', 'Starker Beißer'], 'Velox heißt schnell.'),
    f('Wer darf einem neuen Dino den Namen geben?', ['Die Forscher, die ihn beschreiben', 'Der Bürgermeister', 'Die Kinder im Museum', 'Niemand'], 'Oft steckt der Fundort im Namen.'),
  ],
  // 17 - Federn und Vögel
  [
    f('Welche Tiere stammen von den Dinosauriern ab?', ['Die Vögel', 'Die Fische', 'Die Frösche', 'Die Katzen'], 'Ein Huhn ist ein weit entfernter Verwandter des T-Rex.'),
    f('Wie heißt der berühmte Urvogel?', ['Archaeopteryx', 'Pteranodon', 'Diplodocus', 'Mosasaurus'], 'Er hatte Federn und gleichzeitig Zähne.'),
    f('Wozu dienten Federn bei kleinen Dinos zuerst?', ['Zum Warmhalten', 'Zum Fliegen', 'Zum Schwimmen', 'Zum Graben'], 'Fliegen kam erst viel später dazu.'),
    f('Wie viele gefiederte Flügel hatte der Microraptor?', ['Vier', 'Zwei', 'Sechs', 'Keine'], 'Auch an den Beinen saßen lange Federn.'),
    f('Haben alle Dinos Federn gehabt?', ['Nein, nur ein Teil von ihnen', 'Ja, alle', 'Nur die ganz großen', 'Nur die im Wasser'], 'Viele hatten schuppige Haut.'),
  ],
  // 18 - Fußspuren
  [
    f('Was verraten versteinerte Fußspuren?', ['Wie ein Dino lief', 'Welche Farbe er hatte', 'Wie er hieß', 'Was er dachte'], 'Aus dem Abstand der Spuren rechnet man die Geschwindigkeit aus.'),
    f('Was bedeuten viele Spuren nebeneinander?', ['Die Dinos liefen in einer Herde', 'Es war ein einzelner Dino', 'Dort war ein Fluss', 'Dort stand ein Baum'], 'Herden schützen vor Räubern.'),
    f('Wo bleiben Fußspuren am besten erhalten?', ['In weichem Schlamm, der austrocknet', 'In tiefem Wasser', 'Auf hartem Fels', 'Im Schnee'], 'Später wird der Schlamm zu Stein.'),
    f('Woran erkennt man die Spur eines Räubers?', ['An drei kräftigen Zehen mit Krallen', 'An runden Abdrücken', 'An fünf kleinen Zehen', 'An einem Loch'], 'Pflanzenfresser hinterließen oft runde Abdrücke.'),
    f('Wie groß können Dino-Fußspuren sein?', ['Größer als ein Autoreifen', 'So klein wie ein Cent', 'Immer gleich groß', 'So groß wie ein Fußballfeld'], 'In manche passt ein Kind hinein.'),
  ],
  // 19 - Wo die Dinos lebten
  [
    f('Auf welchen Kontinenten lebten Dinosaurier?', ['Auf allen', 'Nur in Afrika', 'Nur in Europa', 'Nur in Amerika'], 'Sogar in der Antarktis wurden welche gefunden.'),
    f('In welchem Land wurden viele Raptoren gefunden?', ['In der Mongolei', 'In Grönland', 'In Irland', 'In Japan'], 'Die Wüste Gobi ist ein berühmter Fundort.'),
    f('Warum findet man Fossilien oft in Wüsten?', ['Dort liegt wenig Pflanzen und Erde darüber', 'Weil Dinos Wüsten mochten', 'Weil es dort warm ist', 'Weil dort viele Menschen suchen'], 'Wind und Regen tragen den Stein ab.'),
    f('Wie sah die Erde zur Dino-Zeit aus?', ['Die Kontinente lagen anders', 'Genau wie heute', 'Alles war Eis', 'Es gab nur Wasser'], 'Erst hingen alle Kontinente zusammen.'),
    f('Wo wurde der Spinosaurus gefunden?', ['In der Sahara', 'Am Nordpol', 'In Australien', 'In der Schweiz'], 'Früher gab es dort breite Flüsse.'),
  ],
  // 20 - Das Ende der Dinos
  [
    f('Was traf vor 66 Millionen Jahren die Erde?', ['Ein riesiger Asteroid', 'Ein Regenschauer', 'Ein Schneesturm', 'Ein Flugzeug'], 'Der Einschlag veränderte die ganze Welt.'),
    f('Wo schlug der Asteroid ein?', ['In Mexiko', 'In Berlin', 'Am Nordpol', 'In Australien'], 'Dort liegt heute der Chicxulub-Krater.'),
    f('Warum starben danach so viele Tiere?', ['Staub verdunkelte die Sonne', 'Es wurde zu hell', 'Das Wasser wurde süß', 'Die Erde drehte sich rückwärts'], 'Ohne Sonne wuchsen kaum noch Pflanzen.'),
    f('Welche Dinosaurier überlebten bis heute?', ['Die Vögel', 'Die großen Räuber', 'Die Langhälse', 'Gar keine'], 'Jeder Spatz ist ein später Nachfahre.'),
    f('Wie nennt man so ein großes Artensterben?', ['Massenaussterben', 'Winterschlaf', 'Umzug', 'Verwandlung'], 'Auf der Erde gab es davon mehrere.'),
  ],
  // 21 - Kleine Dinos
  [
    f('Wie groß war der Compsognathus etwa?', ['Wie ein Huhn', 'Wie ein Pferd', 'Wie ein Bus', 'Wie eine Maus'], 'Er jagte Eidechsen und Insekten.'),
    f('Was fraßen die ganz kleinen Dinos?', ['Insekten und kleine Tiere', 'Große Dinos', 'Bäume', 'Steine'], 'Für große Beute waren sie zu schwach.'),
    f('Welchen Vorteil hatten kleine Dinos?', ['Sie konnten sich gut verstecken', 'Sie waren stärker', 'Sie konnten fliegen', 'Sie lebten länger'], 'Verstecken schützt vor Räubern.'),
    f('Was war der Microraptor?', ['Ein kleiner gefiederter Dino', 'Ein Meeressaurier', 'Ein Käfer', 'Ein Baum'], 'Er konnte von Baum zu Baum gleiten.'),
    f('Warum finden Forscher kleine Dinos seltener?', ['Ihre dünnen Knochen zerfallen leichter', 'Es gab keine kleinen Dinos', 'Sie versteckten sich', 'Sie lebten im Wasser'], 'Große Knochen bleiben besser erhalten.'),
  ],
  // 22 - Berühmte Funde
  [
    f('Was zeigt das Fossil "Die kämpfenden Dinosaurier"?', ['Zwei Dinos mitten im Kampf', 'Ein Nest voller Eier', 'Einen schlafenden Dino', 'Einen Fußabdruck'], 'Ein Sandsturm hat sie plötzlich zugedeckt.'),
    f('Wo steht das Skelett "Sue"?', ['In einem Museum in Chicago', 'Im Wald', 'Auf einem Schiff', 'In einer Schule'], 'Fast alle Knochen wurden gefunden.'),
    f('Was war einer der ersten Dino-Funde überhaupt?', ['Ein Zahn des Iguanodon', 'Ein ganzes Ei', 'Ein Fußabdruck vom T-Rex', 'Eine Feder'], 'Man hielt ihn zuerst für einen Nashornzahn.'),
    f('Was kann man in Bernstein finden?', ['Eingeschlossene Insekten und Federn', 'Ganze Dinos', 'Gold', 'Wasser'], 'Bernstein ist versteinertes Baumharz.'),
    f('Warum sind vollständige Skelette so selten?', ['Meist zerfallen oder verstreuen die Knochen', 'Forscher suchen zu wenig', 'Dinos hatten keine Knochen', 'Sie sind alle im Museum'], 'Oft findet man nur einzelne Teile.'),
  ],
  // 23 - Sinne und Gehirn
  [
    f('Wie groß war das Gehirn des Stegosaurus?', ['So groß wie eine Walnuss', 'So groß wie ein Fußball', 'So groß wie ein Auto', 'Er hatte keins'], 'Für sein Leben reichte das völlig.'),
    f('Warum sah der T-Rex räumlich sehr gut?', ['Seine Augen zeigten nach vorne', 'Er hatte vier Augen', 'Seine Augen waren blau', 'Er hatte eine Brille'], 'So schätzt man Entfernungen genau ein.'),
    f('Welcher Sinn war beim T-Rex besonders stark?', ['Der Geruchssinn', 'Der Geschmackssinn', 'Der Tastsinn', 'Das Farbsehen'], 'Der Riechbereich im Gehirn war riesig.'),
    f('Wozu dienten die hohlen Kämme mancher Dinos?', ['Zum Rufen über weite Strecken', 'Zum Atmen unter Wasser', 'Zum Wasserspeichern', 'Zum Klettern'], 'Die Rufe klangen wie tiefe Töne.'),
    f('Woher wissen Forscher etwas über Dino-Gehirne?', ['Aus der Form des Schädels innen', 'Aus alten Büchern', 'Von Fotos', 'Sie raten'], 'Der Hohlraum im Schädel zeigt die Form.'),
  ],
  // 24 - Irrtümer über Dinos
  [
    f('War das Mammut ein Dinosaurier?', ['Nein, es war ein Säugetier', 'Ja', 'Ja, ein kleiner', 'Es war ein Vogel'], 'Mammuts lebten erst lange nach den Dinos.'),
    f('Waren alle Dinosaurier riesig?', ['Nein, manche waren hühnergroß', 'Ja, alle', 'Ja, außer den Fliegenden', 'Alle waren gleich groß'], 'Die Größen reichten von Huhn bis Haus.'),
    f('Lebten alle Dinos zur gleichen Zeit?', ['Nein, oft Millionen Jahre auseinander', 'Ja, alle gleichzeitig', 'Ja, in einem Jahr', 'Nur die großen gleichzeitig'], 'Zwischen Stegosaurus und T-Rex liegen mehr Jahre als zwischen T-Rex und uns.'),
    f('Waren Meeressaurier Dinosaurier?', ['Nein', 'Ja', 'Nur die großen', 'Nur die kleinen'], 'Dinosaurier lebten an Land.'),
    f('Wissen wir, welche Farbe alle Dinos hatten?', ['Nein, nur bei wenigen', 'Ja, bei allen', 'Sie waren alle grün', 'Sie waren alle grau'], 'Bei einigen Federfunden kennt man die Farbe.'),
  ],
  // 25 - Großes Dino-Finale
  [
    f('Welcher Dino trug ein Segel auf dem Rücken?', ['Spinosaurus', 'Triceratops', 'Velociraptor', 'Brachiosaurus'], 'Das Segel war fast zwei Meter hoch.'),
    f('Welcher Dino hatte eine Keule am Schwanz?', ['Ankylosaurus', 'Diplodocus', 'Pteranodon', 'Compsognathus'], 'Ein Schlag konnte Beine brechen.'),
    f('Wie viele Hörner hatte Triceratops?', ['Drei', 'Zwei', 'Vier', 'Keins'], 'Sein Name bedeutet Dreihorn-Gesicht.'),
    f('Wodurch starben die Dinos aus?', ['Durch einen Asteroideneinschlag', 'Durch Regen', 'Durch Menschen', 'Durch Hunger im Winter'], 'Danach begann die Zeit der Säugetiere.'),
    f('Welches Tier ist heute mit den Dinos verwandt?', ['Das Huhn', 'Der Hund', 'Der Wal', 'Die Schlange'], 'Vögel sind die letzten lebenden Dinosaurier.'),
  ],
];
