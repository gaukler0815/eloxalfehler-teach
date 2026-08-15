/**
 * 15 Lese- und Verstaendnis-Einheiten (ungerade Level 1, 3, 5 ... 29).
 *
 * 10 Texte ueber Dinosaurier, 5 Texte ueber Tiere und Natur.
 * Kurze Saetze, Erstleser-Wortschatz, ca. 5-8 Minuten Lesezeit.
 * Zu jedem Text gehoeren 5 Fragen mit je 4 Antworten - 4 von 5 richtig
 * (80 %) sind zum Bestehen noetig.
 */

export const LESETEXTE = [
  // 1 -----------------------------------------------------------------
  {
    id: 'trex',
    kategorie: 'dino',
    titel: 'Der Tyrannosaurus Rex',
    absaetze: [
      'Der Tyrannosaurus Rex war ein riesiger Räuber. Sein Name bedeutet "König der Tyrannen-Echsen". Fast alle nennen ihn einfach T-Rex. Er lebte vor etwa 68 Millionen Jahren. Das ist unglaublich lange her. Damals gab es noch keine Menschen auf der Erde.',
      'Ein T-Rex war ungefähr 12 Meter lang. So lang ist ein großer Bus. Allein sein Kopf war größer als du. In seinem Maul saßen etwa 60 Zähne. Manche Zähne waren so lang wie eine Banane. Brach ein Zahn ab, wuchs einfach ein neuer nach.',
      'Der T-Rex lief auf zwei kräftigen Beinen. Seine Arme waren dagegen winzig. Sie waren kaum länger als deine Arme. Trotzdem waren diese Arme sehr stark. Forscher glauben, er konnte damit seine Beute festhalten.',
      'Am besten konnte der T-Rex riechen. Sein Geruchssinn war noch besser als der von einem Hund. So fand er sein Futter schon von weitem. Er fraß andere Dinosaurier. Manchmal jagte er selbst. Manchmal fraß er auch Tiere, die schon tot waren.',
      'Heute suchen Forscher seine Knochen in der Erde. Das bekannteste Skelett hat einen Namen: Sue. Es steht in einem Museum in Amerika. Von Sue wurden fast alle Knochen gefunden. Darum wissen wir heute sehr viel über den T-Rex.',
    ],
    fragen: [
      {
        frage: 'Wie lang war ein Tyrannosaurus Rex ungefähr?',
        optionen: ['2 Meter', '12 Meter', '50 Meter', '100 Meter'],
        richtig: 1,
        tipp: 'Im Text steht: so lang wie ein großer Bus.',
      },
      {
        frage: 'Wie viele Zähne hatte der T-Rex ungefähr?',
        optionen: ['6 Zähne', '16 Zähne', '60 Zähne', '600 Zähne'],
        richtig: 2,
        tipp: 'Schau noch einmal in den zweiten Absatz.',
      },
      {
        frage: 'Was war am T-Rex besonders klein?',
        optionen: ['Seine Arme', 'Seine Beine', 'Sein Kopf', 'Sein Schwanz'],
        richtig: 0,
        tipp: 'Etwas war kaum länger als deine Arme.',
      },
      {
        frage: 'Welchen Sinn hatte der T-Rex besonders gut?',
        optionen: ['Das Hören', 'Das Riechen', 'Das Schmecken', 'Das Fühlen'],
        richtig: 1,
        tipp: 'Er war darin sogar besser als ein Hund.',
      },
      {
        frage: 'Wie heißt das bekannteste T-Rex-Skelett?',
        optionen: ['Rexi', 'Anna', 'Sue', 'Bruno'],
        richtig: 2,
        tipp: 'Der Name steht im letzten Absatz.',
      },
    ],
  },

  // 2 -----------------------------------------------------------------
  {
    id: 'triceratops',
    kategorie: 'dino',
    titel: 'Der Triceratops',
    absaetze: [
      'Der Triceratops war ein Pflanzenfresser. Sein Name bedeutet "Dreihorn-Gesicht". Das passt genau: Er hatte drei Hörner im Gesicht. Zwei lange Hörner saßen über den Augen. Ein kürzeres Horn saß auf der Nase.',
      'Hinter dem Kopf trug er einen großen Nackenschild. Dieser Schild war aus Knochen. Er sah aus wie ein riesiger Kragen. Vielleicht schützte er den Hals. Vielleicht wollte der Triceratops damit auch angeben. Forscher sind sich noch nicht ganz sicher.',
      'Ein Triceratops war etwa 9 Meter lang. Er wog so viel wie vier Autos. Trotzdem war er kein Jäger. Er fraß nur Pflanzen. Mit seinem Schnabel riss er Farne und Blätter ab. In seinem Maul hatte er Hunderte Zähne zum Mahlen.',
      'Der Triceratops lief auf vier starken Beinen. Er war kein schneller Läufer. Wenn ein T-Rex kam, stellte er sich mit den Hörnern nach vorne. Forscher haben Knochen gefunden, an denen man Kämpfe erkennen kann.',
      'Viele Triceratops lebten wahrscheinlich in Gruppen. Junge Tiere blieben dann in der Mitte. Die großen Tiere standen außen. So waren die Kleinen gut geschützt. Der Triceratops lebte zur gleichen Zeit wie der T-Rex.',
    ],
    fragen: [
      {
        frage: 'Was bedeutet der Name Triceratops?',
        optionen: ['Dreihorn-Gesicht', 'Großer Räuber', 'Schneller Läufer', 'Langer Hals'],
        richtig: 0,
        tipp: 'Der Name steht gleich im ersten Absatz.',
      },
      {
        frage: 'Wo saß das kürzeste Horn?',
        optionen: ['Am Schwanz', 'Auf der Nase', 'Am Rücken', 'Über dem Ohr'],
        richtig: 1,
        tipp: 'Zwei Hörner waren über den Augen.',
      },
      {
        frage: 'Was fraß der Triceratops?',
        optionen: ['Fische', 'Andere Dinos', 'Nur Pflanzen', 'Eier'],
        richtig: 2,
        tipp: 'Er riss Farne und Blätter ab.',
      },
      {
        frage: 'Wozu diente vielleicht der Nackenschild?',
        optionen: ['Zum Schwimmen', 'Zum Schutz des Halses', 'Zum Fliegen', 'Zum Graben'],
        richtig: 1,
        tipp: 'Lies den zweiten Absatz noch einmal.',
      },
      {
        frage: 'Wo standen die jungen Tiere in der Gruppe?',
        optionen: ['Ganz vorne', 'Ganz hinten', 'In der Mitte', 'Weit weg'],
        richtig: 2,
        tipp: 'Die großen Tiere standen außen.',
      },
    ],
  },

  // 3 -----------------------------------------------------------------
  {
    id: 'biene',
    kategorie: 'natur',
    titel: 'Die Honigbiene',
    absaetze: [
      'Bienen sind kleine, fleißige Insekten. Sie leben in einem Bienenstock. In einem Stock wohnen sehr viele Bienen zusammen. Manchmal sind es 50000 Tiere. Sie bilden ein großes Volk.',
      'In jedem Volk gibt es eine Königin. Nur sie legt Eier. Jeden Tag legt sie bis zu 2000 Stück. Die anderen Bienen sind Arbeiterinnen. Sie putzen, bauen und sammeln Futter.',
      'Die Arbeiterinnen fliegen von Blüte zu Blüte. Dort sammeln sie süßen Nektar. Aus dem Nektar machen sie später Honig. Sie sammeln auch Blütenstaub. Er heißt Pollen und ist ihr Eiweiß-Futter.',
      'Beim Sammeln passiert etwas Wichtiges. Der Pollen bleibt am Körper der Biene kleben. Die Biene trägt ihn zur nächsten Blume. So können Pflanzen Früchte bilden. Ohne Bienen gäbe es viel weniger Äpfel und Kirschen.',
      'Bienen können sich sogar unterhalten. Dafür tanzen sie im Stock. Mit ihrem Tanz zeigen sie, wo es viele Blumen gibt. Man nennt das den Schwänzeltanz. Die anderen Bienen verstehen das genau und fliegen los.',
    ],
    fragen: [
      {
        frage: 'Wo leben Honigbienen zusammen?',
        optionen: ['Im Bienenstock', 'In einer Höhle', 'Unter der Erde', 'Im Wasser'],
        richtig: 0,
        tipp: 'Es steht im ersten Absatz.',
      },
      {
        frage: 'Wer legt in einem Bienenvolk die Eier?',
        optionen: ['Alle Bienen', 'Nur die Königin', 'Die Arbeiterinnen', 'Niemand'],
        richtig: 1,
        tipp: 'Es ist nur eine einzige Biene.',
      },
      {
        frage: 'Woraus machen Bienen Honig?',
        optionen: ['Aus Wasser', 'Aus Blättern', 'Aus Nektar', 'Aus Erde'],
        richtig: 2,
        tipp: 'Sie holen etwas Süßes aus den Blüten.',
      },
      {
        frage: 'Wie heißt der Blütenstaub noch?',
        optionen: ['Pollen', 'Wachs', 'Harz', 'Mehl'],
        richtig: 0,
        tipp: 'Der Name steht im dritten Absatz.',
      },
      {
        frage: 'Wie zeigen Bienen den Weg zu den Blumen?',
        optionen: ['Sie rufen laut', 'Sie tanzen', 'Sie malen', 'Sie summen ein Lied'],
        richtig: 1,
        tipp: 'Es heißt Schwänzeltanz.',
      },
    ],
  },

  // 4 -----------------------------------------------------------------
  {
    id: 'brachiosaurus',
    kategorie: 'dino',
    titel: 'Der Brachiosaurus',
    absaetze: [
      'Der Brachiosaurus war ein Riese. Er gehörte zu den größten Tieren, die je an Land lebten. Vom Boden bis zum Kopf war er 13 Meter hoch. Das ist so hoch wie ein Haus mit vier Stockwerken.',
      'Sein Hals war unglaublich lang. Allein der Hals maß etwa 9 Meter. Damit kam er an die höchsten Blätter. Kein anderes Tier konnte dort fressen. So hatte er sein Futter fast für sich allein.',
      'Der Brachiosaurus fraß nur Pflanzen. Jeden Tag brauchte er sehr viel davon. Forscher schätzen: bis zu 200 Kilogramm am Tag. Er kaute nicht lange. Er riss die Blätter ab und schluckte sie herunter.',
      'Sein Name bedeutet "Arm-Echse". Das liegt an seinen Beinen. Die vorderen Beine waren länger als die hinteren. Dadurch ging sein Rücken nach hinten abwärts. So sah er aus wie eine riesige Giraffe.',
      'Ein Brachiosaurus wog etwa 40 Tonnen. Das ist so schwer wie acht Elefanten. Seine Fußstapfen waren größer als ein Autoreifen. Bei jedem Schritt hat wahrscheinlich der Boden gezittert.',
    ],
    fragen: [
      {
        frage: 'Wie hoch war ein Brachiosaurus ungefähr?',
        optionen: ['3 Meter', '13 Meter', '30 Meter', '90 Meter'],
        richtig: 1,
        tipp: 'So hoch wie ein Haus mit vier Stockwerken.',
      },
      {
        frage: 'Was war an ihm besonders lang?',
        optionen: ['Der Schwanz', 'Die Zunge', 'Der Hals', 'Die Ohren'],
        richtig: 2,
        tipp: 'Damit kam er an die höchsten Blätter.',
      },
      {
        frage: 'Was bedeutet sein Name?',
        optionen: ['Arm-Echse', 'Hals-Echse', 'Donner-Echse', 'Berg-Echse'],
        richtig: 0,
        tipp: 'Es hat mit seinen Vorderbeinen zu tun.',
      },
      {
        frage: 'Welche Beine waren länger?',
        optionen: ['Die hinteren', 'Die vorderen', 'Alle gleich', 'Er hatte keine'],
        richtig: 1,
        tipp: 'Deshalb ging der Rücken nach hinten abwärts.',
      },
      {
        frage: 'Wie schwer war er ungefähr?',
        optionen: ['So schwer wie ein Auto', 'So schwer wie acht Elefanten', 'So schwer wie ein Hund', 'So schwer wie ein Vogel'],
        richtig: 1,
        tipp: 'Im letzten Absatz steht die Zahl.',
      },
    ],
  },

  // 5 -----------------------------------------------------------------
  {
    id: 'stegosaurus',
    kategorie: 'dino',
    titel: 'Der Stegosaurus',
    absaetze: [
      'Den Stegosaurus erkennt man sofort. Auf seinem Rücken standen große Platten. Sie waren aus Knochen und sehr breit. Die größten Platten waren so groß wie ein Sitzkissen. Insgesamt trug er etwa 17 davon.',
      'Wozu die Platten gut waren, ist spannend. Vielleicht dienten sie als Sonnenschirm. Vielleicht wurden sie auch rot, wenn er wütend war. Viele Forscher glauben: Er konnte damit warm und kalt werden. Wie ein Heizkörper an der Wand.',
      'Am Schwanz hatte er vier lange Stacheln. Jeder Stachel war fast einen Meter lang. Damit konnte er kräftig zuschlagen. Ein Feind bekam so einen ordentlichen Schreck. Forscher nennen diese Waffe den "Thagomizer".',
      'Der Stegosaurus war 9 Meter lang. Sein Kopf war dabei erstaunlich klein. Sein Gehirn war nur so groß wie eine Walnuss. Trotzdem hat er gut gelebt. Er brauchte keine schwierigen Aufgaben zu lösen.',
      'Er fraß Farne, Moose und niedrige Pflanzen. Seinen Kopf hielt er dicht über dem Boden. Aufrecht stehen konnte er nicht gut. Deshalb blieb er immer schön unten. Er lebte vor etwa 150 Millionen Jahren.',
    ],
    fragen: [
      {
        frage: 'Was stand auf dem Rücken des Stegosaurus?',
        optionen: ['Federn', 'Platten', 'Haare', 'Hörner'],
        richtig: 1,
        tipp: 'Sie waren aus Knochen.',
      },
      {
        frage: 'Wie viele Platten trug er ungefähr?',
        optionen: ['3', '7', '17', '70'],
        richtig: 2,
        tipp: 'Die Zahl steht im ersten Absatz.',
      },
      {
        frage: 'Was hatte er am Schwanz?',
        optionen: ['Vier lange Stacheln', 'Eine Keule', 'Eine Flosse', 'Nichts'],
        richtig: 0,
        tipp: 'Jeder war fast einen Meter lang.',
      },
      {
        frage: 'Wie groß war sein Gehirn?',
        optionen: ['Wie ein Fußball', 'Wie eine Walnuss', 'Wie ein Kürbis', 'Wie ein Auto'],
        richtig: 1,
        tipp: 'Es war erstaunlich klein.',
      },
      {
        frage: 'Was hat der Stegosaurus gefressen?',
        optionen: ['Fische', 'Andere Dinos', 'Niedrige Pflanzen', 'Steine'],
        richtig: 2,
        tipp: 'Er hielt den Kopf dicht über dem Boden.',
      },
    ],
  },

  // 6 -----------------------------------------------------------------
  {
    id: 'igel',
    kategorie: 'natur',
    titel: 'Der Igel',
    absaetze: [
      'Der Igel ist ein kleines Säugetier. Auf seinem Rücken trägt er viele Stacheln. Ein erwachsener Igel hat etwa 8000 davon. Die Stacheln sind hohl und leicht. Sie sind aus dem gleichen Stoff wie deine Fingernägel.',
      'Wenn ein Igel Angst hat, rollt er sich ein. Dann wird er zu einer Stachelkugel. Kein Fuchs mag hinein beißen. So schützt sich der Igel ganz ohne Kampf. Junge Igel lernen das schon früh.',
      'Igel sind nachts unterwegs. Am Tag schlafen sie in einem Versteck. Nachts suchen sie Käfer, Würmer und Schnecken. Sie hören und riechen sehr gut. Sehen können sie dagegen eher schlecht.',
      'Im Winter hält der Igel Winterschlaf. Dafür baut er ein Nest aus Laub. Sein Herz schlägt dann viel langsamer. Er atmet nur noch ganz wenig. So spart er Kraft, bis es wieder warm wird.',
      'Du kannst Igeln im Garten helfen. Lass einen Haufen Laub liegen. Stelle eine flache Schale mit Wasser hin. Milch ist übrigens nicht gut für Igel. Davon bekommen sie Bauchweh.',
    ],
    fragen: [
      {
        frage: 'Wie viele Stacheln hat ein erwachsener Igel ungefähr?',
        optionen: ['80', '800', '8000', '80000'],
        richtig: 2,
        tipp: 'Die Zahl steht im ersten Absatz.',
      },
      {
        frage: 'Was macht der Igel bei Gefahr?',
        optionen: ['Er rennt weg', 'Er rollt sich ein', 'Er klettert', 'Er ruft laut'],
        richtig: 1,
        tipp: 'Er wird zu einer Kugel.',
      },
      {
        frage: 'Wann ist der Igel unterwegs?',
        optionen: ['Nachts', 'Am Morgen', 'Am Mittag', 'Nie'],
        richtig: 0,
        tipp: 'Am Tag schläft er.',
      },
      {
        frage: 'Was macht der Igel im Winter?',
        optionen: ['Er zieht weg', 'Er hält Winterschlaf', 'Er sammelt Nüsse', 'Er badet'],
        richtig: 1,
        tipp: 'Er baut ein Nest aus Laub.',
      },
      {
        frage: 'Was solltest du einem Igel nicht geben?',
        optionen: ['Wasser', 'Laub', 'Milch', 'Ein Versteck'],
        richtig: 2,
        tipp: 'Davon bekommt er Bauchweh.',
      },
    ],
  },

  // 7 -----------------------------------------------------------------
  {
    id: 'velociraptor',
    kategorie: 'dino',
    titel: 'Der Velociraptor',
    absaetze: [
      'Der Velociraptor war ein kleiner, schneller Jäger. Sein Name bedeutet "schneller Räuber". In Filmen wird er oft riesig gezeigt. In Wirklichkeit war er aber viel kleiner. Er war ungefähr so groß wie ein Truthahn.',
      'Am Kopf und an den Armen hatte er Federn. Fliegen konnte er damit nicht. Die Federn hielten ihn wahrscheinlich warm. Vielleicht sahen sie auch schön bunt aus. Vögel sind heute seine nächsten Verwandten.',
      'An jedem Hinterfuß hatte er eine große Kralle. Sie war sichelförmig gebogen. Beim Laufen hielt er sie hoch. So blieb sie immer scharf. Damit hielt er seine Beute fest.',
      'Der Velociraptor war sehr schnell. Er konnte etwa 40 Kilometer pro Stunde rennen. Das ist schneller als ein Fahrrad in der Stadt. Sein langer Schwanz war dabei sehr wichtig. Er hielt ihn beim Rennen im Gleichgewicht.',
      'Gelebt hat er in der Wüste Asiens. Dort ist ein besonderes Fossil gefunden worden. Ein Velociraptor und ein Protoceratops liegen darin zusammen. Sie sind mitten im Kampf verschüttet worden. Man nennt sie die "kämpfenden Dinosaurier".',
    ],
    fragen: [
      {
        frage: 'Wie groß war ein Velociraptor wirklich?',
        optionen: ['Wie ein Truthahn', 'Wie ein Elefant', 'Wie ein Bus', 'Wie ein Haus'],
        richtig: 0,
        tipp: 'In Filmen wird er größer gezeigt als er war.',
      },
      {
        frage: 'Was hatte er an Kopf und Armen?',
        optionen: ['Schuppen', 'Federn', 'Stacheln', 'Haare'],
        richtig: 1,
        tipp: 'Vögel sind heute seine Verwandten.',
      },
      {
        frage: 'Wo saß seine große Sichelkralle?',
        optionen: ['Am Kopf', 'Am Schwanz', 'Am Hinterfuß', 'Am Rücken'],
        richtig: 2,
        tipp: 'Beim Laufen hielt er sie hoch.',
      },
      {
        frage: 'Wofür brauchte er seinen langen Schwanz?',
        optionen: ['Zum Schlagen', 'Zum Gleichgewicht', 'Zum Graben', 'Zum Schwimmen'],
        richtig: 1,
        tipp: 'Er half ihm beim schnellen Rennen.',
      },
      {
        frage: 'Wie nennt man das berühmte Fossil aus Asien?',
        optionen: ['Die schlafenden Dinos', 'Die kämpfenden Dinosaurier', 'Die großen Zwei', 'Die Wüstenechsen'],
        richtig: 1,
        tipp: 'Zwei Dinos wurden mitten im Kampf verschüttet.',
      },
    ],
  },

  // 8 -----------------------------------------------------------------
  {
    id: 'flugsaurier',
    kategorie: 'dino',
    titel: 'Die Flugsaurier',
    absaetze: [
      'Flugsaurier waren keine echten Dinosaurier. Sie waren aber ihre nahen Verwandten. Sie lebten zur gleichen Zeit. Und sie konnten etwas Besonderes: richtig fliegen. Damit waren sie die ersten fliegenden Wirbeltiere.',
      'Ihre Flügel sahen anders aus als bei Vögeln. Ein Finger war extrem verlängert. Zwischen diesem Finger und dem Körper spannte sich eine Haut. Diese Flughaut war dünn, aber sehr fest. Sie funktionierte wie ein Segel.',
      'Der Pteranodon hatte eine Spannweite von 7 Metern. Der Quetzalcoatlus war noch größer. Seine Flügel maßen bis zu 11 Meter. Er war so groß wie ein kleines Flugzeug. Am Boden stand er so hoch wie eine Giraffe.',
      'Viele Flugsaurier lebten am Meer. Sie flogen dicht über das Wasser. Dabei fingen sie Fische mit dem Schnabel. Manche hatten einen großen Kamm am Kopf. Damit steuerten sie vielleicht in der Luft.',
      'Nicht alle Flugsaurier waren riesig. Manche waren nur so groß wie ein Spatz. Sie fingen Insekten in der Luft. Vor 66 Millionen Jahren starben sie alle aus. Zur gleichen Zeit verschwanden auch die großen Dinosaurier.',
    ],
    fragen: [
      {
        frage: 'Waren Flugsaurier echte Dinosaurier?',
        optionen: ['Ja, alle', 'Nein, nur Verwandte', 'Ja, die kleinen', 'Sie waren Vögel'],
        richtig: 1,
        tipp: 'Der erste Absatz sagt es genau.',
      },
      {
        frage: 'Woraus bestand ihr Flügel vor allem?',
        optionen: ['Aus Federn', 'Aus Haaren', 'Aus einer Flughaut', 'Aus Knochenplatten'],
        richtig: 2,
        tipp: 'Sie war dünn und funktionierte wie ein Segel.',
      },
      {
        frage: 'Wie groß war die Spannweite des Quetzalcoatlus?',
        optionen: ['1 Meter', '3 Meter', '11 Meter', '50 Meter'],
        richtig: 2,
        tipp: 'Er war so groß wie ein kleines Flugzeug.',
      },
      {
        frage: 'Was fingen viele Flugsaurier über dem Meer?',
        optionen: ['Fische', 'Schnecken', 'Blätter', 'Steine'],
        richtig: 0,
        tipp: 'Sie flogen dicht über dem Wasser.',
      },
      {
        frage: 'Wann starben die Flugsaurier aus?',
        optionen: ['Vor 100 Jahren', 'Vor 66 Millionen Jahren', 'Vor 5 Jahren', 'Sie leben noch'],
        richtig: 1,
        tipp: 'Zur gleichen Zeit wie die großen Dinos.',
      },
    ],
  },

  // 9 -----------------------------------------------------------------
  {
    id: 'wald',
    kategorie: 'natur',
    titel: 'Der Wald',
    absaetze: [
      'Ein Wald ist wie ein großes Haus. Er hat mehrere Stockwerke. Ganz oben sind die Kronen der Bäume. Darunter wachsen kleinere Sträucher. Am Boden liegen Moos, Pilze und altes Laub.',
      'In jedem Stockwerk wohnen andere Tiere. Oben bauen Vögel ihre Nester. In der Mitte springen Eichhörnchen umher. Am Boden laufen Käfer, Mäuse und Igel. Sogar unter der Erde leben Regenwürmer.',
      'Bäume machen etwas Erstaunliches. Sie nehmen Licht, Wasser und Luft auf. Daraus bauen sie sich ihre Nahrung. Dabei geben sie Sauerstoff ab. Diesen Sauerstoff atmen wir Menschen ein.',
      'Der Waldboden ist ein fleißiger Helfer. Altes Laub wird dort zersetzt. Winzige Lebewesen machen daraus neue Erde. So bekommen die Bäume wieder Nahrung. Nichts geht im Wald verloren.',
      'Ein Wald speichert außerdem viel Wasser. Nach dem Regen hält der Boden es fest. Deshalb ist es im Wald oft kühler. Wenn du leise bist, siehst du viele Tiere. Am besten gehst du früh am Morgen los.',
    ],
    fragen: [
      {
        frage: 'Was ist ganz oben im Wald?',
        optionen: ['Das Moos', 'Die Baumkronen', 'Die Sträucher', 'Die Wurzeln'],
        richtig: 1,
        tipp: 'Der Wald hat Stockwerke wie ein Haus.',
      },
      {
        frage: 'Welches Tier springt in der mittleren Etage umher?',
        optionen: ['Das Eichhörnchen', 'Der Regenwurm', 'Der Igel', 'Der Fisch'],
        richtig: 0,
        tipp: 'Es lebt zwischen den Ästen.',
      },
      {
        frage: 'Was geben Bäume an die Luft ab?',
        optionen: ['Regen', 'Sauerstoff', 'Erde', 'Laub'],
        richtig: 1,
        tipp: 'Das atmen wir Menschen ein.',
      },
      {
        frage: 'Was passiert mit altem Laub am Boden?',
        optionen: ['Es wird zu neuer Erde', 'Es verschwindet einfach', 'Es wird zu Wasser', 'Es wird zu Stein'],
        richtig: 0,
        tipp: 'Winzige Lebewesen helfen dabei.',
      },
      {
        frage: 'Warum ist es im Wald oft kühler?',
        optionen: ['Weil dort Schnee liegt', 'Weil der Boden Wasser speichert', 'Weil die Sonne fehlt', 'Weil Tiere pusten'],
        richtig: 1,
        tipp: 'Nach dem Regen hält der Boden es fest.',
      },
    ],
  },

  // 10 ----------------------------------------------------------------
  {
    id: 'spinosaurus',
    kategorie: 'dino',
    titel: 'Der Spinosaurus',
    absaetze: [
      'Der Spinosaurus war noch länger als der T-Rex. Er wurde bis zu 15 Meter lang. Auf seinem Rücken trug er ein großes Segel. Das Segel wurde aus langen Knochen gebildet. Manche davon waren fast zwei Meter hoch.',
      'Wozu das Segel diente, wissen wir nicht genau. Vielleicht half es beim Wärmen. Vielleicht sah der Spinosaurus damit größer aus. Andere Dinos bekamen dann vielleicht Respekt. Möglich ist auch, dass es zum Angeben diente.',
      'Sein Kopf sah aus wie der eines Krokodils. Er war lang und schmal. Seine Zähne waren rund und spitz. Solche Zähne sind super, um Fische festzuhalten. Denn Fisch war seine Lieblingsspeise.',
      'Der Spinosaurus lebte viel im Wasser. Sein Schwanz war hoch und flach. Damit konnte er wie ein Fisch paddeln. Seine Nasenlöcher saßen weit oben am Kopf. So konnte er atmen und trotzdem lauern.',
      'Gefunden wurde er in der Sahara. Heute ist das eine trockene Wüste. Früher gab es dort breite Flüsse. Riesige Fische schwammen darin. Für den Spinosaurus war das ein Paradies.',
    ],
    fragen: [
      {
        frage: 'Wie lang wurde der Spinosaurus?',
        optionen: ['5 Meter', '15 Meter', '40 Meter', '80 Meter'],
        richtig: 1,
        tipp: 'Er war länger als der T-Rex.',
      },
      {
        frage: 'Was trug er auf dem Rücken?',
        optionen: ['Ein Segel', 'Einen Panzer', 'Federn', 'Einen Höcker'],
        richtig: 0,
        tipp: 'Es wurde aus langen Knochen gebildet.',
      },
      {
        frage: 'Wem ähnelte sein Kopf?',
        optionen: ['Einem Hund', 'Einem Krokodil', 'Einem Vogel', 'Einer Kuh'],
        richtig: 1,
        tipp: 'Lang und schmal.',
      },
      {
        frage: 'Was fraß er am liebsten?',
        optionen: ['Blätter', 'Käfer', 'Fisch', 'Eier'],
        richtig: 2,
        tipp: 'Seine runden Zähne passten gut dazu.',
      },
      {
        frage: 'Wo wurde der Spinosaurus gefunden?',
        optionen: ['In der Sahara', 'In Amerika', 'In China', 'Am Nordpol'],
        richtig: 0,
        tipp: 'Heute ist das eine trockene Wüste.',
      },
    ],
  },

  // 11 ----------------------------------------------------------------
  {
    id: 'ankylosaurus',
    kategorie: 'dino',
    titel: 'Der Ankylosaurus',
    absaetze: [
      'Der Ankylosaurus war der Panzer unter den Dinos. Sein ganzer Rücken war mit Knochenplatten bedeckt. Zwischen den Platten saßen dicke Stacheln. Sogar seine Augenlider waren verstärkt. Fast nichts konnte ihm etwas anhaben.',
      'Am Ende seines Schwanzes saß eine Keule. Sie bestand aus festem Knochen. Sie war so schwer wie ein großer Stein. Mit einem Schlag konnte er einem Angreifer die Beine treffen. Sogar ein T-Rex hat sich das gut überlegt.',
      'Er war ungefähr 8 Meter lang und sehr breit. Sein Bauch war die einzige weiche Stelle. Deshalb blieb er immer dicht am Boden. Umdrehen ließ er sich nicht gerne. Wer ihn ärgerte, bekam die Keule zu spüren.',
      'Gefressen hat der Ankylosaurus nur Pflanzen. Seine Zähne waren klein und blattförmig. Er fraß Farne und weiche Blätter dicht über dem Boden. Sein Bauch war sehr groß. Darin wurde das Futter langsam verdaut.',
      'Sein Name bedeutet "verwachsene Echse". Das kommt von den zusammengewachsenen Knochen. Er lebte am Ende der Dino-Zeit. Vor 66 Millionen Jahren verschwand auch er. Seine Panzerplatten findet man heute noch als Fossil.',
    ],
    fragen: [
      {
        frage: 'Womit war der Rücken bedeckt?',
        optionen: ['Mit Federn', 'Mit Knochenplatten', 'Mit Fell', 'Mit Blättern'],
        richtig: 1,
        tipp: 'Dazwischen saßen dicke Stacheln.',
      },
      {
        frage: 'Was hatte er am Schwanzende?',
        optionen: ['Eine Keule', 'Eine Flosse', 'Ein Horn', 'Ein Segel'],
        richtig: 0,
        tipp: 'Sie war so schwer wie ein großer Stein.',
      },
      {
        frage: 'Welche Stelle war bei ihm weich?',
        optionen: ['Der Kopf', 'Der Rücken', 'Der Bauch', 'Der Schwanz'],
        richtig: 2,
        tipp: 'Deshalb blieb er dicht am Boden.',
      },
      {
        frage: 'Was hat er gefressen?',
        optionen: ['Nur Pflanzen', 'Fische', 'Andere Dinos', 'Insekten'],
        richtig: 0,
        tipp: 'Seine Zähne waren klein und blattförmig.',
      },
      {
        frage: 'Was bedeutet sein Name?',
        optionen: ['Panzer-Echse', 'Verwachsene Echse', 'Keulen-Echse', 'Stachel-Echse'],
        richtig: 1,
        tipp: 'Es kommt von den zusammengewachsenen Knochen.',
      },
    ],
  },

  // 12 ----------------------------------------------------------------
  {
    id: 'wale',
    kategorie: 'natur',
    titel: 'Die Wale',
    absaetze: [
      'Wale sehen aus wie riesige Fische. In Wirklichkeit sind sie aber Säugetiere. Sie atmen Luft wie du. Dafür müssen sie an die Oberfläche kommen. Oben blasen sie eine Fontäne aus dem Loch am Kopf.',
      'Der Blauwal ist das größte Tier der Erde. Er wird bis zu 30 Meter lang. Sein Herz ist so groß wie ein kleines Auto. Sogar ein Dinosaurier war nicht schwerer. Trotzdem frisst er nur winzige Krebse.',
      'Diese kleinen Krebse heißen Krill. Der Blauwal nimmt riesige Schlucke Wasser. Dann presst er das Wasser wieder heraus. Der Krill bleibt in seinen Barten hängen. Barten sind wie große Kämme im Maul.',
      'Wal-Babys werden lebend geboren. Sie trinken Milch bei ihrer Mutter. Ein Blauwal-Baby wächst sehr schnell. Am Tag nimmt es bis zu 90 Kilogramm zu. Nach einem Jahr ist es schon riesig.',
      'Wale können sich weit hören. Ihre Rufe reisen viele Kilometer durch das Wasser. Buckelwale singen sogar richtige Lieder. Ein Lied kann eine halbe Stunde dauern. Forscher hören ihnen mit Mikrofonen zu.',
    ],
    fragen: [
      {
        frage: 'Zu welcher Tiergruppe gehören Wale?',
        optionen: ['Zu den Fischen', 'Zu den Säugetieren', 'Zu den Vögeln', 'Zu den Reptilien'],
        richtig: 1,
        tipp: 'Sie atmen Luft und trinken Milch.',
      },
      {
        frage: 'Wie lang wird ein Blauwal?',
        optionen: ['3 Meter', '10 Meter', '30 Meter', '100 Meter'],
        richtig: 2,
        tipp: 'Er ist das größte Tier der Erde.',
      },
      {
        frage: 'Wie heißen die kleinen Krebse, die er frisst?',
        optionen: ['Krill', 'Plankton-Fische', 'Muscheln', 'Quallen'],
        richtig: 0,
        tipp: 'Sie bleiben in den Barten hängen.',
      },
      {
        frage: 'Was trinken Wal-Babys?',
        optionen: ['Wasser', 'Saft', 'Milch', 'Nichts'],
        richtig: 2,
        tipp: 'Wie alle Säugetiere.',
      },
      {
        frage: 'Welche Wale singen richtige Lieder?',
        optionen: ['Buckelwale', 'Blauwale', 'Orcas', 'Delfine'],
        richtig: 0,
        tipp: 'Ein Lied kann eine halbe Stunde dauern.',
      },
    ],
  },

  // 13 ----------------------------------------------------------------
  {
    id: 'dinoeier',
    kategorie: 'dino',
    titel: 'Dino-Eier und Nester',
    absaetze: [
      'Alle Dinosaurier sind aus Eiern geschlüpft. Ihre Eier hatten eine harte Schale. Sie sahen ähnlich aus wie Vogeleier. Manche waren rund, manche länglich. Das größte gefundene Ei ist 60 Zentimeter lang.',
      'Überraschend ist: Die Eier waren nie riesig. Auch bei den größten Dinos nicht. Denn eine zu dicke Schale lässt keine Luft mehr durch. Das Baby im Inneren würde ersticken. Darum blieben die Eier eher klein.',
      'Viele Dinos bauten richtige Nester. Sie scharrten eine Mulde in den Sand. Dort hinein legten sie ihre Eier im Kreis. Manchmal deckten sie das Nest mit Pflanzen zu. Die Pflanzen wärmten wie eine Decke.',
      'Ein Dino heißt sogar "gute Mutter-Echse". Auf Latein: Maiasaura. Bei ihren Nestern fand man Knochen von Jungtieren. Sie waren schon größer, blieben aber im Nest. Das zeigt: Die Eltern brachten ihnen Futter.',
      'Manche Nester lagen dicht beieinander. Viele Tiere brüteten also an einem Platz. So wie Möwen heute an einer Küste. Gemeinsam konnten sie ihre Jungen besser schützen. Solche Nestplätze findet man in der Mongolei und in Amerika.',
    ],
    fragen: [
      {
        frage: 'Wie kamen junge Dinosaurier auf die Welt?',
        optionen: ['Sie wurden geboren', 'Sie schlüpften aus Eiern', 'Sie wuchsen aus Pflanzen', 'Das weiß niemand'],
        richtig: 1,
        tipp: 'Die Eier hatten eine harte Schale.',
      },
      {
        frage: 'Wie lang ist das größte gefundene Ei?',
        optionen: ['6 Zentimeter', '60 Zentimeter', '2 Meter', '6 Meter'],
        richtig: 1,
        tipp: 'Die Zahl steht im ersten Absatz.',
      },
      {
        frage: 'Warum waren Dino-Eier nicht riesig?',
        optionen: ['Sie wären zu schwer', 'Die Schale ließe keine Luft durch', 'Sie wären zu teuer', 'Sie wären zu kalt'],
        richtig: 1,
        tipp: 'Das Baby braucht Luft durch die Schale.',
      },
      {
        frage: 'Was bedeutet der Name Maiasaura?',
        optionen: ['Gute Mutter-Echse', 'Große Echse', 'Nest-Echse', 'Schnelle Echse'],
        richtig: 0,
        tipp: 'Bei ihren Nestern fand man Jungtiere.',
      },
      {
        frage: 'Warum brüteten viele Dinos nah beieinander?',
        optionen: ['Weil es warm war', 'Um die Jungen besser zu schützen', 'Weil es nur wenig Platz gab', 'Weil sie faul waren'],
        richtig: 1,
        tipp: 'So wie Möwen heute an der Küste.',
      },
    ],
  },

  // 14 ----------------------------------------------------------------
  {
    id: 'zugvoegel',
    kategorie: 'natur',
    titel: 'Die Zugvögel',
    absaetze: [
      'Im Herbst wird es bei uns kalt. Dann verschwinden viele Vögel. Sie fliegen in warme Länder. Solche Vögel nennt man Zugvögel. Im Frühling kommen sie wieder zurück.',
      'Sie fliegen weg, weil das Futter fehlt. Im Winter gibt es kaum Insekten. Ohne Futter könnten sie nicht überleben. Im Süden finden sie genug zu fressen. Der weite Weg lohnt sich also.',
      'Störche fliegen bis nach Afrika. Das sind ungefähr 10000 Kilometer. Sie segeln dabei in der warmen Luft. So sparen sie viel Kraft. Trotzdem dauert die Reise mehrere Wochen.',
      'Wie finden Vögel den Weg? Sie merken sich Flüsse und Berge. Sie nutzen den Stand der Sonne. Und sie spüren das Magnetfeld der Erde. Das ist wie ein eingebauter Kompass im Kopf.',
      'Nicht alle Vögel ziehen weg. Meisen, Spatzen und Amseln bleiben hier. Sie heißen Standvögel. Im Winter kannst du ihnen helfen. Ein Futterhaus mit Körnern ist eine gute Idee.',
    ],
    fragen: [
      {
        frage: 'Wann fliegen Zugvögel weg?',
        optionen: ['Im Frühling', 'Im Sommer', 'Im Herbst', 'Nie'],
        richtig: 2,
        tipp: 'Dann wird es bei uns kalt.',
      },
      {
        frage: 'Warum fliegen sie in warme Länder?',
        optionen: ['Weil das Futter fehlt', 'Weil es dort schöner ist', 'Weil sie Urlaub machen', 'Weil sie sich verirren'],
        richtig: 0,
        tipp: 'Im Winter gibt es kaum Insekten.',
      },
      {
        frage: 'Wie weit fliegen Störche ungefähr?',
        optionen: ['100 Kilometer', '1000 Kilometer', '10000 Kilometer', '10 Kilometer'],
        richtig: 2,
        tipp: 'Sie fliegen bis nach Afrika.',
      },
      {
        frage: 'Was hilft Vögeln beim Finden des Weges?',
        optionen: ['Eine Landkarte', 'Das Magnetfeld der Erde', 'Ein Handy', 'Der Wind allein'],
        richtig: 1,
        tipp: 'Wie ein Kompass im Kopf.',
      },
      {
        frage: 'Wie heißen Vögel, die hier bleiben?',
        optionen: ['Standvögel', 'Wintervögel', 'Hausvögel', 'Schnellvögel'],
        richtig: 0,
        tipp: 'Meisen und Amseln gehören dazu.',
      },
    ],
  },

  // 15 ----------------------------------------------------------------
  {
    id: 'fossil',
    kategorie: 'dino',
    titel: 'Wie ein Fossil entsteht',
    absaetze: [
      'Ein Fossil ist ein Rest aus alter Zeit. Meistens sind es Knochen oder Zähne. Manchmal sind es auch Fußspuren. Sogar ein Ei kann ein Fossil sein. Fossilien sind viele Millionen Jahre alt.',
      'Am Anfang stirbt ein Tier. Es muss schnell zugedeckt werden. Am besten mit Sand oder Schlamm. Sonst fressen andere Tiere die Knochen auf. Nur zugedeckte Knochen bleiben erhalten.',
      'Über den Knochen sammelt sich mehr und mehr Sand. Schicht für Schicht drückt darauf. Aus dem Sand wird nach langer Zeit Stein. Wasser sickert dabei durch die Knochen. Es bringt winzige Mineralien mit.',
      'Diese Mineralien setzen sich in den Knochen fest. Nach und nach wird der Knochen selbst zu Stein. Seine Form bleibt aber genau erhalten. So entsteht ein steinernes Abbild. Genau das nennen wir Fossil.',
      'Später hebt sich vielleicht das Land. Wind und Regen tragen den Stein wieder ab. Dann liegt das Fossil an der Oberfläche. Forscher pinseln es vorsichtig frei. Im Museum wird das Skelett dann zusammengesetzt.',
    ],
    fragen: [
      {
        frage: 'Was ist ein Fossil meistens?',
        optionen: ['Ein Blatt', 'Ein Knochen oder Zahn', 'Ein Stück Holz', 'Ein Stück Eis'],
        richtig: 1,
        tipp: 'Auch Fußspuren können Fossilien sein.',
      },
      {
        frage: 'Was muss zuerst passieren?',
        optionen: ['Das Tier muss schnell zugedeckt werden', 'Es muss regnen', 'Es muss frieren', 'Ein Forscher muss kommen'],
        richtig: 0,
        tipp: 'Am besten mit Sand oder Schlamm.',
      },
      {
        frage: 'Was bringt das Wasser in die Knochen?',
        optionen: ['Luft', 'Mineralien', 'Sand', 'Farbe'],
        richtig: 1,
        tipp: 'Sie setzen sich im Knochen fest.',
      },
      {
        frage: 'Was passiert mit dem Knochen nach langer Zeit?',
        optionen: ['Er wird zu Stein', 'Er wird zu Holz', 'Er verschwindet ganz', 'Er wird weich'],
        richtig: 0,
        tipp: 'Die Form bleibt dabei erhalten.',
      },
      {
        frage: 'Wie machen Forscher ein Fossil frei?',
        optionen: ['Mit dem Bagger', 'Mit Wasser', 'Vorsichtig mit dem Pinsel', 'Mit Feuer'],
        richtig: 2,
        tipp: 'Es steht im letzten Absatz.',
      },
    ],
  },
];

/** Welcher Lesetext gehoert zu welchem (ungeraden) Level? */
export function lesetextFuerLevel(levelNr) {
  const index = Math.floor((levelNr - 1) / 2) % LESETEXTE.length;
  return LESETEXTE[index];
}

export const ANZAHL_LESETEXTE = LESETEXTE.length;
