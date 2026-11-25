"""Mini-catalogue de démonstration (en attendant data/processed/movies.parquet de la partie 1).

Mêmes colonnes que le contrat : movie_id, title, year, tmdb_id, redial_id, genres, tags,
overview, poster_url, popularity_bucket (+ une colonne optionnelle `aliases` pour les
titres français). Les identifiants sont FICTIFS : ne pas les mélanger avec MovieLens.
Les résumés sont écrits à la main, très courts.
"""
from __future__ import annotations

import pandas as pd

# (title, year, genres "|", tags ";", popularity_bucket, aliases ";", overview)
_FILMS = [
    # ---- Action grand public
    ("Die Hard", 1988, "Action|Thriller", "hostage;skyscraper;christmas;one-man-army", "head", "Piège de cristal", "Un flic new-yorkais seul face à des braqueurs qui prennent une tour en otage le soir de Noël."),
    ("Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi", "post-apocalyptic;car chase;desert;practical effects", "head", "Mad Max Fury Road", "Une course-poursuite sans fin dans un désert post-apocalyptique pour libérer des captives d'un tyran."),
    ("John Wick", 2014, "Action|Thriller|Crime", "assassin;revenge;gun-fu;stylized", "head", "", "Un tueur à gages retraité reprend du service pour venger son chien."),
    ("The Dark Knight", 2008, "Action|Crime|Drama", "superhero;batman;joker;villain", "head", "Le Chevalier noir", "Batman affronte le Joker, un criminel qui veut faire sombrer Gotham dans le chaos."),
    ("Mission: Impossible - Fallout", 2018, "Action|Adventure|Thriller", "spy;stunts;helicopter;tom cruise", "head", "Mission Impossible Fallout", "Ethan Hunt court contre la montre pour récupérer du plutonium volé."),
    ("Terminator 2: Judgment Day", 1991, "Action|Sci-Fi", "robot;time travel;cyborg;apocalypse", "head", "Terminator 2;Le Jugement dernier", "Un cyborg venu du futur protège un adolescent destiné à sauver l'humanité."),
    ("Gladiator", 2000, "Action|Adventure|Drama", "ancient rome;revenge;arena;epic", "head", "", "Un général romain trahi devient gladiateur et cherche à se venger de l'empereur."),
    ("The Matrix", 1999, "Action|Sci-Fi", "simulation;cyberpunk;kung fu;chosen one", "head", "Matrix", "Un hacker découvre que le monde est une simulation contrôlée par des machines."),
    ("Top Gun: Maverick", 2022, "Action|Drama", "fighter jets;aviation;mentor;sequel", "head", "Top Gun Maverick", "Un pilote vétéran forme une jeune escadrille pour une mission quasi impossible."),
    ("Avengers: Endgame", 2019, "Action|Adventure|Sci-Fi", "superhero;marvel;time travel;ensemble", "head", "Avengers Endgame", "Les héros survivants tentent d'annuler la catastrophe provoquée par Thanos."),
    ("Inception", 2010, "Action|Sci-Fi|Thriller", "dreams;heist;mind-bending;nolan", "head", "", "Une équipe de voleurs s'infiltre dans les rêves pour implanter une idée."),
    ("Edge of Tomorrow", 2014, "Action|Sci-Fi", "time loop;aliens;war", "mid", "Edge of Tomorrow;Live Die Repeat", "Un soldat revit en boucle la même bataille contre des envahisseurs extraterrestres."),
    ("Speed", 1994, "Action|Thriller", "bus;bomb;hostage", "mid", "", "Un bus piégé explose s'il descend sous 80 km/h."),
    ("Baby Driver", 2017, "Action|Crime|Musical", "getaway driver;music;heist;car chase", "mid", "", "Un jeune chauffeur de braqueurs règle chaque poursuite sur sa playlist."),
    ("Heat", 1995, "Action|Crime|Drama", "heist;cops and robbers;los angeles;shootout", "mid", "", "Un flic obsessionnel traque un braqueur méthodique dans Los Angeles."),
    ("Collateral", 2004, "Action|Crime|Thriller", "hitman;taxi;night;los angeles", "mid", "", "Un chauffeur de taxi est forcé de conduire un tueur à gages toute une nuit."),
    ("Point Break", 1991, "Action|Crime", "surfing;fbi;undercover;bank robbery", "mid", "", "Un agent du FBI infiltre une bande de surfeurs braqueurs de banques."),
    ("Ronin", 1998, "Action|Crime|Thriller", "mercenaries;car chase;paris;spy", "mid", "", "Des mercenaires engagés en France pour voler une mallette mystérieuse."),
    ("Drive", 2011, "Action|Crime|Drama", "getaway driver;neo-noir;synthwave;stylized", "mid", "", "Un cascadeur taciturne, chauffeur pour truands la nuit, se retrouve pris dans un engrenage violent."),
    ("Sicario", 2015, "Action|Crime|Thriller", "cartel;border;fbi;tense", "mid", "", "Une agente du FBI rejoint une opération trouble contre un cartel mexicain."),
    ("Oldboy", 2003, "Action|Drama|Mystery|Thriller", "revenge;korean;twist;corridor fight", "mid", "Old Boy", "Séquestré quinze ans sans explication, un homme est libéré et cherche pourquoi."),
    # ---- Action pépites
    ("The Raid", 2011, "Action|Crime|Thriller", "martial arts;indonesian;tower;silat", "tail", "The Raid: Redemption", "Une unité d'élite piégée dans un immeuble contrôlé par un baron de la drogue."),
    ("Hard Boiled", 1992, "Action|Crime|Thriller", "hong kong;john woo;heroic bloodshed;hospital", "tail", "À toute épreuve", "Un flic de Hong Kong et un infiltré affrontent des trafiquants d'armes jusque dans un hôpital."),
    ("Dredd", 2012, "Action|Sci-Fi", "dystopia;judge;tower;comic", "tail", "", "Un juge du futur et une recrue sont pris au piège dans une tour tenue par un gang."),
    ("The Night Comes for Us", 2018, "Action|Thriller", "martial arts;indonesian;triad;brutal", "tail", "", "Un tueur des triades épargne une enfant et devient la cible de son ancien clan."),
    ("Sonatine", 1993, "Action|Crime|Drama", "yakuza;kitano;beach;melancholic", "tail", "", "Un yakuza fatigué envoyé à Okinawa attend la guerre des gangs sur une plage."),
    ("Le Samouraï", 1967, "Crime|Drama|Thriller", "hitman;french;melville;minimalist", "tail", "Le Samourai", "Un tueur à gages solitaire et méthodique est cerné après un contrat."),
    ("Blue Ruin", 2013, "Action|Crime|Thriller", "revenge;low budget;indie;backwoods", "tail", "", "Un vagabond maladroit se lance dans une vengeance qui le dépasse."),
    ("Thief", 1981, "Action|Crime|Drama", "safecracker;heist;michael mann;neo-noir", "tail", "Le Solitaire", "Un perceur de coffres veut faire un dernier coup pour se ranger."),
    ("The Villainess", 2017, "Action|Thriller", "assassin;korean;revenge;one-take", "tail", "", "Une tueuse formée depuis l'enfance cherche à se libérer de ceux qui l'ont créée."),
    ("Headhunters", 2011, "Action|Crime|Thriller", "norwegian;art theft;chase;dark comedy", "tail", "", "Un chasseur de têtes qui vole des tableaux s'attaque à la mauvaise cible."),
    ("Ip Man", 2008, "Action|Drama", "martial arts;wing chun;biopic;hong kong", "mid", "", "Un maître de wing chun résiste à l'occupation japonaise à Foshan."),
    ("Train to Busan", 2016, "Action|Horror|Thriller", "zombies;korean;train;father", "mid", "Dernier train pour Busan", "Une épidémie zombie éclate à bord d'un train à grande vitesse."),
    # ---- Science-fiction / Drame / autres
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi", "space;black hole;father daughter;nolan", "head", "", "Des explorateurs traversent un trou de ver pour trouver une nouvelle planète habitable."),
    ("Blade Runner 2049", 2017, "Drama|Mystery|Sci-Fi", "cyberpunk;replicant;neo-noir;dystopia", "mid", "", "Un réplicant policier découvre un secret qui pourrait bouleverser la société."),
    ("Arrival", 2016, "Drama|Mystery|Sci-Fi", "aliens;language;time;cerebral", "mid", "Premier contact", "Une linguiste tente de communiquer avec des extraterrestres arrivés sur Terre."),
    ("Moon", 2009, "Drama|Mystery|Sci-Fi", "space;isolation;clone;indie", "tail", "", "Seul sur une base lunaire, un technicien fait une découverte troublante."),
    ("Primer", 2004, "Drama|Sci-Fi|Thriller", "time travel;low budget;cerebral;engineers", "tail", "", "Deux ingénieurs inventent par accident une machine à voyager dans le temps."),
    ("Ex Machina", 2014, "Drama|Sci-Fi|Thriller", "artificial intelligence;robot;turing test", "mid", "", "Un programmeur doit évaluer la conscience d'une IA humanoïde."),
    ("Stalker", 1979, "Drama|Sci-Fi", "tarkovsky;zone;philosophical;slow", "tail", "", "Un guide mène deux hommes dans une Zone interdite où un vœu pourrait être exaucé."),
    ("The Shawshank Redemption", 1994, "Crime|Drama", "prison;friendship;hope", "head", "Les Évadés", "Un banquier condamné à tort se lie d'amitié avec un détenu en prison."),
    ("Pulp Fiction", 1994, "Comedy|Crime|Drama", "tarantino;nonlinear;gangsters;dialogue", "head", "", "Des histoires de truands à Los Angeles s'entremêlent dans le désordre."),
    ("The Godfather", 1972, "Crime|Drama", "mafia;family;classic", "head", "Le Parrain", "Le fils cadet d'une famille mafieuse devient malgré lui son chef impitoyable."),
    ("Parasite", 2019, "Comedy|Drama|Thriller", "class;korean;twist;bong joon-ho", "head", "", "Une famille pauvre s'infiltre peu à peu chez une famille riche."),
    ("Memories of Murder", 2003, "Crime|Drama|Mystery|Thriller", "serial killer;korean;investigation;bong joon-ho", "tail", "", "Deux policiers de campagne enquêtent sur une série de meurtres dans la Corée des années 80."),
    ("Prisoners", 2013, "Crime|Drama|Mystery|Thriller", "kidnapping;investigation;dark", "mid", "", "Un père prend les choses en main quand sa fille disparaît."),
    ("Zodiac", 2007, "Crime|Drama|Mystery|Thriller", "serial killer;journalism;obsession;fincher", "mid", "", "Un dessinateur de presse s'obsède pour un tueur en série jamais arrêté."),
    ("Se7en", 1995, "Crime|Mystery|Thriller", "serial killer;detectives;seven sins;dark", "head", "Seven", "Deux inspecteurs traquent un tueur qui s'inspire des sept péchés capitaux."),
    ("La Haine", 1995, "Crime|Drama", "banlieue;french;black and white;police", "mid", "", "Vingt-quatre heures dans la vie de trois jeunes de banlieue après une émeute."),
    ("Un prophète", 2009, "Crime|Drama", "prison;french;mafia;coming of age", "tail", "A Prophet", "Un jeune détenu gravit les échelons du crime organisé en prison."),
    ("Amélie", 2001, "Comedy|Romance", "paris;whimsical;french;feel-good", "head", "Le Fabuleux Destin d'Amélie Poulain;Amelie", "Une serveuse de Montmartre décide d'améliorer en secret la vie des autres."),
    ("Intouchables", 2011, "Comedy|Drama", "friendship;french;feel-good;disability", "head", "The Intouchables", "Un aristocrate tétraplégique engage un jeune de banlieue comme aide à domicile."),
    ("The Grand Budapest Hotel", 2014, "Comedy|Drama", "wes anderson;hotel;quirky;caper", "mid", "", "Un concierge légendaire et son groom sont mêlés au vol d'un tableau."),
    ("Superbad", 2007, "Comedy", "teen;party;friendship", "mid", "SuperGrave", "Deux lycéens tentent d'apporter de l'alcool à une soirée avant la fin du lycée."),
    ("Hot Fuzz", 2007, "Action|Comedy|Crime", "british;cop;parody;village", "mid", "", "Un super-flic londonien muté dans un village trop paisible pour être honnête."),
    ("What We Do in the Shadows", 2014, "Comedy|Horror", "vampires;mockumentary;new zealand", "tail", "Vampires en toute intimité", "Un faux documentaire sur des colocataires vampires à Wellington."),
    ("The Big Lebowski", 1998, "Comedy|Crime", "bowling;cult;slacker;coen brothers", "mid", "", "Un fainéant confondu avec un millionnaire est entraîné dans une affaire de rançon."),
    ("Get Out", 2017, "Horror|Mystery|Thriller", "social thriller;race;twist", "mid", "", "Un jeune homme noir découvre un secret glaçant chez les parents de sa copine."),
    ("Hereditary", 2018, "Drama|Horror|Mystery", "grief;occult;family;disturbing", "mid", "Hérédité", "Après la mort de la grand-mère, une famille découvre un héritage terrifiant."),
    ("The Shining", 1980, "Horror|Mystery", "hotel;kubrick;isolation;madness", "head", "Shining", "Un écrivain gardien d'un hôtel isolé sombre dans la folie."),
    ("It Follows", 2014, "Horror|Mystery", "curse;indie;synth;teen", "tail", "", "Une malédiction transmise sexuellement pousse une entité à suivre sa victime."),
    ("The Thing", 1982, "Horror|Sci-Fi", "antarctica;alien;paranoia;practical effects", "mid", "The Thing;La Chose", "Une équipe en Antarctique affronte une créature qui imite ses victimes."),
    ("Spirited Away", 2001, "Adventure|Animation|Fantasy", "ghibli;miyazaki;spirits;coming of age", "head", "Le Voyage de Chihiro", "Une fillette doit travailler dans un établissement de bains pour esprits pour sauver ses parents."),
    ("Toy Story", 1995, "Adventure|Animation|Children|Comedy", "pixar;toys;friendship;family", "head", "", "Les jouets d'un enfant s'animent quand il a le dos tourné."),
    ("Spider-Man: Into the Spider-Verse", 2018, "Action|Adventure|Animation", "superhero;multiverse;stylized;coming of age", "mid", "Spider-Man New Generation", "Un ado de Brooklyn rencontre des Spider-Men venus d'autres univers."),
    ("Perfect Blue", 1997, "Animation|Horror|Thriller", "satoshi kon;idol;psychological;identity", "tail", "", "Une idole japonaise devenue actrice voit la frontière entre réel et fiction se brouiller."),
    ("Princess Mononoke", 1997, "Action|Adventure|Animation|Fantasy", "ghibli;nature;gods;epic", "mid", "Princesse Mononoké", "Un prince maudit se retrouve entre les dieux de la forêt et une cité minière."),
    ("La La Land", 2016, "Comedy|Drama|Musical|Romance", "jazz;los angeles;dreams", "head", "", "Une actrice et un pianiste de jazz tombent amoureux en poursuivant leurs rêves."),
    ("Before Sunrise", 1995, "Drama|Romance", "dialogue;vienna;train;one night", "tail", "", "Deux inconnus rencontrés dans un train passent une nuit à parler dans Vienne."),
    ("In the Mood for Love", 2000, "Drama|Romance", "wong kar-wai;hong kong;melancholic;stylized", "tail", "", "Deux voisins découvrent que leurs conjoints ont une liaison."),
    ("Titanic", 1997, "Drama|Romance", "ship;disaster;epic", "head", "", "Une passagère de première classe tombe amoureuse d'un artiste pauvre à bord du Titanic."),
    ("The Good, the Bad and the Ugly", 1966, "Action|Adventure|Western", "spaghetti western;leone;gold;duel", "mid", "Le Bon, la Brute et le Truand", "Trois hommes cherchent un trésor caché pendant la guerre de Sécession."),
    ("No Country for Old Men", 2007, "Crime|Drama|Thriller|Western", "coen brothers;hitman;texas;cat and mouse", "mid", "Non, ce pays n'est pas pour le vieil homme", "Un chasseur trouve une valise d'argent et devient la proie d'un tueur implacable."),
    ("Saving Private Ryan", 1998, "Action|Drama|War", "world war ii;d-day;spielberg", "head", "Il faut sauver le soldat Ryan", "Un commando traverse la Normandie pour ramener un soldat chez lui."),
    ("Come and See", 1985, "Drama|War", "world war ii;belarus;harrowing", "tail", "Requiem pour un massacre", "Un adolescent biélorusse découvre l'horreur de la guerre."),
    ("Paddington 2", 2017, "Adventure|Children|Comedy", "bear;london;feel-good;family", "mid", "", "Un ourson accusé à tort d'un vol tente de prouver son innocence."),
    ("Knives Out", 2019, "Comedy|Crime|Mystery", "whodunit;detective;family", "mid", "À couteaux tirés", "Un détective enquête sur la mort d'un riche romancier entouré d'une famille avide."),
    ("Whiplash", 2014, "Drama", "jazz;drums;mentor;obsession", "mid", "", "Un jeune batteur subit les méthodes extrêmes d'un professeur tyrannique."),
    ("Everything Everywhere All at Once", 2022, "Action|Adventure|Comedy|Sci-Fi", "multiverse;family;martial arts;absurd", "mid", "", "Une gérante de laverie doit sauver le multivers en explorant ses autres vies."),
]


def load_mock_catalog() -> pd.DataFrame:
    rows = []
    for i, (title, year, genres, tags, bucket, aliases, overview) in enumerate(_FILMS, start=1):
        rows.append(
            dict(
                movie_id=i,
                title=title,
                year=year,
                tmdb_id=None,
                redial_id=None,
                genres=genres,
                tags=tags,
                overview=overview,
                poster_url=None,
                popularity_bucket=bucket,
                aliases=aliases,
            )
        )
    return pd.DataFrame(rows)
