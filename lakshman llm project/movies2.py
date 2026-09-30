import sys

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
DATASET_DESCRIPTION = "User-provided dataset of 100 movies across international genres and decades."

MOVIE_FIELDS = ("title", "director", "year", "genre", "mood", "description")
MOVIES = [
    dict(zip(MOVIE_FIELDS, record))
    for record in [
        ("The Shawshank Redemption", "Frank Darabont", 1994, "Drama", "Hopeful, emotional, inspirational", "A banker is imprisoned for a crime he did not commit and gradually builds a remarkable life of resilience, friendship, and hope inside prison."),
        ("The Godfather", "Francis Ford Coppola", 1972, "Crime, Drama", "Dark, intense, powerful", "The aging patriarch of a powerful crime family prepares to hand over control of his empire while his reluctant son becomes increasingly involved in the family business."),
        ("There Will Be Blood", "Paul Thomas Anderson", 2007, "Drama, Historical", "Dark, intense, ruthless", "An ambitious oil prospector builds an enormous fortune while his obsession with power and wealth destroys his relationships and humanity."),
        ("Forrest Gump", "Robert Zemeckis", 1994, "Drama, Romance, Comedy", "Heartwarming, nostalgic, emotional", "A kind-hearted man with a simple outlook on life unintentionally becomes involved in major historical events while searching for the woman he loves."),
        ("Dallas Buyers Club", "Jean-Marc Vallée", 2013, "Biography, Drama", "Gritty, emotional, inspiring", "After being diagnosed with HIV, a Texas electrician fights the medical establishment and creates an underground network to provide alternative treatments to patients."),
        ("The Shining", "Stanley Kubrick", 1980, "Horror, Psychological Thriller", "Eerie, disturbing, tense", "A writer takes a winter caretaker job at an isolated hotel where supernatural forces and psychological deterioration threaten his family."),
        ("Dead Poets Society", "Peter Weir", 1989, "Drama", "Inspirational, emotional, reflective", "An unconventional English teacher inspires his students to think independently, embrace life, and challenge the expectations imposed on them."),
        ("Cars", "John Lasseter", 2006, "Animation, Comedy, Family", "Fun, heartwarming, uplifting", "A selfish race car becomes stranded in a small town and learns valuable lessons about friendship, humility, and what truly matters."),
        ("Toy Story", "John Lasseter", 1995, "Animation, Adventure, Comedy", "Fun, nostalgic, heartwarming", "A cowboy toy feels threatened when a new space-themed toy arrives and becomes his owner's favorite."),
        ("Good Will Hunting", "Gus Van Sant", 1997, "Drama, Romance", "Emotional, thoughtful, inspirational", "A brilliant but troubled young man from a working-class background struggles to overcome his past with the help of a therapist."),
        ("The Dark Knight", "Christopher Nolan", 2008, "Action, Crime, Drama", "Dark, intense, suspenseful", "Batman faces a criminal mastermind who pushes Gotham City and its heroes toward moral and psychological chaos."),
        ("Iron Man", "Jon Favreau", 2008, "Action, Science Fiction, Superhero", "Exciting, humorous, adventurous", "A wealthy weapons manufacturer builds an advanced armored suit after being captured by terrorists and begins transforming into a superhero."),
        ("Spider-Man", "Sam Raimi", 2002, "Action, Superhero, Romance", "Exciting, emotional, fun", "A shy teenager gains spider-like abilities and learns that extraordinary powers come with extraordinary responsibilities."),
        ("F1", "Joseph Kosinski", 2025, "Sports, Drama, Action", "Thrilling, intense, inspirational", "A former Formula One driver returns to racing to mentor a young teammate while attempting to revive his career and help his struggling team."),
        ("Whiplash", "Damien Chazelle", 2014, "Drama, Music", "Intense, obsessive, stressful", "An ambitious young jazz drummer enters a brutal relationship with an abusive instructor who pushes him toward musical greatness."),
        ("Sinners", "Ryan Coogler", 2025, "Horror, Thriller, Drama", "Dark, atmospheric, intense", "Twin brothers return to their hometown and become caught in a supernatural threat while attempting to build a new life through music."),
        ("Deadpool & Wolverine", "Shawn Levy", 2024, "Action, Comedy, Superhero", "Hilarious, chaotic, violent", "Deadpool teams up with Wolverine on a multiverse-spanning mission that forces two very different heroes to work together."),
        ("Dune", "Denis Villeneuve", 2021, "Science Fiction, Adventure, Drama", "Epic, atmospheric, serious", "A young nobleman becomes involved in an interplanetary struggle over a desert planet that contains the universe's most valuable resource."),
        ("The Holdovers", "Alexander Payne", 2023, "Comedy, Drama", "Warm, melancholic, heartfelt", "A difficult teacher, a rebellious student, and a grieving cook form an unexpected bond while spending Christmas together at an empty boarding school."),
        ("The Hangover", "Todd Phillips", 2009, "Comedy", "Hilarious, chaotic, outrageous", "Three friends wake up in Las Vegas after a wild bachelor party with no memory of the previous night and must find the missing groom."),
        ("Everything Everywhere All at Once", "Daniel Kwan and Daniel Scheinert", 2022, "Science Fiction, Comedy, Action", "Surreal, emotional, chaotic", "A struggling laundromat owner discovers that she must connect with alternate versions of herself across the multiverse to stop an existential threat."),
        ("Oppenheimer", "Christopher Nolan", 2023, "Biography, Drama, History", "Intense, serious, haunting", "The story of J. Robert Oppenheimer and his leadership of the Manhattan Project, followed by the personal and political consequences of creating the atomic bomb."),
        ("Arrival", "Denis Villeneuve", 2016, "Science Fiction, Drama, Mystery", "Thoughtful, emotional, mysterious", "A linguist is recruited to communicate with mysterious extraterrestrial visitors and gradually develops a new understanding of language, time, and human experience."),
        ("Incendies", "Denis Villeneuve", 2010, "Drama, Mystery, War", "Dark, tragic, emotionally devastating", "After their mother's death, twins travel to the Middle East to uncover their family's hidden history and discover a devastating truth about their origins."),
        ("Inception", "Christopher Nolan", 2010, "Science Fiction, Action, Thriller", "Mind-bending, intense, suspenseful", "A skilled thief who enters people's dreams is offered a chance to erase his criminal past by planting an idea inside another person's mind."),
        ("Midnight in Paris", "Woody Allen", 2011, "Fantasy, Comedy, Romance", "Whimsical, nostalgic, romantic", "A screenwriter visiting Paris discovers a mysterious way to travel into the city's artistic past and meets legendary writers and artists."),
        ("The Prestige", "Christopher Nolan", 2006, "Mystery, Drama, Thriller", "Dark, mysterious, obsessive", "Two rival magicians become consumed by their competition, sacrificing relationships and morality in their pursuit of the ultimate illusion."),
        ("Eyes Wide Shut", "Stanley Kubrick", 1999, "Psychological Drama, Mystery, Thriller", "Dreamlike, unsettling, mysterious", "A married doctor embarks on a mysterious nighttime journey after his wife reveals intimate fantasies that challenge his understanding of their relationship."),
        ("The Devil Wears Prada", "David Frankel", 2006, "Comedy, Drama, Fashion", "Stylish, entertaining, ambitious", "A young aspiring journalist becomes assistant to a demanding fashion magazine editor and struggles to balance career ambition with her personal values."),
        ("Eternal Sunshine of the Spotless Mind", "Michel Gondry", 2004, "Science Fiction, Romance, Drama", "Melancholic, romantic, surreal", "After a painful breakup, a couple undergoes a procedure to erase memories of each other but discovers that love may survive even without memories."),
        ("Django Unchained", "Quentin Tarantino", 2012, "Western, Drama, Action", "Violent, stylish, revenge-driven", "A freed slave becomes a bounty hunter and sets out to rescue his wife from a brutal plantation owner."),
        ("Pulp Fiction", "Quentin Tarantino", 1994, "Crime, Drama, Black Comedy", "Stylish, violent, unpredictable", "Several interconnected stories involving criminals, hitmen, boxers, and gangsters unfold through a nonlinear narrative."),
        ("Inglourious Basterds", "Quentin Tarantino", 2009, "War, Drama, Action", "Tense, violent, darkly humorous", "During World War II, a group of Jewish-American soldiers and a cinema owner develop separate plans to strike against the Nazi leadership."),
        ("All Quiet on the Western Front", "Edward Berger", 2022, "War, Drama, Historical", "Grim, tragic, brutal", "A young German soldier experiences the horrifying reality of trench warfare during World War I."),
        ("1917", "Sam Mendes", 2019, "War, Drama, Historical", "Tense, immersive, emotional", "Two young British soldiers must cross enemy territory to deliver a message that could save hundreds of soldiers from a deadly trap."),
        ("Full Metal Jacket", "Stanley Kubrick", 1987, "War, Drama", "Harsh, disturbing, bleak", "Two stages of a Marine's experience during the Vietnam War reveal the brutal psychological effects of military training and combat."),
        ("The Wolf of Wall Street", "Martin Scorsese", 2013, "Biography, Crime, Comedy", "Wild, excessive, energetic", "A stockbroker rises to extraordinary wealth through aggressive sales tactics, fraud, and corruption while living an increasingly extravagant lifestyle."),
        ("What's Eating Gilbert Grape", "Lasse Hallström", 1993, "Drama, Coming-of-Age, Romance", "Emotional, gentle, bittersweet", "A young man trapped by family responsibilities struggles to find his own identity while caring for his troubled family in a small town."),
        ("Titanic", "James Cameron", 1997, "Romance, Drama, Historical", "Romantic, tragic, epic", "A young aristocratic woman and a poor artist fall in love aboard the ill-fated RMS Titanic."),
        ("27 Dresses", "Anne Fletcher", 2008, "Romantic Comedy, Romance", "Lighthearted, romantic, charming", "A woman who has served as a bridesmaid 27 times finally confronts her own romantic desires when her sister becomes engaged to the man she secretly loves."),
        ("Laapataa Ladies", "Kiran Rao", 2024, "Comedy, Drama, Social", "Heartwarming, humorous, uplifting", "Two newlywed brides become accidentally separated from their husbands during a train journey, leading to an unexpected story about identity, independence, and society."),
        ("Dangal", "Nitesh Tiwari", 2016, "Biography, Sports, Drama", "Inspirational, emotional, determined", "A former wrestler trains his daughters to become world-class wrestlers despite social opposition and personal challenges."),
        ("12th Fail", "Vidhu Vinod Chopra", 2023, "Biography, Drama, Inspirational", "Inspirational, emotional, hopeful", "A young man from a poor background refuses to give up after academic failure and works toward becoming an Indian civil servant."),
        ("Barfi!", "Anurag Basu", 2012, "Romance, Comedy, Drama", "Heartwarming, bittersweet, whimsical", "A charming young man who cannot speak or hear experiences friendship and love through two women while navigating life's difficulties."),
        ("Kal Ho Naa Ho", "Nikkhil Advani", 2003, "Romance, Drama, Comedy", "Emotional, romantic, bittersweet", "A cheerful man enters the lives of two friends in New York and changes their understanding of love, friendship, and living in the present."),
        ("Sultan", "Ali Abbas Zafar", 2016, "Sports, Drama, Romance", "Inspirational, emotional, determined", "A former wrestling champion struggles with personal failure before fighting his way back into the sport and rebuilding his life."),
        ("Zindagi Na Milegi Dobara", "Zoya Akhtar", 2011, "Comedy, Drama, Adventure", "Feel-good, adventurous, reflective", "Three childhood friends take a road trip through Spain that forces them to confront their fears, relationships, and ideas about life."),
        ("Dev.D", "Anurag Kashyap", 2009, "Romance, Drama, Black Comedy", "Dark, rebellious, chaotic", "A modern adaptation of Devdas follows a self-destructive young man as he spirals through heartbreak, addiction, and reckless relationships."),
        ("Andhadhun", "Sriram Raghavan", 2018, "Crime, Thriller, Black Comedy", "Suspenseful, darkly humorous, unpredictable", "A pianist pretending to be blind becomes entangled in a murder mystery after witnessing a crime he was never supposed to see."),
        ("Dil Dhadakne Do", "Zoya Akhtar", 2015, "Comedy, Drama, Family", "Entertaining, emotional, sophisticated", "A wealthy dysfunctional family takes a Mediterranean cruise where hidden conflicts, relationships, and personal expectations come to the surface."),
        ("Uri: The Surgical Strike", "Aditya Dhar", 2019, "War, Action, Drama", "Patriotic, intense, inspirational", "An Indian military officer plans a covert operation following a deadly attack on Indian soldiers."),
        ("3 Idiots", "Rajkumar Hirani", 2009, "Comedy, Drama, Education", "Funny, inspirational, emotional", "Three engineering students navigate friendship, academic pressure, family expectations, and the pursuit of meaningful success."),
        ("Munna Bhai M.B.B.S.", "Rajkumar Hirani", 2003, "Comedy, Drama, Social", "Funny, heartwarming, inspirational", "A gangster enters medical college to fulfill his father's dream and learns that compassion can be more important than conventional medical training."),
        ("Chhichhore", "Nitesh Tiwari", 2019, "Comedy, Drama, Romance", "Emotional, nostalgic, inspirational", "A father recalls his college years to help his son understand that failure is not the end of life."),
        ("Kai Po Che!", "Abhishek Kapoor", 2013, "Drama, Sports, Friendship", "Emotional, intense, bittersweet", "Three friends build a cricket academy while their friendship is tested by ambition, politics, communal tensions, and tragedy."),
        ("Memories of Murder", "Bong Joon Ho", 2003, "Crime, Mystery, Thriller", "Dark, haunting, suspenseful", "Two detectives investigate a series of murders in rural South Korea while struggling with limited evidence and an elusive killer."),
        ("The Wailing", "Na Hong-jin", 2016, "Horror, Mystery, Thriller", "Eerie, disturbing, mysterious", "A police officer investigates a series of strange illnesses and violent incidents in a rural village after the arrival of a mysterious stranger."),
        ("Parasite", "Bong Joon Ho", 2019, "Thriller, Drama, Black Comedy", "Dark, satirical, suspenseful", "A poor family gradually infiltrates the household of a wealthy family, leading to increasingly dangerous consequences."),
        ("Spirited Away", "Hayao Miyazaki", 2001, "Animation, Fantasy, Adventure", "Magical, mysterious, emotional", "A young girl enters a mysterious spirit world and must find courage and resourcefulness to rescue her transformed parents."),
        ("Grave of the Fireflies", "Isao Takahata", 1988, "Animation, War, Drama", "Tragic, heartbreaking, devastating", "Two siblings struggle to survive in Japan during the final months of World War II after losing their home and family."),
        ("Oldboy", "Park Chan-wook", 2003, "Mystery, Thriller, Action", "Dark, violent, disturbing", "After being imprisoned for 15 years without explanation, a man is suddenly released and begins a violent search for the person responsible."),
        ("Chungking Express", "Wong Kar-wai", 1994, "Romance, Crime, Drama", "Dreamy, melancholic, romantic", "Two lonely Hong Kong police officers experience unexpected encounters with women while dealing with heartbreak and loneliness."),
        ("Cure", "Kiyoshi Kurosawa", 1997, "Psychological Horror, Mystery, Crime", "Hypnotic, unsettling, eerie", "A detective investigates a series of seemingly unrelated murders whose perpetrators all confess without understanding why they committed them."),
        ("Perfect Blue", "Satoshi Kon", 1997, "Psychological Thriller, Animation, Horror", "Disturbing, surreal, intense", "A former pop idol becomes increasingly psychologically unstable as the boundaries between her real life, career, and fictional persona begin to disappear."),
        ("Train to Busan", "Yeon Sang-ho", 2016, "Horror, Action, Thriller", "Intense, emotional, terrifying", "Passengers aboard a high-speed train struggle to survive when a zombie outbreak spreads across South Korea."),
        ("Forgotten", "Jang Hang-jun", 2017, "Mystery, Thriller, Psychological", "Mysterious, tense, mind-bending", "A young man investigates the strange disappearance and return of his brother, only to discover that his memories may not reflect reality."),
        ("Seven Samurai", "Akira Kurosawa", 1954, "Action, Drama, Adventure", "Epic, heroic, emotional", "Seven samurai are hired by a poor village to protect its people from a group of bandits."),
        ("Akira", "Katsuhiro Otomo", 1988, "Animation, Science Fiction, Cyberpunk", "Dark, chaotic, intense", "In a dystopian Neo-Tokyo, a biker gang member develops terrifying psychic powers that threaten to unleash catastrophic destruction."),
        ("Incantation", "Kevin Ko", 2022, "Horror, Found Footage, Mystery", "Terrifying, disturbing, occult", "A mother attempts to protect her daughter from a dangerous curse after violating a mysterious religious taboo."),
        ("Ikiru", "Akira Kurosawa", 1952, "Drama", "Reflective, emotional, profound", "A terminally ill bureaucrat realizes that he has wasted his life and desperately searches for a meaningful way to spend his remaining time."),
        ("Kumbalangi Nights", "Madhu C. Narayanan", 2019, "Drama, Family, Romance", "Warm, realistic, emotional", "Four brothers living in a dysfunctional family gradually learn to overcome their differences and build healthier relationships."),
        ("Eko", "Dinjith Ayyathan", 2025, "Mystery, Thriller, Horror", "Atmospheric, mysterious, tense", "A suspense-driven Malayalam mystery involving hidden secrets and unexplained events that gradually reveal a larger truth."),
        ("Ustad Hotel", "Anwar Rasheed", 2012, "Drama, Romance, Food", "Warm, uplifting, emotional", "A young man who dreams of becoming a chef reconnects with his grandfather and discovers deeper meaning in food, family, and life."),
        ("Ennu Ninte Moideen", "R. S. Vimal", 2015, "Romance, Drama, Biography", "Romantic, tragic, emotional", "Based on a real-life love story about Moideen and Kanchanamala whose relationship faces social and political barriers in 1960s Kerala."),
        ("Aattam", "Anand Ekarshi", 2023, "Drama, Mystery, Social", "Tense, thought-provoking, uncomfortable", "After a troubling incident involving a theatre group, its members debate what happened and reveal their conflicting values and prejudices."),
        ("Charlie", "Martin Prakkat", 2015, "Adventure, Romance, Drama", "Whimsical, feel-good, adventurous", "A young woman follows the trail of a mysterious free-spirited man whose unconventional life inspires her own journey of discovery."),
        ("Premam", "Alphonse Puthren", 2015, "Romance, Comedy, Drama", "Nostalgic, romantic, youthful", "A man's romantic life unfolds through three different stages, exploring love, heartbreak, friendship, and growing up."),
        ("Thanmathra", "Blessy", 2005, "Drama, Family", "Emotional, heartbreaking, realistic", "A devoted family struggles when a successful man's life is gradually affected by Alzheimer's disease."),
        ("Bramayugam", "Rahul Sadasivan", 2024, "Horror, Folk Horror, Thriller", "Dark, atmospheric, terrifying", "A young folk singer becomes trapped in an isolated ancestral mansion where supernatural forces and sinister secrets threaten his survival."),
        ("Dies Irae", "Rahul Sadasivan", 2025, "Horror, Psychological Thriller, Mystery", "Dark, eerie, disturbing", "A man begins experiencing increasingly disturbing supernatural phenomena after a traumatic event, leading him into a terrifying confrontation with an unexplained force."),
        ("Kishkindha Kaandam", "Dinjith Ayyathan", 2024, "Mystery, Thriller, Drama", "Mysterious, tense, cerebral", "A newly married man discovers disturbing secrets within his wife's family while investigating a mysterious disappearance."),
        ("Aavesham", "Jithu Madhavan", 2024, "Action, Comedy, Drama", "Wild, energetic, hilarious", "Three college students seek the help of a flamboyant gangster in Bengaluru but quickly discover that their new mentor is far more unpredictable than expected."),
        ("Memories", "Jeethu Joseph", 2013, "Crime, Mystery, Thriller", "Dark, suspenseful, emotional", "A troubled police officer returns to investigate a series of murders that appear connected by a disturbing pattern."),
        ("Njan Prakashan", "Sathyan Anthikad", 2018, "Comedy, Drama, Satire", "Humorous, lighthearted, relatable", "A young man obsessed with escaping his ordinary life repeatedly schemes for an easier path to success but gradually learns important lessons about responsibility."),
        ("Lokah: Chapter 1 – Chandra", "Dominic Arun", 2025, "Superhero, Fantasy, Action", "Stylish, mysterious, adventurous", "A young woman with extraordinary abilities becomes involved in a mysterious supernatural world while confronting threats connected to her hidden identity."),
        ("Meiyazhagan", "C. Prem Kumar", 2024, "Drama, Family, Romance", "Warm, nostalgic, emotional", "A man returning to his hometown unexpectedly reconnects with a joyful and mysterious relative whose presence forces him to confront memories of his past."),
        ("Maharaja", "Nithilan Saminathan", 2024, "Action, Crime, Thriller", "Dark, intense, shocking", "A barber seeks justice after a mysterious incident involving a valuable object, gradually uncovering a complex web of crime and revenge."),
        ("The Pursuit of Happyness", "Gabriele Muccino", 2006, "Biography, Drama, Inspirational", "Inspirational, emotional, hopeful", "A struggling salesman becomes homeless while caring for his young son and relentlessly pursues an opportunity to change their lives."),
        ("Vada Chennai", "Vetrimaaran", 2018, "Crime, Drama, Gangster", "Gritty, violent, intense", "A talented carrom player becomes caught in the violent political and criminal world of North Chennai over several decades."),
        ("96", "C. Prem Kumar", 2018, "Romance, Drama", "Nostalgic, melancholic, emotional", "Two former schoolmates reunite years after graduation and revisit the memories and unresolved emotions of their teenage love."),
        ("Bangalore Days", "Anjali Menon", 2014, "Comedy, Drama, Romance", "Feel-good, youthful, warm", "Three cousins move to Bengaluru and experience love, friendship, career challenges, and personal growth while building new lives."),
        ("O Kadhal Kanmani", "Mani Ratnam", 2015, "Romance, Drama", "Romantic, modern, vibrant", "Two young professionals in Mumbai begin a relationship without wanting marriage but gradually confront their feelings about commitment and love."),
        ("Asuran", "Vetrimaaran", 2019, "Action, Drama, Social", "Gritty, intense, emotional", "A farmer is forced to confront violence and caste-based oppression when his family becomes the target of a powerful landowning family."),
        ("Pather Panchali", "Satyajit Ray", 1955, "Drama, Coming-of-Age", "Poetic, emotional, realistic", "A poor Bengali family struggles with poverty and hardship while young Apu and his sister experience the joys and tragedies of childhood."),
        ("The Green Mile", "Frank Darabont", 1999, "Drama, Fantasy, Prison", "Emotional, supernatural, heartbreaking", "A prison guard develops an extraordinary bond with a gentle death-row inmate who possesses mysterious healing abilities."),
        ("Vikram", "Lokesh Kanagaraj", 2022, "Action, Crime, Thriller", "Intense, violent, stylish", "A covert black-ops team investigates a series of murders and becomes involved in a large criminal network connected to powerful figures."),
        ("Pizza", "Karthik Subbaraj", 2012, "Horror, Mystery, Thriller", "Suspenseful, eerie, unpredictable", "A pizza delivery worker enters a mysterious house and experiences a series of terrifying events that challenge his understanding of reality."),
        ("Pariyerum Perumal", "Mari Selvaraj", 2018, "Drama, Social", "Powerful, emotional, thought-provoking", "A young law student from a marginalized community faces caste discrimination and violence while trying to build friendships and pursue his education."),
        ("Notting Hill", "Roger Michell", 1999, "Romantic Comedy, Romance", "Charming, romantic, lighthearted", "A quiet London bookseller unexpectedly falls in love with a world-famous actress whose celebrity status complicates their relationship."),
        ("Pretty Woman", "Garry Marshall", 1990, "Romantic Comedy, Romance", "Charming, romantic, feel-good", "A wealthy businessman hires a prostitute to accompany him socially and unexpectedly develops a genuine romantic relationship."),
    ]
]


def build_text(item):
    return (
        f"Title: {item['title']}. Director: {item['director']}. "
        f"Genre: {item['genre']}. Year: {item['year']}. "
        f"Mood: {item['mood']}. Description: {item['description']}"
    )


def rank_movies(query, model, items, movie_embeddings, limit=5):
    """Embed the query, then rank movie embeddings by cosine similarity."""
    query_embedding = model.encode([query], convert_to_numpy=True)

    # cosine_similarity returns one cosine score for the query against each movie.
    scores = cosine_similarity(query_embedding, movie_embeddings)[0]
    ranked = sorted(
        zip(items, scores),
        key=lambda result: float(result[1]),
        reverse=True,
    )
    return ranked[:limit]


def print_results(query, results):
    print(f"\nQUERY: {query}\n")
    for index, (item, score) in enumerate(results, start=1):
        print(
            f"{index}. {item['title']} — {item['director']} "
            f"({item['genre']}, {item['year']})"
        )
        print(f"   Cosine similarity: {float(score):.4f}")
        print(f"   Why it matches: {item['description']}")


DEMO_QUERIES = [
    "A hopeful story about a prisoner who builds a friendship and finds freedom.",
    "A suspenseful psychological thriller about a detective investigating strange murders.",
    "A heartwarming Indian sports drama about a parent training his daughters to become wrestlers.",
    "A surreal science-fiction adventure about alternate universes and family relationships.",
    "Recommend a good movie for tonight.",
]

DEMO_ASSESSMENTS = [
    "The Shawshank Redemption should be highly relevant because it directly matches prison, friendship, and hope. Other prison dramas may share the setting without the hopeful friendship focus. This shows how descriptions and mood words help connect related ideas; rankings can still favor broad theme overlap.",
    "Memories of Murder and Cure are plausible strong matches because both involve detectives or investigations and unusual murders. Other crime thrillers may be related but lack the psychological or strange-murder elements. The model captures genre and plot relationships, but short descriptions can omit distinctions.",
    "Dangal is the clearest match: a parent trains his daughters in wrestling. Other sports dramas may share determination and competition but not that family story. The query includes distinctive relationships that help retrieval; the small catalog limits alternatives.",
    "Everything Everywhere All at Once should rank strongly for alternate universes and family relationships. Other science-fiction films may share only one aspect, such as space or time. The model relates paraphrases and themes, but may treat loosely connected science fiction as relevant.",
    "This is a deliberately poor, very broad query: it gives no genre, mood, or story preference. The varied genres in the top five show that the system has little basis for choosing what this user would enjoy. More detail would make retrieval more useful; cosine similarity measures text similarity, not movie quality or personal taste.",
]


def main():
    print("Dataset:", DATASET_DESCRIPTION)
    print(f"Total movies: {len(MOVIES)}")
    if not MOVIES:
        print("Add movie records to MOVIES in movies.py before running recommendations.")
        return

    demo_mode = len(sys.argv) == 2 and sys.argv[1] == "--demo"
    model = SentenceTransformer(MODEL_NAME)
    movie_texts = [build_text(item) for item in MOVIES]
    movie_embeddings = model.encode(movie_texts, convert_to_numpy=True)

    if demo_mode:
        print("\nFive-query retrieval demonstration")
        for query, assessment in zip(DEMO_QUERIES, DEMO_ASSESSMENTS):
            results = rank_movies(query, model, MOVIES, movie_embeddings)
            print_results(query, results)
            print("Manual relevance assessment:", assessment)
        return

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("What kind of movie are you looking for? ").strip()
        if not query:
            query = "Recommend a thoughtful, atmospheric movie with memorable characters."

    results = rank_movies(query, model, MOVIES, movie_embeddings)
    print_results(query, results)


if __name__ == "__main__":
    main()
