POS_BLACKLIST = ['INTJ', 'AUX', 'CCONJ', 
            'ADP', 'DET', 'NUM', 'PART', 
            'PRON', 'SCONJ', 'PUNCT',
            'SYM', 'X']

STOPWORDS = ["word", 
    "a", "a's", "able", "about", "above", "according",
    "accordingly", "across", "actually", "after", "afterwards", 
    "again", "against", "ago", "aim", "ain't", "all", "allow",
    "allows", "almost", "alone", "along", "already", "also", 
    "although", "always", "am", "among", "amongst", "an", "and",
    "another", "any", "anybody", "anyhow", "anyone", "anything", 
    "anyway", "anyways", "anywhere", "apart", "appear", "appreciate",
    "approach", "appropriate", "are", "area", "areas", "aren't", 
    "around", "as", "aside", "ask", "asked", "asking", "asks", 
    "associated", "at", "available", "away", "awfully", "b", "back", 
    "backed", "backing", "backs", "bad", "based", "be", "became",
    "because", "become", "becomes", "becoming", "been", "before", 
    "beforehand", "began", "behind", "being", "beings", "believe", 
    "below", "beside", "besides", "best", "better", "between", 
    "beyond", "big", "bit", "both", "brief", "bring", "but", "by", 
    "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", 
    "case", "cases", "cause", "causes", "certain", "certainly", 
    "changes", "clear", "clearly", "co", "com", "come", "comes", 
    "concerning", "consequently", "consider", "considering", 
    "contain", "containing", "contains", "continue", "corresponding", 
    "could", "couldn't", "course", "currently", "d", "definitely", 
    "described", "despite", "did", "didn't", "differ", "different", 
    "differently", "do", "does", "doesn't", "doing", "don't", "done", 
    "down", "downed", "downing", "downs", "downwards", "dr", "during", 
    "e", "each", "earlier", "early", "edu", "eg", "eight", "either", 
    "else", "elsewhere", "end", "ended", "ending", "ends", "enough", 
    "entirely", "especially", "et", "etc", "even", "evenly", "ever", 
    "every", "everybody", "everyone", "everything", "everywhere", "ex", 
    "exactly", "example", "except", "f", "face", "faces", "fact", 
    "facts", "far", "felt", "few", "fifth", "find", "finds", "first", 
    "five", "flawed", "focusing", "followed", "following", "follows", 
    "for", "former", "formerly", "forth", "four", "from", "full", 
    "fully", "fun", "further", "furthered", "furthering", 
    "furthermore", "furthers", "g", "gave", "general", "generally", 
    "get", "gets", "getting", "gigot", "give", "given", "gives", "go", 
    "goes", "going", "gone", "good", "goods", "got", "gotten", "great", 
    "greater", "greatest", "greetings", "group", "grouped", "grouping", 
    "groups", "h", "had", "hadn't", "half", "happens", "hardly", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", 
    "he's", "held", "hello", "help", "hence", "her", "here", "here's",
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", 
    "hi", "high", "higher", "highest", "him", "himself", "his", 
    "hither", "hopefully", "how", "how's", "howbeit", "however", "i", 
    "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "ii", 
    "immediate", "immediately", "important", "in", "inasmuch", "inc", 
    "include", "including", "indeed", "indicate", "indicated", 
    "indicates", "inevitable", "inner", "insofar", "instead", 
    "interest", "interested", "interesting", "interests", "into", 
    "involving", "inward", "is", "isn't", "issue", "it", "it'd", 
    "it'll", "it's", "its", "itself", "ix", "j", "just", "k", "keep", 
    "keeps", "kept", "kind", "knew", "know", "known", "knows", "l", 
    "large", "largely", "last", "lately", "later", "latest", 
    "latter", "latterly", "lead", "least", "led", "less", "lest", 
    "let", "let's", "lets", "letting", "like", "liked", "likely", 
    "likes", "line", "listen", "little", "long", "longer", "longest", 
    "look", "looking", "looks", "lot", "ltd", "m", "m.d", "made", 
    "mainly", "make", "makes", "making", "man", "many", "may", "maybe", 
    "me", "mean", "meant", "meanwhile", "member", "members", "men", 
    "merely", "messrs", "met", "might", "more", "moreover", "most",
    "mostly", "move", "mr", "mrs", "ms", "much", "must", "mustn't",
    "my", "myself", "n", "name", "namely", "nd", "near", "nearly",
    "necessary", "need", "needed", "needing", "needs", "neither",
    "never", "nevertheless", "new", "newer", "newest", "next",
    "nine", "no", "nobody", "non", "none", "nonetheless", "noone",
    "nor", "normally", "not", "nothing", "novel", "now", "nowhere",
    "number", "numbers", "o", "obviously", "of", "off", "often",
    "oh", "ok", "okay", "old", "older", "oldest", "on", "once",
    "one", "ones", "only", "onto", "open", "opened", "opening", 
    "opens", "or", "order", "ordered", "ordering", "orders", 
    "other", "others", "otherwise", "ought", "our", "ours", 
    "ourselves", "out", "outside", "over", "overall", 
    "overwhelming", "own", "p", "part", "parted", "particular",
    "particularly", "parting", "parts", "people", "per", "perhaps",
    "place", "placed", "places", "please", "plus", "point", "pointed",
    "pointing", "points", "possible", "prefer", "present", "presented", 
    "presenting", "presents", "presumably", "probably", "problem", 
    "problems", "prof", "provides", "put", "puts", "putting", "q", 
    "que", "quite", "qv", "r", "rather", "rd", "re", "really", 
    "reasonably", "recently", "regarding", "regardless", "regards", 
    "relatively", "respectively", "right", "room", "rooms", "s", 
    "said", "same", "saw", "say", "saying", "says", "sec", "second", 
    "secondly", "seconds", "see", "seeing", "seem", "seemed", 
    "seeming", "seemingly", "seems", "seen", "sees", "self", "selves", 
    "sensible", "sent", "serious", "seriously", "set", "seven", 
    "several", "shall", "shan't", "she", "she'd", "she'll", "she's", 
    "shortly", "should", "shouldn't", "show", "showed", "showing", 
    "shows", "side", "sides", "simply", "since", "six", "small", 
    "smaller", "smallest", "so", "some", "somebody", "somehow", 
    "someone", "something", "sometime", "sometimes", "somewhat", 
    "somewhere", "soon", "sorry", "specified", "specify", "specifying",
    "st", "state", "states", "still", "sub", "such", "sup", "sure", 
    "t", "t's", "take", "taken", "tell", "tends", "th", "than", 
    "thank", "thanks", "thanx", "that", "that's", "thats", "the", 
    "their", "theirs", "them", "themselves", "then", "thence", "there",
    "there's", "thereafter", "thereby", "therefore", "therein", 
    "theres", "thereupon", "these", "they", "they'd", "they'll", 
    "they're", "they've", "thing", "things", "think", "thinks", 
    "third", "this", "thorough", "thoroughly", "those", "though", 
    "thought", "thoughts", "three", "through", "throughout", "thru", 
    "thus", "to", "today", "together", "told", "too", "took", "top", 
    "toward", "towards", "tried", "tries", "truly", "try", "trying", 
    "turn", "turned", "turning", "turns", "twice", "two", "u", "un", 
    "under", "unfortunately", "unless", "unlike", "unlikely", "until", 
    "unto", "up", "upon", "us", "use", "used", "useful", "uses",
    "using", "usually", "uucp", "v", "value", "various", "very", "via",
    "viz", "vs", "w", "want", "wanted", "wanting", "wants", "was",
    "wasn't", "watched", "way", "ways", "we", "we'd", "we'll", "we're",
    "we've", "welcome", "well", "wells", "went", "were", "weren't", 
    "what", "what's", "whatever", "when", "when's", "whence",
    "whenever", "where", "where's", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while",
    "whither", "who", "who's", "whoever", "whole", "whom", "whose",
    "why", "why's", "will", "willing", "wish", "with", "within",
    "without", "won't", "wonder", "work", "worked", "working",
    "works", "worst", "would", "wouldn't", "x", "y", "year", "years",
    "yes", "yet", "you", "you'd", "you'll", "you're", "you've",
    "young", "younger", "youngest", "your", "yours", "yourself", 
    "yourselves", "z", "zero", "mr", "ms", "mrs", "mssrs", "mssr", 
    "also", "said", "should", "could", "would", "week", "weeks", 
    "month", "months", "year", "years"]