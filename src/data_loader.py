import re
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from itertools import islice
from tqdm import tqdm

class DataLoader:
    def __init__(self, clean=True, split_sentence=False):
        self.clean = clean
        self.split_sentence = split_sentence

    # ====== Basic preprocessing ======
    def _clean_text(self, text):
        """Clean noise characters at both ends + merge extra spaces."""
        if not self.clean:
            return text
        # text = text.strip(' \n\t\r*"\'`~!@#$%^&*()-_=+[]{}|;:,.<>?/')  # remove unwanted characters at both ends
        text = text.strip(' \n\t\r*"\'`~@#$%^&*()-_=+[]{}|;:,<>/')  # don't remove ".", "!", "?"
        text = re.sub(r'\s+', ' ', text)  # merge multiple whitespace
        return text.strip()

    def _maybe_split(self, text):
        """Split into sentences if enabled."""
        if not self.split_sentence:
            return text
        sentences = sent_tokenize(text)
        return [self._clean_text(s) for s in sentences if s.strip()]

    # ====== Method 1: Load the Essays dataset ======
    def load_essays(self):
        """Load essays (kyssen/kg-gen-evaluation-essays)."""
        ds = load_dataset("kyssen/kg-gen-evaluation-essays")
        raw = ds["train"]["content"]
        list_of_essays = [self._maybe_split(self._clean_text(x)) for x in raw]
        flat_list_of_sentences = [x for sublist in list_of_essays for x in sublist]
        return flat_list_of_sentences

    # ====== Method 2: Load the PG-19 dataset ======
    def load_txt_and_split_sentences(self, filepath):
        # 1. read file
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        text = re.sub(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", "", text, flags=re.S)
        text = re.sub(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*", "", text, flags=re.S)

        #text = re.sub(r"\s+", " ", text).strip()

        sentences = self._maybe_split(self._clean_text(text))

        #sentences = re.split(r'(?<=[.!?])\s+', text)

        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    # ====== Method 3: Load WikiText-103 (by article) ======
    def load_wikitext103(self, split="test"):
        """Load WikiText-103 and split by article titles."""
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        lines = ds[split]["text"]

        articles = []
        current = []

        def is_title(line):
            """Check whether a line is a Wikipedia-style article title."""
            return bool(re.match(r"^=+[^=].*[^=]=+$", line.strip()))

        for line in lines:
            # New article starts when encountering a title
            if is_title(line):
                if current:
                    art = "\n".join(current).strip()
                    articles.append(art)
                    current = []
                current.append(line)
            else:
                current.append(line)

        # Add last article
        if current:
            articles.append("\n".join(current).strip())

        # Apply preprocessing to every article
        list_of_articles = [self._maybe_split(self._clean_text(a)) for a in articles]
        flat_list_of_sentences = [x for sublist in list_of_articles for x in sublist]
        return flat_list_of_sentences

    # ====== Method 4: Load JSON file ======
    def load_json(self, path, field=None):
        """Load a JSON file and optionally extract a specific field."""
        import json
        data = json.load(open(path, 'r', encoding='utf-8'))
        if field:
            raw = [item[field] for item in data]
        else:
            raw = data
        return [self._maybe_split(self._clean_text(x)) for x in raw]


    # ====== Method 5 (FIXED): Load Wikipedia via wiki40b ======
    def load_wikipedia(
        self,
        split="train",
        max_articles=10,
        streaming=True
    ):
        """
        Load Wikipedia articles using wiki40b (script-free, HF-supported).

        Args:
            split: dataset split
            max_articles: number of Wikipedia articles to load
            streaming: whether to stream data (recommended)

        Returns:
            flat_list_of_sentences: List[str]
        """

        ds = load_dataset(
            "wiki40b",
            "en",
            split=split,
            streaming=streaming
        )

        all_sentences = []

        iterator = islice(ds, max_articles)
        for item in tqdm(
            iterator,
            total=max_articles,
            desc="Loading Wikipedia (wiki40b)"
        ):
            text = item.get("text", "")
            if not isinstance(text, str):
                continue

            text = self._clean_text(text)
            if not text:
                continue

            sents = self._maybe_split(text)
            if isinstance(sents, list):
                all_sentences.extend(sents)
            else:
                all_sentences.append(sents)

        return all_sentences



    # ====== Method 6 (FIXED): Load arXiv abstracts (HF) ======
    def load_arxiv_streaming(
            self,
            dataset_id="common-pile/arxiv_abstracts",
            split="train",
            max_papers=10
    ):
        """
        Stream arXiv abstracts and take only the first max_papers items.
        """

        ds = load_dataset(dataset_id, split=split, streaming=True)

        candidate_fields = ["abstract", "text", "summary", "content"]

        all_units = []
        #for item in islice(ds, max_papers):
        for item in tqdm(islice(ds, max_papers), total=max_papers, desc="Loading arXiv abstracts"):
            field = None
            for f in candidate_fields:
                if f in item:
                    field = f
                    break
            if field is None:
                continue

            text = self._clean_text(item[field])
            if not text:
                continue

            sents = self._maybe_split(text)
            if isinstance(sents, list):
                all_units.extend(sents)
            else:
                all_units.append(sents)

        return all_units





if __name__ == "__main__":
    loader = DataLoader(clean=True, split_sentence=True)
    # sentences = loader.load_txt_and_split_sentences("../data/pg1342.txt")
    # sentences = loader.load_txt_and_split_sentences("../data/pg11.txt")
    # sentences = loader.load_wikitext103("test")
    #sentences = loader.load_wikipedia(max_articles=5000)
    #sentences = loader.load_arxiv_streaming(max_papers=5000)
    sentences = loader.load_essays()




