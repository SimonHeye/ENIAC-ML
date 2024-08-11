import torch
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Set, Dict, Union
from torch.utils.data import Dataset, Sampler
from pydantic import BaseModel
from utils import find_span, load_wiki_relation_map, mark_fewrel_entity, mark_wiki_entity
from collections import Counter
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import ipdb

def get_tf_idf_key():
    vectorizer = TfidfVectorizer(use_idf=True,max_df=0.5, min_df=1,ngram_range=(1,3))
    json_data = json.load(open('fewrel_train.json'))

    rel_list = []
    for rel,tokens in json_data.items():
        tt = []
        for text in tokens:
            tt.extend(text['tokens'])
        rel_list.append(' '.join(tt))
    vectors = vectorizer.fit_transform(rel_list)
    dict_of_tokens = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}

    tfidf_vectors = []  # all deoc vectors by tfidf
    for row in vectors:
        tfidf_vectors.append({dict_of_tokens[column]: value for
                              (column, value) in
                              zip(row.indices, row.data)})
    print("The number of document vectors = ", len(tfidf_vectors),
          "\nThe dictionary of document[0] :", tfidf_vectors[0])
    doc_sorted_tfidfs =[]  # 带有tfidf权重的文档特征列表
    # 对文档的每个字典进行排序
    for dn in tfidf_vectors:
        newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
        newD = dict(newD)
        doc_sorted_tfidfs.append(newD)

    tfidf_kw = []
    for doc_tfidf in doc_sorted_tfidfs:
        ll = list(doc_tfidf.keys())
        tfidf_kw.append(ll)

    TopN= 100
    print(tfidf_kw[0][0:TopN])

def train_test_split(*args, **kwargs) -> list:
    raise NotImplementedError

class RelationSentence(BaseModel):
    tokens: List[str]
    head: List[int]
    tail: List[int]
    label: str
    head_id: str = ""
    tail_id: str = ""
    label_id: str = ""
    error: str = ""
    raw: str = ""
    keyword: List[str]=['']
    score: float = 0.0
    zerorc_included: bool = True

    def as_tuple(self) -> Tuple[str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        return head, self.label, tail

    def as_line(self) -> str:
        return self.json() + "\n"

    def is_valid(self) -> bool:
        for x in [self.tokens, self.head, self.tail, self.label]:
            if len(x) == 0:
                return False
        for x in [self.head, self.tail]:
            if -1 in x:
                return False
        return True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @classmethod
    def from_spans(cls, text: str, head: str, tail: str, label: str, keyword=[''], strict=True):
        tokens = text.split()
        sent = cls(
            tokens=tokens,
            head=find_span(head, tokens),
            tail=find_span(tail, tokens),
            label=label,
            keyword=keyword,
        )
        if strict:
            assert sent.is_valid(), (head, label, tail, text)
        return sent

    @classmethod
    def from_spans_rel(cls, text: str, head: str, tail: str, label: str, label_id: str = None, keyword=[''], strict=True):
        tokens = text.split()
        # ipdb.set_trace()
        sent = cls(
            tokens=tokens, 
            head=find_span(head, tokens),
            tail=find_span(tail, tokens),
            label=label,
            label_id=label_id,
            keyword=keyword,
        )
        if strict:
            assert sent.is_valid(), (head, label, tail, text)
        return sent

    def as_marked_text(self) -> str:
        tokens = list(self.tokens)
        for i, template in [
            (self.head[0], "[H {}"),
            (self.head[-1], "{} ]"),
            (self.tail[0], "[T {}"),
            (self.tail[-1], "{} ]"),
        ]:
            tokens[i] = template.format(tokens[i])
        return " ".join(tokens)

class Sentence(BaseModel):
    triplets: List[RelationSentence]

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0

class WikiDataset:
    def __init__(self, mode, data, pid2vec, property2idx):
        assert mode in ["train", "dev", "test"]
        self.mode = mode
        self.data = data
        self.pid2vec = pid2vec
        self.property2idx = property2idx
        self.len = len(self.data)

    def load_edges(
        self, i: int, label_ids: Optional[Set[str]] = None
    ) -> List[RelationSentence]:
        g = self.data[i]
        tokens = g["tokens"]
        sents = []
        for j in range(len(g["edgeSet"])):
            property_id = g["edgeSet"][j]["kbID"]
            edge = g["edgeSet"][j]
            head, tail = mark_wiki_entity(edge)
            if label_ids and property_id not in label_ids:
                continue
            s = RelationSentence(
                tokens=tokens, head=head, tail=tail, label="", label_id=property_id
            )
            sents.append(s)
        return sents

    def __getitem__(self, item: int) -> RelationSentence:
        # The ZS-BERT setting is throw away all except first edge
        return self.load_edges(item)[0]

    def __len__(self):
        return self.len

class Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")

    @classmethod
    def load_fewrel(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        groups = {}

        with open(path) as f:
            for i, lst in tqdm(json.load(f).items()):
                for raw in lst:
                    head, tail = mark_fewrel_entity(raw)
                    t = RelationSentence(
                        tokens=raw["tokens"],
                        head=head,
                        tail=tail,
                        label=relation_map[i].pLabel,
                        label_id=i,
                    )
                    groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        return cls(sents=sents)

    @classmethod
    def load_wiki(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        sents = []
        with open(path) as f:
            ds = WikiDataset(
                mode="train", data=json.load(f), pid2vec=None, property2idx=None
            )
            for i in tqdm(range(len(ds))):
                triplets = ds.load_edges(i)
                triplets = [t for t in triplets if t.label_id in relation_map.keys()]
                for t in triplets:
                    t.label = relation_map[t.label_id].pLabel
                if triplets:
                    # ZSBERT only includes first triplet in each sentence
                    for t in triplets:
                        t.zerorc_included = False
                    triplets[0].zerorc_included = True

                    s = Sentence(triplets=triplets)
                    sents.append(s)

        data = cls(sents=sents)
        counter = Counter(t.label for s in data.sents for t in s.triplets)
        threshold = sorted(counter.values())[-113]  # Based on ZSBERT data stats
        labels = [k for k, v in counter.items() if v >= threshold]
        data = data.filter_labels(labels)
        return data

    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                s = s.copy(deep=True)
                s.triplets = triplets
                sents.append(s)
        return Dataset(sents=sents)

    def train_test_split(self, test_size: int, random_seed: int, by_label: bool):
        random.seed(random_seed)

        if by_label:
            labels = self.get_labels()
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        else:
            sents_train = [s for s in self.sents]
            sents_test = random.sample(self.sents, k=test_size)

        banned = set(s.text for s in sents_test)  # Prevent sentence overlap
        sents_train = [s for s in sents_train if s.text not in banned]
        assert len(self.sents) == len(sents_train) + len(sents_test)
        return Dataset(sents=sents_train), Dataset(sents=sents_test)

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            unique_texts=len(set(s.triplets[0].text for s in self.sents)),
            lengths=str(Counter(len(s.triplets) for s in self.sents)),
            labels=len(self.get_labels()),
        )
        print(json.dumps(info, indent=2))



class RelationData(BaseModel):
    sents: List[RelationSentence]

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            lines = f.readlines()
            sents = [
                RelationSentence(**json.loads(x))
                for x in tqdm(lines, desc="RelationData.load")
            ]
        return cls(sents=sents)

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write("".join([s.as_line() for s in self.sents]))

    @property
    def unique_labels(self) -> List[str]:
        return sorted(set([s.label for s in self.sents]))

    def train_test_split(
        self, test_size: Union[int, float], random_seed: int, by_label: bool = False
    ):
        if by_label:
            labels_train, labels_test = train_test_split(
                self.unique_labels, test_size=test_size, random_state=random_seed
            )
            train = [s for s in self.sents if s.label in labels_train]
            test = [s for s in self.sents if s.label in labels_test]
        else:
            groups = self.to_sentence_groups()
            keys_train, keys_test = train_test_split(
                sorted(groups.keys()), test_size=test_size, random_state=random_seed
            )
            train = [s for k in keys_train for s in groups[k]]
            test = [s for k in keys_test for s in groups[k]]

        # Enforce no sentence overlap
        texts_test = set([s.text for s in test])
        train = [s for s in train if s.text not in texts_test]

        data_train = RelationData(sents=train)
        data_test = RelationData(sents=test)
        if by_label:
            assert len(data_test.unique_labels) == test_size
            assert not set(data_train.unique_labels).intersection(
                data_test.unique_labels
            )

        info = dict(
            sents_train=len(data_train.sents),
            sents_test=len(data_test.sents),
            labels_train=len(data_train.unique_labels),
            labels_test=len(data_test.unique_labels),
        )
        print(json.dumps(info, indent=2))
        return data_train, data_test

    def to_sentence_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.text, []).append(s)
        return groups

    def to_label_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.label, []).append(s)
        return groups

    def filter_group_sizes(self, min_size: int = 0, max_size: int = 999):
        groups = self.to_sentence_groups()
        sents = [
            s
            for k, lst in groups.items()
            for s in lst
            if min_size <= len(lst) <= max_size
        ]
        return RelationData(sents=sents)

    def filter_errors(self):
        def check_valid_span(span: List[int]) -> bool:
            start = sorted(span)[0]
            end = sorted(span)[-1] + 1
            return span == list(range(start, end))

        sents = []
        for s in self.sents:
            if s.is_valid():
                if check_valid_span(s.head) and check_valid_span(s.tail):
                    sents.append(s)

        print(dict(filter_errors_success=len(sents) / len(self.sents)))
        return RelationData(sents=sents)

    def analyze(self, header: Optional[str] = None):
        labels = self.unique_labels
        groups = self.to_sentence_groups()
        spans = []
        words = []
        for s in self.sents:
            head, label, tail = s.as_tuple()
            spans.append(head)
            spans.append(tail)
            words.extend(s.tokens)
        info = dict(
            header=header,
            sents=len(self.sents),
            labels=str([len(labels), labels]),
            unique_texts=len(groups.keys()),
            unique_spans=len(set(spans)),
            unique_words=len(set(words)),
            group_sizes=str(Counter([len(lst) for lst in groups.values()])),
        )
        print(json.dumps(info, indent=2))
        return info