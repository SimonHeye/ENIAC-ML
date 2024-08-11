import json
from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer
from zsre_dataset import RelationData, RelationSentence
import ipdb

def encode_to_line(x: str, y: str) -> str:
    # Refer to original transformers readme
    text = json.dumps(dict(text=x, summary=y)) + "\n"
    assert decode_from_line(text) == (x, y)
    return text

def decode_from_line(text: str) -> Tuple[str, str]:
    # ipdb.set_trace()
    d = json.loads(text)
    # return text["text"], text["summary"]
    return d["text"], d["summary"]

class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError

    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def encode_to_line_with_keyword(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError


class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"[SENT] : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("[SENT] : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"[HEAD] {s} , [TAIL] {o} , [REL] {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(", [REL] ")
        label = label.strip()[:-2]
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return RelationSentence.from_spans(context, head.strip(), tail.strip(), label.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"[HEAD] {head} , [TAIL] {tail} , [REL] "

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def decode_to_relationprompt(self, y: str,):
        front, label = y.split(", [REL] ")
        # label = label.strip()
        label = label.strip()[:-2]
        # ipdb.set_trace()
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return head.strip(), tail.strip(), label.strip()

class ExtractEncoder_plus(Encoder):
    def encode_x(self, text: str) -> str:
        return f"[SENT] : {text}"

    def decode_x(self, x: str) -> str:
        return x.split("[SENT] : ")[-1]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"[HEAD] {s} , [TAIL] {o} , [REL] {r} ."

    def encode_output_y(self, sent: RelationSentence, rel) -> str:
        s, r, o = sent.as_tuple()
        return f"[HEAD] {s} , [TAIL] {o} , [REL] {rel} ."

    def decode_y(self, x: str, y: str, label_id: str = None) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(", [REL] ")
        # label = label.strip()
        label = label.strip()[:-2]
        # ipdb.set_trace()
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return RelationSentence.from_spans_rel(text = context, head = head.strip(), tail = tail.strip(), label = label.strip(), label_id = label_id.strip())

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"[HEAD] {head} , [TAIL] {tail} , [REL] "

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def encode_rel(self, sent: RelationSentence, rel) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_output_y(sent, rel)
        return y

    def decode(self, x: str, y: str, label_id: str = None) -> RelationSentence:
        # ipdb.set_trace()
        return self.decode_y(x, y, label_id)

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return encode_to_line(x, y)

    def encode_output_rel(self, sent: RelationSentence, rel_name) -> str:
        return self.encode_rel(sent, rel_name)
       

    def decode_from_line(self, line: str) -> RelationSentence:
        # ipdb.set_trace()
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return decode_from_line(line)

    def safe_decode(self, x: str, y: str, label_id: str = None) -> RelationSentence:
        # ipdb.set_trace()
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y, label_id=label_id)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def safe_decode_raw(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode_raw(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def decode_raw(self, x: str, y: str,) -> RelationSentence:
        # ipdb.set_trace()
        return self.decode_y_raw(x, y)
    
    def decode_y_raw(self, x: str, y: str,) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(", [REL] ")
        # label = label.strip()
        label = label.strip()[:-2]
        # ipdb.set_trace()
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return RelationSentence.from_spans(text = context, head = head.strip(), tail = tail.strip(), label = label.strip(),)

    def decode_to_relationprompt(self, y: str,):
        front, label = y.split(", [REL] ")
        # label = label.strip()
        label = label.strip()[:-2]
        # ipdb.set_trace()
        front, tail = front.split(", [TAIL] ")
        _, head = front.split("[HEAD] ")
        return head.strip(), tail.strip(), label.strip()