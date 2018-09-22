import json

from text import TextDataSource


class WikidumpDataSource(TextDataSource):
    def __iter__(self):
        for f in self.files:
            with open(f, "r") as fin:
                for line in fin:
                    if line:
                        yield json.loads(line).get('text')
