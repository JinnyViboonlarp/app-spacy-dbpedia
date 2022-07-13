# app-spacy-dbpedia

### Usage

testing cased mmif input file

```
$ python app.py -t example-mmif.json out.json
```

testing uncased mmif input file (with the uncased NER model preprocessing)

```
$ python app.py -t -u example-mmif-uncased.json out-uncased.json
```
