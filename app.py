"""app.py

Wrapping Spacy NLP to extract tokens, tags, lemmas, sentences, chunks, named
entities, and (if specified so) syntactic relations.

Usage:

$ python app.py -t example-mmif.json out.json
$ python app.py [--develop]

The first invocation is to just test the app without running a server. The
second is to start a server, which you can ping with

$ curl -H "Accept: application/json" -X POST -d@example-mmif.json http://0.0.0.0:5000/

With the --develop option you get a FLask server running in development mode,
without it Gunicorn will be used for a more stable server.

Normally you would run this in a Docker container, see README.md.

"""

import os
import sys
import collections
import json
import urllib
import argparse

import spacy
import spacy_dbpedia_spotlight

from clams.app import ClamsApp
from clams.restify import Restifier
from clams.appmetadata import AppMetadata
from mmif.serialize import Mmif
from mmif.vocabulary import AnnotationTypes, DocumentTypes
from lapps.discriminators import Uri

# Load small English core model
nlp = spacy.load("en_core_web_sm")
# add the dbpedia_spotlight pipeline stage
nlp.add_pipe('dbpedia_spotlight')

# Jinny: Define NER model that would be loaded if args.uncased == True
ner = None

APP_VERSION = '0.0.8'
APP_LICENSE = 'Apache 2.0'
MMIF_VERSION = '0.4.0'
MMIF_PYTHON_VERSION = '0.4.6'
CLAMS_PYTHON_VERSION = '0.5.1'
SPACY_VERSION = '3.1.2'
SPACY_LICENSE = 'MIT'


# We need this to find the text documents in the documents list
TEXT_DOCUMENT = os.path.basename(str(DocumentTypes.TextDocument))

DEBUG = False
dep_choice = False # choice to use dependency parser


class SpacyApp(ClamsApp):

    def _appmetadata(self):
        metadata = AppMetadata(
            identifier='https://apps.clams.ai/spacy_nlp',
            url='https://github.com/clamsproject/app-spacy-nlp',
            name="Spacy NLP",
            description="Apply spaCy NLP to all text documents in a MMIF file.",
            app_version=APP_VERSION,
            app_license=APP_LICENSE,
            analyzer_version=SPACY_VERSION,
            analyzer_license=SPACY_LICENSE,
            mmif_version=MMIF_VERSION
        )
        metadata.add_input(DocumentTypes.TextDocument)
        metadata.add_output(Uri.TOKEN)
        metadata.add_output(Uri.POS)
        metadata.add_output(Uri.LEMMA)
        metadata.add_output(Uri.NCHUNK)
        metadata.add_output(Uri.SENTENCE)
        metadata.add_output(Uri.NE)
        metadata.add_output(Uri.DEPENDENCY)
        metadata.add_output(Uri.GENERIC_RELATION)
        return metadata

    def _annotate(self, mmif, **kwargs):
        Identifiers.reset()
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)
        for doc in text_documents(self.mmif.documents):
            new_view = self._new_view(doc.id)
            self._add_tool_output(doc, new_view)
        for view in list(self.mmif.views):
            docs = self.mmif.get_documents_in_view(view.id)
            if docs:
                new_view = self._new_view()
                for doc in docs:
                    doc_id = view.id + ':' + doc.id
                    self._add_tool_output(doc, new_view, doc_id=doc_id)
        return self.mmif

    def _new_view(self, docid=None):
        view = self.mmif.new_view()
        self.sign_view(view)
        for attype in (Uri.TOKEN, Uri.POS, Uri.LEMMA,
                       Uri.NCHUNK, Uri.SENTENCE, Uri.NE,
                       Uri.DEPENDENCY, Uri.GENERIC_RELATION):
            view.new_contain(attype, document=docid)
        return view

    def _read_text(self, textdoc):
        """Read the text content from the document or the text value."""
        if textdoc.location:
            fh = urllib.request.urlopen(textdoc.location)
            text = fh.read().decode('utf8')
        else:
            text = textdoc.properties.text.value
        if DEBUG:
            print('>>> %s%s' % (text.strip()[:100],
                                ('...' if len(text) > 100 else '')))
        return text

    def _add_tool_output(self, doc, view, doc_id=None):

        text_orig = self._read_text(doc)
        input_text = text_orig
        # Jinny: the off-the-shelf spaCy model does not recognize 'jim lehrer' (lowercased) as a person
        # but the spaCy NER model trained on CoNLL does. If the user specifies to use the uncased model,
        # that model would be used to "truecase" its recognized named entities
        if(ner != None):
            input_text_list = list(input_text.lower()) # python string is immutable, but we want to modify input_text
            spacy_doc_ner = ner(input_text.lower())
            # keep track of char offsets of all tokens
            tok_idx_ner = {}
            for (n, tok) in enumerate(spacy_doc_ner):
                if(tok.ent_type_ != ""):
                    start = tok.idx
                    input_text_list[start] = input_text_list[start].upper()
            input_text = "".join(input_text_list)
            print(input_text)

        spacy_doc = nlp(input_text)
        
        # Jinny, if the uncased NER model is used, the "truecased" sentence will be uncased back before annotated
        def case_or_not(text):
            return text.lower() if (ner != None) else text
        
        # keep track of char offsets of all tokens
        tok_idx = {}
        for (n, tok) in enumerate(spacy_doc):
            p1 = tok.idx
            p2 = p1 + len(tok.text)
            tok_idx[n] = (p1, p2)
            add_annotation(
                view, Uri.TOKEN, Identifiers.new("t"),
                doc_id, p1, p2,
                { "pos": tok.tag_, "lemma": tok.lemma_, "text": text_orig[p1:p2] , "i": tok.i})
        for (n, chunk) in enumerate(spacy_doc.noun_chunks):
            p1 = tok_idx[chunk.start][0]; p2 = tok_idx[chunk.end - 1][1]
            add_annotation(
                view, Uri.NCHUNK, Identifiers.new("nc"),
                doc_id, p1, p2, { "text": text_orig[p1:p2] })     
        for (n, sent) in enumerate(spacy_doc.sents):
            p1 = tok_idx[sent.start][0]; p2 = tok_idx[sent.end - 1][1]
            add_annotation(
                view, Uri.SENTENCE, Identifiers.new("s"),
                doc_id, p1, p2, { "text": text_orig[p1:p2] })

        def find_dbpedia_type(ent):
            prefix = 'DBpedia:'
            interested_types = ['Person','Place','Organisation','Device']
            try:
                types_list = ent._.dbpedia_raw_result['@types'].split(',')
                for category in interested_types:
                    if (prefix+category) in ent._.dbpedia_raw_result['@types']:
                        return category
            except:
                return None
            return None

        # Now we're going to iterate over named entities
        # I am going to also keep a dictionary that store the entities' properties (with their root_i as keys)
        # this dictionary will be used for relation extraction later
        entity_dict = {}
        for (n, ent) in enumerate(spacy_doc.ents):
            category = find_dbpedia_type(ent)
            if(category != None):
                p1 = tok_idx[ent.start][0]; p2 = tok_idx[ent.end - 1][1]
                p1_root = ent.root.idx; p2_root = (p1_root + len(ent.root.text))
                properties = { "text": text_orig[p1:p2], "category": category, \
                                "root_i": ent.root.i, "root_text": text_orig[p1_root:p2_root], "kb_id": ent.kb_id_}
                entity_dict[ent.root.i] = properties
                add_annotation(
                    view, Uri.NE, Identifiers.new("ne"), doc_id, p1, p2, properties)   

        """
        dependency_identifiers = [Identifiers.new("dep") for n in range(len(spacy_doc))]
        # for example, dependency_identifiers would look like [ "dep0", "dep1", "dep2" ]
        """

        # Now we're going to iterate over dependencies
        # I am going to also keep a dictionary that store the token's governer (with the dependent token's root_i as keys)
        # In an SVO sentence, S and O is going to be dependent on V. For example, "I love you" will have the tokens "I" and "you" depends on "love"
        governer_dict = {}
        for (n, tok) in enumerate(spacy_doc):
            p1_dep = tok.idx; p2_dep = (p1_dep + len(tok.text))
            p1_gov = tok.head.idx; p2_gov = (p1_gov + len(tok.head.text))
            properties_dependent = { "dependent_text": text_orig[p1_dep:p2_dep], "dependent_lemma": tok.lemma_, "dependent_i": tok.i}
            properties_governer = {"dep": tok.dep_, "governer_text": text_orig[p1_gov:p2_gov], "governer_lemma": tok.head.lemma_, "governer_i": tok.head.i}
            properties = {**properties_dependent, **properties_governer} # merge two dicts
            if(tok.i in entity_dict):
                governer_dict[tok.i] = properties_governer
            add_annotation(
                view, Uri.DEPENDENCY, Identifiers.new("dep"),
                doc_id, None, None, properties)

        # use the two dicts to find relations between named entities
        for i1 in entity_dict:
            for i2 in entity_dict:
                if ((i1 < i2) and governer_dict[i1]["governer_i"] == governer_dict[i2]["governer_i"]):
                        properties = {  "rel_text": governer_dict[i1]['governer_text'], "rel_lemma": governer_dict[i1]['governer_lemma'], \
                                        "rel_i": governer_dict[i1]['governer_i'], \
                                        "e1_text": entity_dict[i1]['text'], "e1_root_i": entity_dict[i1]['root_i'], \
                                        "e1_kb_id": entity_dict[i1]['kb_id'], "e1_dep": governer_dict[i1]['dep'], \
                                        "e2_text": entity_dict[i2]['text'], "e2_root_i": entity_dict[i2]['root_i'], \
                                        "e2_kb_id": entity_dict[i2]['kb_id'], "e2_dep": governer_dict[i2]['dep'] }
                        add_annotation(
                            view, Uri.GENERIC_RELATION, Identifiers.new("rel"),
                            doc_id, None, None, properties)         


    def print_documents(self):
        for doc in self.mmif.documents:
            print("%s %s location=%s text=%s" % (
                doc.id, doc.at_type, doc.location, doc.properties.text.value))


def text_documents(documents):
    """Utility method to get all text documents from a list of documents."""
    return [doc for doc in documents if str(doc.at_type).endswith(TEXT_DOCUMENT)]
    # TODO: replace with the following line and remove TEXT_DOCUMENT variable
    # when mmif-python is updated
    # return [doc for doc in documents if doc.is_type(DocumentTypes.TextDocument)]


def add_annotation(view, attype, identifier, doc_id, start, end, properties):
    """Utility method to add an annotation to a view."""
    a = view.new_annotation(attype, identifier)
    if doc_id is not None:
        a.add_property('document', doc_id)
    if start is not None:
        a.add_property('start', start)
    if end is not None:
        a.add_property('end', end)
    for prop, val in properties.items():
        a.add_property(prop, val)

class Identifiers(object):

    """Utility class to generate annotation identifiers. You could, but don't have
    to, reset this each time you start a new view. This works only for new views
    since it does not check for identifiers of annotations already in the list
    of annotations."""

    identifiers = collections.defaultdict(int)

    @classmethod
    def new(cls, prefix):
        cls.identifiers[prefix] += 1
        return "%s%d" % (prefix, cls.identifiers[prefix])

    @classmethod
    def reset(cls):
        cls.identifiers = collections.defaultdict(int)



def test(infile, outfile):
    """Run spacy on an input MMIF file. This bypasses the server and just pings
    the annotate() method on the SpacyApp class. Prints a summary of the views
    in the end result."""
    print(SpacyApp().appmetadata(pretty=True))
    with open(infile) as fh_in, open(outfile, 'w') as fh_out:
        mmif_out_as_string = SpacyApp().annotate(fh_in.read(), pretty=True)
        mmif_out = Mmif(mmif_out_as_string)
        fh_out.write(mmif_out_as_string)
        for view in mmif_out.views:
            print("<View id=%s annotations=%s app=%s>"
                  % (view.id, len(view.annotations), view.metadata['app']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',  action='store_true', help="bypass the server")
    parser.add_argument('--develop',  action='store_true', help="start a development server")
    parser.add_argument('-u', '--uncased',  action='store_true', help="select the NER model trained for uncased data")
    parser.add_argument('infile', nargs='?', help="input MMIF file")
    parser.add_argument('outfile', nargs='?', help="output file")
    args = parser.parse_args()

    if args.uncased:
        ner = spacy.load("ner_models/model-best-uncased-sm")

    if args.test:
        test(args.infile, args.outfile)
    else:
        app = SpacyApp()
        service = Restifier(app)
        if args.develop:
            service.run()
        else:
            service.serve_production()
