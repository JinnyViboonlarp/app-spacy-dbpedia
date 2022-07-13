import spacy
import spacy_dbpedia_spotlight

# testing dbpedia_spotlight wrapped in spaCy

if __name__ == "__main__":
    
    # load your model as usual
    nlp = spacy.load('en_core_web_sm')
    # add the pipeline stage
    nlp.add_pipe('dbpedia_spotlight')
    # get the document
    doc = nlp('Hello, this is Jim Lehrer with the NewsHour on PBS. In the nineteen eighties, barking dogs have increasingly become a problem in urban areas.')
    # see the entities
    print('Entities', [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])
    # output: Entities [('Jim Lehrer', 'DBPEDIA_ENT', 'http://dbpedia.org/resource/Jim_Lehrer'), ('PBS', 'DBPEDIA_ENT', 'http://dbpedia.org/resource/PBS')]
    
    # inspect the raw data from DBpedia spotlight
    print(doc.ents[0]._.dbpedia_raw_result)
    """ output: {'@URI': 'http://dbpedia.org/resource/Jim_Lehrer', '@support': '197',
'@types': 'Http://xmlns.com/foaf/0.1/Person,Wikidata:Q5,Wikidata:Q24229398,Wikidata:Q215627,DUL:NaturalPerson,DUL:Agent,Schema:Person,DBpedia:Agent,DBpedia:Person',
'@surfaceForm': 'Jim Lehrer', '@offset': '15', '@similarityScore': '1.0', '@percentageOfSecondRank': '0.0'} """
