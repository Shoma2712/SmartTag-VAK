from backend.stages.lda.lda_modeling import LDATopicModeler, run_lda_modeling

__all__ = ["LDATopicModeler", "run_lda_modeling", "get_document_topics"]


def get_document_topics(lda_model, corpus):
    """Доминирующая LDA-тема для каждого документа в корпусе."""
    topics = []
    for doc_bow in corpus:
        dist = lda_model.get_document_topics(doc_bow)
        if dist:
            topics.append(max(dist, key=lambda x: x[1])[0])
        else:
            topics.append(-1)
    return topics
