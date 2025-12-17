
def build_prompt(context_chunk,query):
    context = "\n\n".join(context_chunk)
    return f"""use the following context to answer the question.
    context : {context}
    question : {query}
    Answer : """