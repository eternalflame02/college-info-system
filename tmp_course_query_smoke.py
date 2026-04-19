from chatbot import answer_question

queries = [
    ("Q1", "What are the electives for s7ct and explain syllabus in detail?"),
    ("Q2", "I'm not a fan of theory subjects so which one should I pick from these electives?"),
    ("Q3", "What type of subject is Artificial Intelligence? Theory based or application?"),
    ("Q4", "What are common subjects for cs and ct in S6?"),
    ("Q5", "I'm planning to move into a ui/ux field so what minor will be apt for that?"),
]

for key, query in queries:
    response = answer_question(query)
    print("\n" + "=" * 110)
    print(f"{key}: {query}")
    print(f"query_type={response.query_type} quality={response.quality} sources={len(response.source_chunks)}")
    top_sources = [
        (chunk.content_type, chunk.source_file, round(float(chunk.distance), 3))
        for chunk in response.source_chunks[:3]
    ]
    print(f"top_sources={top_sources}")
    preview = (response.answer or "").replace("\n", " ")
    print(f"answer_preview={preview[:600]}")
