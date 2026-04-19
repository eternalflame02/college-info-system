from chatbot import answer_question

queries = [
    "What are the electives for s7ct and explain syllabus in detail?",
    "I'm not a fan of theory subjects so which one should I pick from these electives?",
    "What type of subject is Artificial Intelligence? Theory based or application?",
    "What are common subjects for cs and ct in S6?",
    "I'm planning to move into a ui/ux field so what minor will be apt for that?",
]

for i, query in enumerate(queries, 1):
    print("\n" + "=" * 120)
    print(f"Q{i}: {query}")
    print("-" * 120)
    try:
        response = answer_question(query)
    except Exception as exc:
        print(f"ERROR: {exc}")
        continue

    print(f"query_type={response.query_type} quality={response.quality} response_time_ms={response.response_time_ms}")
    print(f"source_count={len(response.source_chunks)}")

    for rank, chunk in enumerate(response.source_chunks[:3], 1):
        print(
            f"source_{rank}: type={chunk.content_type} distance={chunk.distance:.3f} "
            f"file={chunk.source_file}"
        )

    print("ANSWER_START")
    print((response.answer or "").strip())
    print("ANSWER_END")
