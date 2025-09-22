# --- Utility Functions ---
def call_llm(prompt):
    from ollama import chat
    response = chat(
            model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content


def extract_json_from_codeblock(text: str):
    import re
    import json

    json_codeblock_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = json_codeblock_pattern.search(text)

    if match:
        json_content_str = match.group(1)
        # Extracts and loads without error-handling
        # This is meant to be handled by PocketFlow retry logic
        return json.loads(json_content_str)
    else:
        return None

def semantic_chunking(sentences):
    import json
    chunk_number = 0
    prompt = """You are a SEMANTIC BOUNDARY DETECTOR.
                When given to strings, your ONLY response is either:
                true: These two strings together form ONE coherent thought 
                false: These two strings represent TWO distinct ideas.

                You will be given a group of sentences (might be just one) along
                with one new sentence.

                You will respond True, if the new string belongs semantically with the rest.
                And False otherwise.

                Your output will follow a JSON format as follows:
                
                ```json 
                {
                    "response": true/false,
                    "explanation": The why you think these two strings do or don't belong together
                }
                ```

                Example 1:

                { "curr_chunk": "I like cheese on pizza. I like dairy in general.",
                  "new_sentence": "I do not like people"}

                Your response:
                {
                    "response": false,
                    "explanation": "The first chunk talks about the user's general love for dairy, while the new sentence is about their general distaste of people."
                }

                Example 2:
                { "curr_chunk": "I am a man of big ideals. I want to conquer the earth. I want to discover new horizons.",
                  "new_sentence": "I want to have a beautiful wife who shares my ideals."}

                Your response:
                {
                    "response": true,
                    "explanation": "The curr_chunk reflects the general ambitions of a man, while the new_sentence is about another ambition of theirs."
                }
                """
    if not sentences:
        raise ValueError("Passed empty list to semantic_chunking")
    
    cur_chunk = sentences[0]
    
    if len(sentences) == 1:
        return ([(cur_chunk,1)], 1)

    chunks = []

    for (i, sentence) in enumerate(sentences[1:]):
        response = call_llm(prompt + "\n\n" + json.dumps({"curr_chunk" : cur_chunk, "new_sentence": sentence}))

        # We leave this to the retry logic 
        # (This might be heavy for the retry though)
        # I'll potentially benchmark this later
        evaluation = extract_json_from_codeblock(response)
        # If the LLM is a prick, we'll just assume we should split 
        should_add_to_chunk = evaluation.get("response", False) if evaluation else False
            
        if should_add_to_chunk:
            cur_chunk += "\n" + sentence
        else:
            chunk_number += 1
            chunks.append((cur_chunk, chunk_number))
            
            # start the next chunk
            cur_chunk = sentence
    
    # Cleanup (Adding the last chunk)
    if cur_chunk:
        chunk_number += 1
        chunks.append((cur_chunk, chunk_number))

    # Return the chunks and the number of "ideas" covered
    return (chunks, chunk_number)


