import chromadb
import asyncio
from transformers import pipeline
import chardet
#chroma_client = chromadb.HttpClient(host='localhost', port=8000)

BATCH_SIZE = 5000

def chunk_text(text, chunk_size=1000, overlap=100):

    """
    Splits text into chunks of chunk_size with optional overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
def get_top_chunks(results, top_k=3):
    # Get the top_k most relevant chunks (lowest distances)
    docs = results['documents'][0]
    distances = results['distances'][0]
    # Pair and sort by distance
    sorted_chunks = sorted(zip(docs, distances), key=lambda x: x[1])
    # Return the top_k chunks as a single string
    return "\n".join(chunk for chunk, _ in sorted_chunks[:top_k])

async def main():
    chroma_client = await chromadb.AsyncHttpClient()
    # with open("C:/Users/RZbas/Projects/ChatBot/Renezachatbot/FaqFromSources.txt", "r", encoding="UTF-8") as f:
    #     file_content = f.read()
    # with open("C:/Users/RZbas/Projects/ChatBot/Renezachatbot/FaqFromSources.txt", "rb") as f:
    #         raw_data = f.read()
    #         detected = chardet.detect(raw_data)
    #         encoding = detected['encoding']
    #         print("Detected encoding:", encoding)
    #         file_content = raw_data.decode(encoding)
    import re

    def load_clean_utf8_file(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Remove non-printable/control characters
        cleaned_text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
        return cleaned_text

    file_content = load_clean_utf8_file("C:/Users/RZbas/Projects/ChatBot/Renezachatbot/FaqFromSources.txt")
    print("CLEANED PREVIEW:\n", file_content[:500])
        
    chunks = chunk_text(file_content, chunk_size=1000, overlap=100)
    ids = [f"id{i+1}" for i in range(len(chunks))]
    print("=== Chunk Samples to Be Stored in ChromaDB ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"[Chunk {i}]:\n{chunk}\n{'-'*40}")
    await chroma_client.delete_collection(name="poslovniModeli")    
    collection = await chroma_client.get_or_create_collection(name="poslovniModeli")
    for i in range(0, len(chunks), BATCH_SIZE):
        await collection.add(
            ids=ids[i:i+BATCH_SIZE],
            documents=chunks[i:i+BATCH_SIZE],
        )
    results = await collection.query(
    query_texts=["This is a query document about designs"], # Chroma will embed this for you
    )
    # for i, query_results in enumerate(results['documents']):
    #     print(f"Query {i}:")

    
    context = get_top_chunks(results, top_k=3)
    print("Context for ML model:\n", context)

    # Example: Use Hugging Face pipeline for text generation
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Based on the following context, answer the question: What is design thinking?\n\n{context}\nAnswer:"
    output = generator(prompt, max_length=200)
    print("ML Model Output:\n", output[0]['generated_text'])

asyncio.run(main())

#collection = chroma_client.create_collection(name="poslovniModeli")

#with open("C:/Users/RZbas/Projects/ChatBot/Renezachatbot/DelftDesignGuide-2010.txt", "r", encoding="latin-1") as f:
#    file_content = f.read()

# collection.add(
#     ids=["id1"],
#     documents=[file_content],  # This is the text content you want to add
# )

# results = collection.query(
#     query_texts=["This is a query document about designs"], # Chroma will embed this for you
# )
# for i, query_results in enumerate(results['documents']):
#     print(f"Query {i}:")
# print(results)