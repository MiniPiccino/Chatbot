import chromadb
import asyncio
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

async def main():
    client = await chromadb.AsyncHttpClient()
    collection = await client.create_collection(name="my_collection")
    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )

asyncio.run(main())