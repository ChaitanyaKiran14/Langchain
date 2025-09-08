from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

video_id = "106DaA8WdHg" 
ytt_api = YouTubeTranscriptApi()
vals = ytt_api.fetch(video_id)
full_transcript = " ".join([snippet.text for snippet in vals])

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1000,
    chunk_overlap=150,
)
chunks = splitter.create_documents([full_transcript])
print(chunks)     
print(len(chunks))