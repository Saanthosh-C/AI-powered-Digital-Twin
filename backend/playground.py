"""Agno Assist - Your Assistant for Agno Framework!

Install dependencies: `pip install google-genai lancedb numpy pandas todoist-api-python tantivy elevenlabs sqlalchemy agno`
"""

from pathlib import Path
from textwrap import dedent
from fastapi import FastAPI
from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.storage.sqlite import SqliteStorage
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.python import PythonTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.playground import Playground, serve_playground_app
from agno.tools.todoist import TodoistTools
from pydantic import BaseModel
from typing import List
import json


class FileUploadRequest(BaseModel):
    file_url: str

cwd = Path(__file__).parent
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

urlfile_Path = Path("datas/urls.txt")
urlfile_Path.parent.mkdir(parents=True, exist_ok=True)


if not urlfile_Path.exists():
    urlfile_Path.write_text("")

with urlfile_Path.open("r") as f:
    urls = [line.strip() for line in f.readlines() if line.strip()]


# Initialize knowledge & storage
agent_knowledge = UrlKnowledge(

        urls=urls,
    vector_db=LanceDb(
        uri=str(tmp_dir.joinpath("lancedb")),
        table_name="agno_assist_knowledge",
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(id="gemini-embedding-exp-03-07",api_key="AIzaSyCleH8Tjoza7TAMhQsxz-t8dj_jskc7nMw"),
    ),
)
agent_storage = SqliteStorage(
    table_name="agno_assist_sessions",
    db_file=str(tmp_dir.joinpath("agent_sessions.db")),
)

agno_assist = Agent(
    name="Digital Twin",
    model=Gemini(id="gemini-2.0-flash",api_key="AIzaSyCleH8Tjoza7TAMhQsxz-t8dj_jskc7nMw"),
    description=dedent("""\
       You are an AI-powered Digital Twin that mirrors a student's academic behavior, learning patterns, 
       and performance. Your purpose is to analyze historical data, predict future performance, 
       and provide personalized learning recommendations to optimize educational outcomes."""),
    instructions=dedent("""\
       As a Student Digital Twin, your mission is to provide comprehensive academic support and personalization. 
       Follow these steps for optimal performance:

       1. **Student Profile Analysis**
           - Collect and analyze the student's academic history, learning preferences, and performance data
           - Identify patterns in study habits, strengths, and areas needing improvement
           - Maintain continuous updates to the student's digital profile

       2. **Predictive Modeling**
           - Forecast academic performance based on current trends
           - Simulate different study scenarios and their potential outcomes
           - Identify at-risk subjects or assignments early

       3. **Personalized Recommendations**
           - Suggest optimized study schedules based on circadian rhythms and past performance
           - Recommend learning resources tailored to the student's preferred modalities
           - Provide targeted exercises for weak areas while reinforcing strengths

       4. **Real-time Adaptation**
           - Adjust recommendations based on ongoing performance data
           - Detect changes in study effectiveness and suggest modifications
           - Provide motivational support based on the student's psychological profile

       5. **Visualization and Reporting**
           - Generate clear visual representations of progress and predictions
           - Create comparative analyses showing potential improvement paths
           - Provide audio explanations of complex concepts when requested

       Implementation Guidelines:
       - Always maintain privacy and data security for student information
       - Use positive reinforcement in all communications
       - Provide options for different learning styles (visual, auditory, kinesthetic)
       - Include stress and workload management recommendations
       - Offer both micro (daily) and macro (semester-long) perspectives

       Technical Requirements:
       - Store all student data securely with proper encryption
       - Create modular components that can adapt to different academic subjects
       - Include API connections to common learning management systems
       - Implement robust error handling for data inconsistencies"""),
    add_datetime_to_instructions=True,
    knowledge=agent_knowledge,
    show_tool_calls=True,
    storage=agent_storage,
    tools=[
        PythonTools(base_dir=tmp_dir.joinpath("agents"), read_files=True),
        ElevenLabsTools(
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_multilingual_v2",
            target_directory=str(tmp_dir.joinpath("audio").resolve()),
            api_key="sk_765f6dfaef18264b676562f6c849bc41d9c31282b6c301a3"
        ),
        TodoistTools(api_token="7304b56106210a362687667b12f4b47997d72983")
    ],
    # To provide the agent with the chat history
    # We can either:
    # 1. Provide the agent with a tool to read the chat history
    # 2. Automatically add the chat history to the messages sent to the model
    #
    # 1. Provide the agent with a tool to read the chat history
    read_chat_history=True,
    # 2. Automatically add the chat history to the messages sent to the model
    add_history_to_messages=True,
    # Number of historical responses to add to the messages.
    num_history_responses=3,
    markdown=True,

)



app = Playground(agents=[agno_assist]).get_app()

@app.post("/file")
async def upload_file(data : FileUploadRequest):
    my_txt_path = Path("datas/urls.txt")
    my_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with my_txt_path.open("a", encoding="utf-8") as f:
        f.write(data.file_url + "\n")
    print("Received file URL:", data.file_url)

    try:
        agent_knowledge.add_url(data.file_url)
        print("Successfully added to knowledge base.")
        return {"message": "File URL received and added to knowledge base."}
    except Exception as e:
        print(f"Error adding URL to knowledge base: {e}")
        return {"message": "File URL received but failed to add to knowledge base."}


if __name__ == "__main__":
    # Set to False after the knowledge base is loaded
    load_knowledge = True
    if load_knowledge:
        agent_knowledge.load()

    serve_playground_app("playground:app", reload=True)
