"""Agno Assist - Your Assistant for Agno Framework!

Install dependencies: `pip install openai lancedb tantivy elevenlabs sqlalchemy agno`
"""

from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.knowledge.text import TextKnowledgeBase
from agno.storage.sqlite import SqliteStorage
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.python import PythonTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.playground import Playground, serve_playground_app
from agno.tools.todoist import TodoistTools

# Setup paths
cwd = Path(__file__).parent
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Initialize knowledge & storage
agent_knowledge = TextKnowledgeBase(
    path="datas/my.txt",
    vector_db=LanceDb(
        uri=str(tmp_dir.joinpath("lancedb")),
        table_name="agno_assist_knowledge",
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(id="gemini-embedding-exp-03-07"),
    ),
)
agent_storage = SqliteStorage(
    table_name="agno_assist_sessions",
    db_file=str(tmp_dir.joinpath("agent_sessions.db")),
)

agno_assist = Agent(
    name="Digital Twin",
    model=Gemini(id="gemini-2.0-flash"),
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
        ),
        TodoistTools()
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

if __name__ == "__main__":
    # Set to False after the knowledge base is loaded
    load_knowledge = True
    if load_knowledge:
        agent_knowledge.load()

    serve_playground_app("playground:app", reload=True)
