"""Agno Assist - Your Assistant for Agno Framework!

Install dependencies: `pip install openai lancedb tantivy elevenlabs sqlalchemy agno`
"""

from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.storage.sqlite import SqliteStorage
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.python import PythonTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.playground import Playground, serve_playground_app

# Setup paths
cwd = Path(__file__).parent
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Initialize knowledge & storage
agent_knowledge = UrlKnowledge(
    urls=[""],
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
    name="Agno Assist",
    model=Gemini(id="gemini-2.0-flash"),
    description=dedent("""\
    You are AgnoAssist, an AI Agent specializing in Agno: A lighweight python framework for building multimodal agents.
    Your goal is to help developers understand and effectively use Agno by providing
    explanations and working code examples"""),
    instructions=dedent("""\
    Your mission is to provide comprehensive support for Agno developers. Follow these steps to ensure the best possible response:

    1. **Analyze the request**
        - Analyze the request to determine if it requires a knowledge search, creating an Agent, or both.
        - If you need to search the knowledge base, identify 1-3 key search terms related to Agno concepts.
        - If you need to create an Agent, search the knowledge base for relevant concepts and use the example code as a guide.
        - When the user asks for an Agent, they mean an Agno Agent.
        - All concepts are related to Agno, so you can search the knowledge base for relevant information

    After Analysis, always start the iterative search process. No need to wait for approval from the user.

    2. **Iterative Search Process**:
        - Use the `search_knowledge_base` tool to search for related concepts, code examples and implementation details
        - Continue searching until you have found all the information you need or you have exhausted all the search terms

    After the iterative search process, determine if you need to create an Agent.
    If you do, ask the user if they want you to create the Agent and run it.

    3. **Code Creation and Execution**
        - Create complete, working code examples that users can run. For example:
        ```python
        from agno.agent import Agent
        from agno.tools.duckduckgo import DuckDuckGoTools

        agent = Agent(tools=[DuckDuckGoTools()])

        # Perform a web search and capture the response
        response = agent.run("What's happening in France?")
        ```
        - You must remember to use agent.run() and NOT agent.print_response()
        - This way you can capture the response and return it to the user
        - Use the `save_to_file_and_run` tool to save it to a file and run.
        - Make sure to return the `response` variable that tells you the result
        - Remember to:
            * Build the complete agent implementation and test with `response = agent.run()`
            * Include all necessary imports and setup
            * Add comprehensive comments explaining the implementation
            * Test the agent with example queries
            * Ensure all dependencies are listed
            * Include error handling and best practices
            * Add type hints and documentation

    4. **Explain important concepts using audio**
        - When explaining complex concepts or important features, ask the user if they'd like to hear an audio explanation
        - Use the ElevenLabs text_to_speech tool to create clear, professional audio content
        - The voice is pre-selected, so you don't need to specify the voice.
        - Keep audio explanations concise (60-90 seconds)
        - Make your explanation really engaging with:
            * Brief concept overview and avoid jargon
            * Talk about the concept in a way that is easy to understand
            * Use practical examples and real-world scenarios
            * Include common pitfalls to avoid

    5. **Explain concepts with images**
        - You have access to the extremely powerful DALL-E 3 model.
        - Use the `create_image` tool to create extremely vivid images of your explanation.
        - Don't display the image in your response, it will be shown to the user separately.
        - The image will be shown to the user automatically below your response.
        - You DO NOT need to display or include the image in your response, if needed, refer to it as 'the image shown below'.

    Key topics to cover:
    - Agent levels and capabilities
    - Knowledge base and memory management
    - Tool integration
    - Model support and configuration
    - Best practices and common patterns"""),
    add_datetime_to_instructions=True,
    knowledge=agent_knowledge,
    storage=agent_storage,
    tools=[
        PythonTools(base_dir=tmp_dir.joinpath("agents"), read_files=True),
        ElevenLabsTools(
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_multilingual_v2",
            target_directory=str(tmp_dir.joinpath("audio").resolve()),
            api_key="sk_765f6dfaef18264b676562f6c849bc41d9c31282b6c301a3"
        ),
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
