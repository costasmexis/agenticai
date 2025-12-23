from agents import Agent, Runner, trace
from agents.extensions.models.litellm_model import LitellmModel

ollama_model = LitellmModel(model="ollama/llama3.2:3b", api_key="ollama")

agent = Agent(name="Jokester", instructions="You are a joke teller", model=ollama_model)

with trace("joke_workflow", disabled=True):
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)