from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, output_guardrail, OutputGuardrailTripwireTriggered, function_tool, RunContextWrapper
import asyncio
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from dataclasses import dataclass


# Load environment variables from .env file
load_dotenv()

@dataclass
class User:
    id: int
    name: str
    age: int


class MathHomeWorkOutput(BaseModel):
    is_math_work: bool
    reasoning: str

class PhysicsHomeWorkOutput(BaseModel):
    is_physics_work: bool
    reasoning: str

class MainMessageOutput(BaseModel):
    response: str

async def main():
    MODEL_NAME = "gemini-2.0-flash"
    API_KEY = os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    external_client = AsyncOpenAI(
        api_key=API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    inputGuardrailAgent = Agent(
        name="input guardrail agent",
        instructions="you have to check either the user's query is related to math or not.",
        output_type=MathHomeWorkOutput,
        model=model
    )
    outputGuardrailAgent = Agent(
        name="output guardrail agent",
        instructions="you have to check either the user's query is related to physics or not.",
        output_type=PhysicsHomeWorkOutput,
        model=model
    )

    @input_guardrail
    async def math_guardrail(ctx, agent, input):
        # print("Prompt: " + " " + input)
        print(ctx.context)
        result = await Runner.run(inputGuardrailAgent, input)

        # print("Guard rails output: ", result.final_output)

        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.is_math_work
        )
    @output_guardrail
    async def physics_guardrail(ctx, agent, output):
        # print("Prompt: " + " " + input)
        result = await Runner.run(outputGuardrailAgent, output.response)

        # print("Guard rails output: ", result.final_output)

        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.is_physics_work
        )

    @function_tool
    def get_user_age(wrapper: RunContextWrapper[User]):
        """Return user's age"""
        print("get_user_age Tool called")
        return f"{wrapper.context.name} is {wrapper.context.age} years old"


    customer_support_agent = Agent(
        name="customer support agent",
        instructions = """
            Your task is to get the user's data asked in query for that you can call the appropriate tool for that
        """,
        model = model,
        output_type=MainMessageOutput,
        input_guardrails=[math_guardrail],
        output_guardrails=[physics_guardrail],
        tools=[get_user_age]
    )

    try:
        user = User(123, "Hoorain", 40)
        result =  await Runner.run(starting_agent=customer_support_agent, input="what is user's age?", context=user)
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print("this is math work")
    except OutputGuardrailTripwireTriggered:
        print("this is physics work")

if __name__ == "__main__":
    asyncio.run(main())
