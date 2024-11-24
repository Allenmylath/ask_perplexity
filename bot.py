import asyncio
import os
import sys
from loguru import logger
from dotenv import load_dotenv
import requests

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import StartFrame, EndFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService, OpenAILLMContextFrame
from websocket_server import WebsocketServerParams, WebsocketServerTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from noisereduce_filter import NoisereduceFilter

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

def ask_perplexity(question: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": question}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]
    
async def handle_perplexity(function_name, tool_call_id, args, llm, context, result_callback):
    response = ask_perplexity(args["question"])
    await result_callback([{"role": "assistant", "content": response}])

async def main():
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="",
            port=int(os.environ["PORT"]),
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_in_filter=NoisereduceFilter(),
        )
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",  # British Lady
    )

    messages = []
    context = OpenAILLMContext(messages=messages)
    context.add_message({
        "role": "system",
        "content": "You are a helpful voice assistant. When a user asks a question, use the ask_perplexity function to get the answer."
    })
    context.set_tools([
        {
            "type": "function",
            "function": {
                "name": "ask_perplexity",
                "description": "Get an answer from Perplexity AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question"
                        }
                    },
                    "required": ["question"]
                }
            }
        }
    ])

    context_aggregator = llm.create_context_aggregator(context)
    llm.register_function("ask_perplexity", handle_perplexity)

    fl = FrameLogger("LLM Output")

    pipeline = Pipeline([
        transport.input(),  # WebSocket input
        stt,  # Speech-To-Text
        context_aggregator.user(),  # User responses
        llm,  # LLM
        fl,  # Frame logger
        tts,  # Text-To-Speech
        transport.output(),  # WebSocket output
        context_aggregator.assistant(),  # Assistant responses
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.messages.clear()
        context_aggregator = llm.create_context_aggregator(context)
        print("New client connected")
        await task.queue_frames([OpenAILLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
