import asyncio
import os
import sys
from datetime import datetime
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

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="DEBUG"
)

# Define system message as a constant
SYSTEM_MESSAGE = """You are a helpful voice assistant. When a user asks a question, use the ask_perplexity function to get the answer.Only use the ask_perpexity function when user asks.No unnecessary prompting.
Remember your output is connected to TTS, so avoid symbols in response. Speak in a conversational way.
To insert breaks (or pauses) in generated speech, use <break time="1s" /> tags. 
To spell out text, wrap it in <spell> tags.
Use appropriate punctuation and add pauses with '-' where needed."""

async def ask_perplexity(question: str) -> str:
    try:
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    logger.error(f"Perplexity API error: {response.status}")
                    return "I apologize, but I encountered an error getting the answer. Please try again."
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error in ask_perplexity: {str(e)}")
        return "I apologize, but I encountered an error getting the answer. Please try again."
    
async def handle_perplexity(function_name, tool_call_id, args, llm, context, result_callback):
    try:
        response = await ask_perplexity(args["question"])
        await result_callback([{"role": "assistant", "content": response}])
    except Exception as e:
        logger.error(f"Error in handle_perplexity: {str(e)}")
        await result_callback([{
            "role": "assistant", 
            "content": "I encountered an error processing your request. Please try again."
        }])

async def create_initial_context():
    context = OpenAILLMContext(messages=[
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        }
    ])
    
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
    
    return context

async def main():
    try:
        port = int(os.getenv("PORT", "8080"))
        
        # Validate required environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "DEEPGRAM_API_KEY",
            "CARTESIA_API_KEY",
            "PERPLEXITY_API_KEY"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        transport = WebsocketServerTransport(
            params=WebsocketServerParams(
                host="0.0.0.0",
                port=port,
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
                audio_in_filter=NoisereduceFilter(),
                max_clients=1
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4"
        )
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",
        )

        # Create initial context
        context = await create_initial_context()
        context_aggregator = llm.create_context_aggregator(context)
        llm.register_function("ask_perplexity", handle_perplexity)

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            fl,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            try:
                logger.info(f"New client connected from {client.remote_address if hasattr(client, 'remote_address') else 'unknown'}")
                context.messages.clear()
                context.add_message({
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                })
                new_context_aggregator = llm.create_context_aggregator(context)
                await task.queue_frames([OpenAILLMContextFrame(context)])
            except Exception as e:
                logger.error(f"Error in on_client_connected: {str(e)}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            try:
                logger.info(f"Client disconnected from {client.remote_address if hasattr(client, 'remote_address') else 'unknown'}")
                await task.queue_frames([EndFrame()])
            except Exception as e:
                logger.error(f"Error in on_client_disconnected: {str(e)}")

        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
