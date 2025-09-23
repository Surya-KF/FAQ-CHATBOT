"""
LLM and embedding integration module for Hospital FAQ Chatbot

This module handles integration with Gemini 2.5 Flash as the base LLM 
and SentenceTransformer (all-MiniLM-L6-v2) for embeddings.
"""

import logging
from typing import List
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiLLMClient:
    """Client for interacting with Gemini 2.5 Flash model."""

    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model_name = getattr(settings, "gemini_model_name", None) or "gemini-2.5-flash"

        if not self.api_key:
            logger.error("GEMINI_API_KEY is not configured.")
            raise ValueError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=self.api_key)
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1,
            max_tokens=2048,
            top_p=0.8,
            top_k=40,
            convert_system_message_to_human=True,
        )
        logger.info(f"Gemini LLM client initialized with model: {self.model_name}")

    def generate_response(self, prompt: str, system_message: str = None) -> str:
        """
        Generate a response from the LLM.
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self.model.invoke(messages)
            logger.info("Successfully generated response from Gemini LLM.")
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate response from Gemini LLM: {e}")
            return "Sorry, I'm having technical difficulties. Please try again later."

    def summarize_conversation(self, conversation_history: List[dict]) -> str:
        """
        Generate a summary of the conversation.
        """
        try:
            prompt = "Summarize the following conversation:\n"
            for msg in conversation_history:
                prompt += f"{msg['role']}: {msg['content']}\n"

            response = self.model.invoke(prompt)
            logger.info("Successfully generated conversation summary.")
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return "Summary unavailable."

    def generate_contextual_response(
        self, question: str, context: str, conversation_history: list
    ) -> str:
        """
        Generate a response from the LLM with context.
        """
        try:
            # Detect greetings and similar intent
            greetings = [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "greetings", "how are you", "what's up", "howdy", "yo", "sup"
            ]
            q_lower = question.strip().lower()
            is_greeting = any(greet in q_lower for greet in greetings)

            # Modern, conversational prompt template
            template = """
            You are a helpful, friendly, and modern hospital FAQ chatbot. 
            Answer the user's question in a conversational, natural, and approachable style. 
            Do NOT copy the context verbatim—paraphrase and make the answer sound like a real human assistant.
            If the question is a greeting or small talk, respond warmly and do NOT include any sources or citations.
            If you don't know the answer, just say you don't know.

            Context:
            {context}

            Conversation History:
            {history}

            Question: {question}

            Helpful Answer:
            """

            history = ""
            if conversation_history:
                for msg in conversation_history:
                    history += f"{msg['role']}: {msg['content']}\n"

            prompt = template.format(context=context, history=history, question=question)

            response = self.model.invoke(prompt)
            logger.info("Successfully generated contextual response from Gemini LLM.")
            # If greeting, strip any sources/citations from the answer
            if is_greeting:
                import re
                # Remove anything that looks like a citation or source
                answer = re.sub(r"\[Source:.*?\]", "", response.content)
                answer = re.sub(r"\[Relevance:.*?\]", "", answer)
                return answer.strip()
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate contextual response from Gemini LLM: {e}")
            return "Sorry, I'm having technical difficulties. Please try again later."


class SentenceTransformerEmbeddingClient:
    """Client for generating embeddings using SentenceTransformer."""

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        logger.info(
            "SentenceTransformer embedding client initialized with model: sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = (
                self.model.encode(
                    texts, show_progress_bar=False, convert_to_numpy=True
                ).tolist()
            )
            logger.info(
                f"Generated embeddings for {len(texts)} documents (SentenceTransformer)."
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate document embeddings (SentenceTransformer): {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = (
                self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
            )
            logger.info("Generated query embedding (SentenceTransformer).")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding (SentenceTransformer): {e}")
            raise

    def get_embedding_function(self):
        """Return self to be used as a LangChain compatible embedding function."""
        return self


def get_llm_client() -> GeminiLLMClient:
    """Factory function to get a Gemini LLM client instance."""
    return GeminiLLMClient()


def get_embedding_client() -> SentenceTransformerEmbeddingClient:
    """Get a SentenceTransformer embedding client instance."""
    return SentenceTransformerEmbeddingClient()


def test_gemini_connection() -> bool:
    """Quick connection test to Gemini API."""
    try:
        client = GeminiLLMClient()
        resp = client.generate_response("Hello! Can you confirm you are working?")
        return resp is not None and len(resp) > 0
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing LLM and Embedding integration...")

    if test_gemini_connection():
        print("✅ Gemini LLM connection test successful")

        # Test embedding client
        try:
            embedding_client = get_embedding_client()
            test_embedding = embedding_client.embed_query("test query")
            print(
                f"✅ Embedding client test successful (dimension: {len(test_embedding)})"
            )
        except Exception as e:
            print(f"❌ Embedding client test failed: {e}")
    else:
        print("❌ Gemini LLM connection test failed")
