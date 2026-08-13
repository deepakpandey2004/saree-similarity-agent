"""
LangChain Agent with tool schema for similarity search.

The tool schema is designed to be:
- Self-descriptive (LLM understands when to call it)
- Structured input/output (predictable)
- Rich in metadata (so LLM can explain results naturally)
"""
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src import config
from src.search_engine import get_search_engine


# ==================== TOOL SCHEMA ====================

class SareeSearchInput(BaseModel):
    """Input schema for the saree similarity search tool"""
    image_reference: str = Field(
        ...,
        description=(
            "Reference to the user's uploaded image. "
            "Always pass the exact string 'USER_UPLOADED_IMAGE' when the user has uploaded an image. "
            "The system will resolve this to the actual image."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of similar sarees to return (1-10). Default 5."
    )


class SareeSimilaritySearchTool(BaseTool):
    """
    Tool for finding visually similar sarees from the catalog.
    Uses hybrid search: Fashion-CLIP embeddings + color histograms + texture features.
    """
    name: str = "search_similar_sarees"
    description: str = (
        "Finds visually similar sarees from the byrappa silks catalog based on an uploaded image. "
        "Use this tool ONLY when the user wants to find similar sarees and has uploaded/provided an image. "
        "Do NOT use this tool for general questions or when no image is available. "
        "Returns top matches with similarity scores, product names, prices, and links."
    )
    args_schema: Type[BaseModel] = SareeSearchInput
    
    # Class-level storage for the current query image (set by app before agent call)
    _current_image = None
    
    @classmethod
    def set_query_image(cls, pil_image):
        """Called by the app before invoking the agent"""
        cls._current_image = pil_image
    
    def _run(self, image_reference: str, top_k: int = 5) -> str:
        """Execute the search"""
        if self._current_image is None:
            return "ERROR: No image has been uploaded. Please ask the user to upload a saree image first."
        
        try:
            engine = get_search_engine()
            results = engine.search(self._current_image, top_k=top_k)
            
            if not results:
                return "No similar sarees found in the catalog."
            
            # Format results for the LLM
            formatted = f"Found {len(results)} similar sarees:\n\n"
            for i, r in enumerate(results, 1):
                formatted += (
                    f"{i}. **{r['name']}** (SKU: {r['sku']})\n"
                    f"   - Overall Match Score: {r['score']:.2%}\n"
                    f"   - Visual similarity: {r['clip_score']:.2%} | "
                    f"Color match: {r['color_score']:.2%} | "
                    f"Texture match: {r['texture_score']:.2%}\n"
                    f"   - Price: ₹{r['price']:.0f} (Discounted: ₹{r['discounted_price']:.0f})\n"
                    f"   - Link: {r['website_link']}\n\n"
                )
            return formatted
        
        except Exception as e:
            return f"ERROR during search: {str(e)}"


# ==================== AGENT SETUP ====================

SYSTEM_PROMPT = """You are a helpful shopping assistant for Byrappa Silks, a saree e-commerce store.

Your role:
1. Chat naturally and warmly with users about sarees
2. When a user uploads an image and asks for similar sarees, use the `search_similar_sarees` tool
3. When calling the tool, always pass image_reference='USER_UPLOADED_IMAGE'
4. After getting results, present them in a friendly, conversational way — mention the key details (name, price, why it matches)
5. If no image is uploaded and user wants similarity search, politely ask them to upload one
6. For general chat (greetings, questions about sarees, etc.), respond naturally WITHOUT using the tool

Guidelines:
- Be concise but warm
- Highlight top matches naturally
- Never make up saree details — only use what the tool returns
- If user asks about fabric/color/style differences, explain based on tool results
"""


class SareeAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )
        self.tool = SareeSimilaritySearchTool()
        self.llm_with_tools = self.llm.bind_tools([self.tool])
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
    
    def chat(self, user_message: str, uploaded_image=None) -> str:
        """
        Process a user message. If image uploaded, agent may call the search tool.
        
        Returns: assistant's final response string
        """
        # Update tool's image context
        if uploaded_image is not None:
            SareeSimilaritySearchTool.set_query_image(uploaded_image)
        
        # Add user message (with hint about image if uploaded)
        user_content = user_message
        if uploaded_image is not None:
            user_content += "\n\n[User has uploaded an image. If they want similar sarees, use the search tool.]"
        
        self.history.append(HumanMessage(content=user_content))
        
        # First LLM call
        response = self.llm_with_tools.invoke(self.history)
        self.history.append(response)
        
        # Handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_output = self.tool.invoke(tool_call["args"])
                from langchain_core.messages import ToolMessage
                self.history.append(
                    ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                )
            
            # Get final response after tool execution
            final_response = self.llm_with_tools.invoke(self.history)
            self.history.append(final_response)
            return final_response.content
        
        return response.content
    
    def reset(self):
        """Reset conversation history"""
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
        SareeSimilaritySearchTool._current_image = None


# Singleton
_agent = None

def get_agent() -> SareeAgent:
    global _agent
    if _agent is None:
        _agent = SareeAgent()
    return _agent