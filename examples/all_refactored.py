import asyncio
import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Required,
    Set,
    Tuple,
    Union,
)
from zoneinfo import ZoneInfo

import redis
import yaml
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from livekit.agents import (
    Agent,
    ChatContext,
    ChatMessage,
    FunctionTool,
    ModelSettings,
    StopResponse,
    llm,
)
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from core.jupiter_logger_config import logger
from core.logger_config import format_metrics
from ...config import get_settings
from ...frameworks.langgraph.langgraph_tools import create_tools_dict
from ...services.analyzer_client import AnalyzerClient
from core.jupiter_redis_event_handler import RedisEventHandler
from core.jupiter_startup_connection_tests import run_startup_connection_tests
from core.jupiter_utils import Utils
from core.minimal_llm import MinimalLLM
from core.minimal_tts import MinimalTTS
from core.multi_user_transcriber import MultiUserTranscriber
from crm.jupiter_crm_api import CRMApiClient
from jupiter_tradesly_llm.app.utils import (
    InitInboundCallRequest,
    init_inbound_call,
)
from livekit import api, rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)
from livekit.agents.voice.room_io.room_io import TextInputEvent
from livekit.plugins import (
    assemblyai,
    cartesia,
    deepgram,
    elevenlabs,
    google,
    groq,
    noise_cancellation,
    openai,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from prompts.prompt_loader import PromptLoader

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

FILLER_MESSAGE_BREAK_TIME = os.getenv("FILLER_MESSAGE_BREAK_TIME")
ENABLE_TOKEN_STREAMING = os.getenv("ENABLE_TOKEN_STREAMING", "false").lower() == "true"
DEFAULT_MESSAGE_TIMEOUT = float(os.getenv("MESSAGE_TIMEOUT", "300"))
DEFAULT_IDLE_TIMEOUT = float(os.getenv("IDLE_TIMEOUT", "1800"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))

_pool: Optional[AsyncConnectionPool] = None
_conversation_manager: Optional["ConversationManager"] = None
_manager_lock = asyncio.Lock()


# ============================================================================
# ENUMS
# ============================================================================

class LiveKitPipelineEvent(Enum):
    """Types of events that can be triggered in the agent."""
    USER_INPUT_TRANSCRIBED = "user_input_transcribed"
    FUNCTION_TOOLS_EXECUTED = "function_tools_executed"
    METRICS_COLLECTED = "metrics_collected"
    SPEECH_CREATED = "speech_created"
    ERROR = "error"
    CLOSE = "close"


class LiveKitRoomEvent(Enum):
    """Types of events that can be triggered in the room."""
    ROOM_STARTED = "room_started"
    PARTICIPANT_CONNECTED = "participant_connected"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    TRACK_PUBLISHED = "track_published"
    TRACK_UNPUBLISHED = "track_unpublished"
    TRACK_SUBSCRIBED = "track_subscribed"
    TRACK_UNSUBSCRIBED = "track_unsubscribed"
    TRACK_SUBSCRIPTION_FAILED = "track_subscription_failed"
    TRACK_MUTED = "track_muted"
    TRACK_UNMUTED = "track_unmuted"
    ACTIVE_SPEAKERS_CHANGED = "active_speakers_changed"
    ROOM_METADATA_CHANGED = "room_metadata_changed"
    PARTICIPANT_METADATA_CHANGED = "participant_metadata_changed"
    PARTICIPANT_NAME_CHANGED = "participant_name_changed"
    CONNECTION_QUALITY_CHANGED = "connection_quality_changed"
    DATA_RECEIVED = "data_received"
    SIP_DTMF_RECEIVED = "sip_dtmf_received"
    CONNECTION_STATE_CHANGED = "connection_state_changed"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    RECONNECTED = "reconnected"
    TRANSCRIPTION_RECEIVED = "transcription_received"


class RouterMode(str, Enum):
    """Routing strategy for selecting the active agent during a call."""
    HANDOFF = "handoff"
    ORCHESTRATOR = "orchestrator"


# ============================================================================
# DATA MODELS - Configuration
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: str
    system_prompt: str
    tools: Optional[List[str]] = None
    model_name: Optional[str] = Field(default=None, description="LLM model name")
    responsibility: Optional[str] = Field(default=None, description="Agent responsibility")


class AgentsConfig(BaseModel):
    """Configuration for all agents in the system."""
    global_system_prompt_template: str
    agents: List[AgentConfig]
    silence_tool_message: Optional[bool] = False
    router_mode: RouterMode = RouterMode.HANDOFF
    orchestrator_prompt: Optional[str] = None
    security_prompt: Union[str, Literal["Auto"]] = "Auto"
    crm_integration: Optional[str] = None
    sanitized_keywords: List[str] = Field(
        default_factory=lambda: [
            keyword.strip()
            for keyword in os.environ.get(
                "AI_MESSAGE_SANITIZED_KEYWORDS", "[INTERNAL], [CRITICAL]"
            ).split(",")
            if keyword.strip()
        ]
    )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "AgentsConfig":
        """Load configuration from YAML string."""
        config_dict = yaml.safe_load(yaml_content)
        if "agents" in config_dict:
            config_dict["agents"] = [
                AgentConfig(**agent_data) for agent_data in config_dict["agents"]
            ]
        return cls(**config_dict)


# ============================================================================
# DATA MODELS - Reference Data
# ============================================================================

@dataclass
class RefData:
    """Reference data structure for ID validation."""
    id: int
    name: str


@dataclass
class ServiceTitanRefJobType(RefData):
    """ServiceTitan job type reference data."""
    business_unit_ids: List[int]
    business_unit_names: List[str] = field(default_factory=list)


@dataclass
class ServiceTitanRefJob:
    """ServiceTitan job reference."""
    jobTypeId: int
    businessUnitId: int
    summary: str


@dataclass
class ServiceTitanRefAppointment:
    """ServiceTitan appointment reference."""
    status: str
    job: Optional[ServiceTitanRefJob] = None


# ============================================================================
# DATA MODELS - Business Entities
# ============================================================================

class Ticket(BaseModel):
    """Ticket entity."""
    id: int
    createdAt: datetime
    summary: Optional[str] = None
    status: Optional[str] = None


@dataclass
class Appointment:
    """Appointment entity."""
    id: int
    start: str
    end: str
    summary: Optional[str] = None
    status: str = ""
    staffName: Optional[str] = None
    syncedAppointment: Optional[ServiceTitanRefAppointment] = None


class Job(BaseModel):
    """Job entity."""
    id: int
    createdAt: datetime
    summary: Optional[str] = None
    status: Optional[str] = None
    location: Optional[str] = None
    appointments: List[Appointment] = Field(default_factory=list)


class ConfirmedContext(BaseModel):
    """Confirmed context for appointments, jobs, and tickets."""
    appointment: Optional[Appointment] = None
    job: Optional[Job] = None
    ticket: Optional[Ticket] = None


# ============================================================================
# DATA MODELS - State Management
# ============================================================================

def identity_reducer(a, b) -> Any:
    """Reducer that always returns the second value."""
    return b


def concat_if_not_none(a, b) -> Any:
    """Reducer that concatenates lists, removing duplicates."""
    if a is None and b is None:
        return []
    if b is None:
        return a
    if a is None:
        return b
    return list(set(a + b))


def merge_kv_reducer(a, b) -> dict:
    """Merge two flat dicts with deletion support via None values."""
    base = dict(a or {})
    if not b:
        return base
    for k, v in b.items():
        if v is None:
            base.pop(k, None)
        else:
            base[k] = v
    return base


class TradeslyState(AgentState):
    """Complete state of a call assistance session."""
    
    # Company information
    company_name: Annotated[Optional[str], identity_reducer] = "not_provided"
    company_phone_number: Annotated[Optional[str], identity_reducer] = "not_provided"
    company_timezone: Annotated[Optional[str], identity_reducer] = "America/New_York"

    # Caller information
    customer_id: Annotated[Optional[int], identity_reducer] = None
    caller_first_name: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_last_name: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_address: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_city: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_state_province: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_zip_code: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_email: Annotated[Optional[str], identity_reducer] = "not_provided"
    caller_phone_number: Annotated[Optional[str], identity_reducer] = "not_provided"
    zip_code_in_service_area: Annotated[Optional[str], identity_reducer] = "not_provided"
    customer_type: Annotated[Optional[str], identity_reducer] = "not_provided"

    # Session information
    thread_id: Annotated[Optional[str], identity_reducer] = "not_provided"
    room_name: Annotated[Optional[str], identity_reducer] = "not_provided"
    active_agent: Annotated[Optional[str], identity_reducer] = "not_provided"
    active_agent_prompt: Annotated[Optional[str], identity_reducer] = "not_provided"
    now: Annotated[Optional[str], identity_reducer] = datetime.now(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%dT%H:%M:%S")
    call_id: Annotated[Optional[int], identity_reducer] = None
    ticket_id: Annotated[Optional[int], identity_reducer] = None
    lead_id: Annotated[Optional[int], identity_reducer] = None
    job_id: Annotated[Optional[int], identity_reducer] = None
    organization_id: Annotated[Optional[int], identity_reducer] = None
    orchestrator_suggested_agent: Annotated[Optional[str], identity_reducer] = None

    # Confidence scores
    stt_score: Annotated[Optional[float], identity_reducer] = None
    intent_recognition_score: Annotated[Optional[float], identity_reducer] = None
    contextual_matching_score: Annotated[Optional[float], identity_reducer] = None
    sentiment_score: Annotated[Optional[float], identity_reducer] = None
    overall_score: Annotated[Optional[float], identity_reducer] = None
    goal: Annotated[str, identity_reducer] = "not_provided"

    # Service information
    dispatch_fee: Annotated[Optional[str], identity_reducer] = "not_provided"
    key_points: Annotated[Optional[List[str]], concat_if_not_none] = field(default_factory=list)

    # Ticket details
    source: Annotated[Optional[str], identity_reducer] = "not_provided"
    campaign: Annotated[Optional[str], identity_reducer] = "not_provided"
    description: Annotated[Optional[str], identity_reducer] = "not_provided"
    ticket_type: Annotated[Optional[str], identity_reducer] = "not_provided"
    priority: Annotated[Optional[str], identity_reducer] = "not_provided"
    disposition: Annotated[Optional[str], identity_reducer] = "not_provided"
    ticket_description: Annotated[str, identity_reducer] = "not_provided"
    ticket_title: Annotated[Optional[str], identity_reducer] = "not_provided"

    # Appointment information
    time_booked: Annotated[Optional[str], identity_reducer] = "not_provided"
    duration: Annotated[Optional[int], identity_reducer] = None

    # Reference information
    ref_call_types: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_campaigns: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_job_statuses: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_job_types: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_lead_sources: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_lead_statuses: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_ticket_priorities: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_ticket_statuses: Annotated[Optional[str], identity_reducer] = "not_provided"
    ref_ticket_types: Annotated[Optional[str], identity_reducer] = "not_provided"

    # ServiceTitan master data
    st_business_units: Annotated[Optional[List[RefData]], identity_reducer] = None
    st_campaigns: Annotated[Optional[List[RefData]], identity_reducer] = None
    st_job_types: Annotated[Optional[List[ServiceTitanRefJobType]], identity_reducer] = None
    st_jobs: Annotated[Optional[List[ServiceTitanRefJob]], identity_reducer] = None
    st_appointments: Annotated[Optional[List[ServiceTitanRefAppointment]], identity_reducer] = None

    # Agent state (flat key/value store)
    agent_state: Annotated[Dict[str, str], merge_kv_reducer] = field(default_factory=dict)

    # CRM integration
    crm_integration: Annotated[Optional[str], identity_reducer] = None

    # Appointments and context
    future_appointments: Annotated[Optional[List[Appointment]], identity_reducer] = None
    confirmed_context: Annotated[Optional[ConfirmedContext], identity_reducer] = None


# ============================================================================
# DATA MODELS - Metrics
# ============================================================================

@dataclass
class LLMMetrics:
    """Metrics data for LLM operations."""
    timestamp: float
    ttft: float
    duration: float
    request_id: str
    label: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    tokens_per_second: float
    error: Optional[str] = None
    cancelled: bool = False
    speech_id: Optional[str] = None
    type: Literal["llm_metrics"] = "llm_metrics"


@dataclass
class MetricsEvent:
    """Event wrapper for LLM metrics."""
    metrics: LLMMetrics


@dataclass
class TotalLatencyMetrics:
    """Total latency metrics for a conversation cycle."""
    total: float
    eou: float
    llm: float
    tts: float


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class DotAccessor:
    """Safe dot/item access wrapper for template usage."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if isinstance(self._data, dict):
            return DotAccessor(self._data.get(name))
        return DotAccessor(None)

    def __getitem__(self, key):
        if isinstance(self._data, dict):
            return DotAccessor(self._data.get(key))
        if isinstance(self._data, (list, tuple)):
            try:
                return DotAccessor(self._data[key])
            except Exception:
                return DotAccessor(None)
        return DotAccessor(None)

    def __str__(self) -> str:
        d = self._data
        if d is None:
            return ""
        if isinstance(d, str):
            return d
        if isinstance(d, (int, float, bool)):
            return str(d)
        if isinstance(d, (list, tuple)):
            return ", ".join(str(DotAccessor(x)) for x in d)
        return ""

    def __repr__(self) -> str:
        return f"DotAccessor({self._data!r})"

    def __bool__(self) -> bool:
        d = self._data
        if d is None:
            return False
        if isinstance(d, (str, list, tuple, dict)):
            return bool(d)
        return True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def get_connection_pool() -> AsyncConnectionPool:
    """Return a singleton AsyncConnectionPool instance for LangGraph database."""
    global _pool
    if _pool is None:
        max_size = int(os.getenv("LANGGRAPH_DB_POOL_SIZE", "20"))
        _pool = AsyncConnectionPool(
            conninfo=os.getenv("DB_URI"),
            max_size=max_size,
            open=False,
            kwargs=os.getenv("CONNECTION_KWARGS", {}),
        )
        await _pool.open()
    return _pool


def sanitize_message(
    message: str, sanitized_keywords: List[str]
) -> Tuple[str, bool, Optional[str]]:
    """Sanitize an AI message by removing content after specified keywords."""
    if not message or not sanitized_keywords:
        return message, False, None

    earliest_pos = len(message)
    matched_marker = None

    for marker in sanitized_keywords:
        pos = message.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            matched_marker = marker

    if matched_marker is not None:
        sanitized_message = message[:earliest_pos].strip()
        removed_content = message[earliest_pos:].strip()
        logger.debug(
            f"Message sanitized. Found marker '{matched_marker}'. "
            f"Removed content length: {len(removed_content)}"
        )
        return sanitized_message, True, removed_content

    return message, False, None


def inject_agent_state(state: dict) -> dict:
    """Format state dictionary for prompt template processing."""
    fmt_state = dict(state)
    if isinstance(fmt_state.get("agent_state"), dict):
        fmt_state["agent_state"] = DotAccessor(fmt_state.get("agent_state", {}))
    return fmt_state


def format_conversation_history(state: TradeslyState) -> str:
    """Format conversation history from state messages."""
    messages = state.get("messages", [])
    conversation_history = []

    for msg in messages:
        role, content = _extract_message_role_content(msg)
        if role and content:
            conversation_history.append(f"{role}: {content}")

    return (
        "\n".join(conversation_history)
        if conversation_history
        else "No conversation history available."
    )


def _extract_message_role_content(msg) -> Tuple[Optional[str], Optional[str]]:
    """Extract role and content from a message in various formats."""
    role = None
    content = None

    if isinstance(msg, tuple) and len(msg) >= 2:
        role, content = msg[0], msg[1]
    elif hasattr(msg, "type") and hasattr(msg, "content"):
        role = "Customer" if msg.type == "human" else "Agent"
        content = msg.content
    elif isinstance(msg, dict):
        msg_role = msg.get("role", msg.get("type", "human"))
        role = "Customer" if msg_role == "human" else "Agent"
        content = msg.get("content", str(msg))

    if role:
        role = _normalize_role(role)

    return role, content


def _normalize_role(role: str) -> str:
    """Normalize role names to standard Customer/Agent format."""
    role_lower = role.lower()
    if role_lower in ["human", "user", "customer"]:
        return "Customer"
    elif role_lower in ["ai", "assistant", "agent"]:
        return "Agent"
    return role


def make_handoff_tool(*, agent_name: str, config: AgentsConfig):
    """Create a tool that can return handoff via a Command."""
    tool_name = f"hand_off_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(state: Annotated[dict, InjectedState]):
        """Ask another agent for help."""
        logger.info(f"Handing off to agent {agent_name}")
        agent_config = next(
            (agent for agent in config.agents if agent.name == agent_name), None
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": state["messages"][:-1],
                "active_agent": agent_name,
                "active_agent_prompt": agent_config.system_prompt if agent_config else "",
            },
        )

    return handoff_to_agent


def get_model_for_agent(model_name: str) -> ChatOpenAI:
    """Get LLM model instance for an agent."""
    settings = get_settings()
    model = ChatOpenAI(
        model_name=model_name,
        temperature=settings.LLM_TEMPERATURE,
        max_completion_tokens=settings.LLM_MAX_TOKENS,
        top_p=settings.LLM_TOP_P,
        seed=settings.LLM_SEED,
        streaming=True,
    )

    if len(settings.FALLBACK_LLM_MODELS) > 0:
        model.extra_body = {"models": settings.FALLBACK_LLM_MODELS}
    if settings.OPEN_ROUTER_PROVIDER_SORT:
        model.extra_body = model.extra_body or {}
        model.extra_body["provider"] = {"sort": settings.OPEN_ROUTER_PROVIDER_SORT}

    return model


def inject_security_guardrails(prompt: str, agents_config: AgentsConfig) -> str:
    """Inject security guardrails into the prompt."""
    prompt = prompt.strip()
    security_guardrails_prompt = _load_security_guardrails(agents_config)

    if security_guardrails_prompt:
        prompt = f"{security_guardrails_prompt}\n\n{prompt}"

    return prompt


def _load_security_guardrails(agents_config: AgentsConfig) -> str:
    """Load security guardrails from config or file."""
    if agents_config.security_prompt != "Auto":
        return agents_config.security_prompt

    try:
        base_app_dir = Path(__file__).resolve().parents[2]
        guardrail_path = base_app_dir / "prompts" / "security_guardrail_prompt.md"
        if guardrail_path.exists():
            return guardrail_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass

    return ""


# ============================================================================
# AGENT ORCHESTRATOR
# ============================================================================

class AgentOrchestrator:
    """Orchestrates multiple agents using LangGraph."""

    NO_AGENT_PROVIDED = "not_provided"
    MAX_SENT_EVENTS = 1000
    ORCHESTRATOR_NODE = "orchestrator"
    ANALYZER_NODE = "analyzer"
    DECISION_NODE = "decision"
    NEW_ACTIVE_AGENT_NODE = "new_active_agent"

    def __init__(
        self, chat_id: str, config: AgentsConfig, initial_context: TradeslyState
    ):
        self.chat_id = chat_id
        self.config = config
        self.initial_context = initial_context
        self.graph = None

        self.state = self._initialize_state(initial_context)
        self.langgraph_config = self._create_langgraph_config(chat_id)

        self.sent_events: Dict[str, bool] = {}
        self.is_first = True
        self.settings = get_settings()
        self.background_tasks: set = set()
        self.checkpointer = None
        self.analyzer_client = None
        self.last_activity = datetime.now()

    def _initialize_state(self, initial_context: TradeslyState) -> TradeslyState:
        """Initialize orchestrator state."""
        state = initial_context
        state["active_agent"] = self.config.agents[0].name
        state["active_agent_prompt"] = self.config.agents[0].system_prompt
        state["crm_integration"] = self.config.crm_integration
        state["thread_id"] = self.chat_id
        return state

    def _create_langgraph_config(self, chat_id: str) -> dict:
        """Create LangGraph configuration."""
        return {
            "configurable": {
                "thread_id": chat_id,
                "ticket_id": self.state.get("ticket_id"),
            },
            "recursion_limit": 20,
        }

    async def initialize(self) -> bool:
        """Initialize the graph asynchronously."""
        if not self.config:
            logger.error("No configuration provided for orchestrator")
            return False

        if self.graph:
            logger.warning("Graph already initialized")
            return True

        try:
            await self._setup_dependencies()
            await self._build_graph()

            if not self.graph:
                logger.error("Failed to build graph")
                return False

            logger.info(f"Orchestrator initialized for chat {self.chat_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
            return False

    async def _setup_dependencies(self):
        """Setup checkpointer and analyzer client."""
        connection_pool = await get_connection_pool()
        self.checkpointer = AsyncPostgresSaver(connection_pool)

        try:
            await self.checkpointer.setup()
        except Exception as e:
            logger.warning(f"Checkpointer setup warning: {e}")

        self.analyzer_client = AnalyzerClient()
        await self.analyzer_client.connect()

    async def _make_agent(
        self,
        agent_name: str,
        model: ChatOpenAI,
        tools: list,
        global_system_prompt: str,
    ):
        """Create a React agent with the given configuration."""

        def prompt_modifier(state):
            fmt_state = inject_agent_state(state)
            active_agent_prompt = fmt_state.get("active_agent_prompt", "")
            if active_agent_prompt:
                active_agent_prompt = active_agent_prompt.strip().format(**fmt_state)
                fmt_state["active_agent_prompt"] = active_agent_prompt

            prompt = inject_security_guardrails(global_system_prompt, self.config)
            return [
                ("system", prompt.strip().format(**fmt_state)),
                *state["messages"],
            ]

        agent = create_react_agent(
            model,
            name=agent_name,
            tools=tools,
            prompt=prompt_modifier,
            state_schema=TradeslyState,
        )
        return agent

    async def _build_graph(self):
        """Build agent graph based on router mode."""
        if not self.config.agents:
            logger.error("No agents configured")
            return

        builder = StateGraph(TradeslyState)
        tools_dict = create_tools_dict(self.state)

        if self.config.router_mode == RouterMode.HANDOFF:
            builder = await self._build_graph_for_handoff_router(builder, tools_dict)
        elif self.config.router_mode == RouterMode.ORCHESTRATOR:
            builder = await self._build_graph_for_orchestrator_router(builder, tools_dict)
        else:
            logger.warning(
                f"Unrecognized router_mode: {self.config.router_mode}, defaulting to handoff"
            )
            builder = await self._build_graph_for_handoff_router(builder, tools_dict)

        self.graph = builder.compile(checkpointer=self.checkpointer)
        self._log_graph_diagram()

    def _log_graph_diagram(self):
        """Log the graph diagram for debugging."""
        logger.debug("Graph built successfully")
        try:
            logger.debug(self.graph.get_graph().draw_mermaid())
        except Exception as e:
            logger.warning(f"Failed to draw graph diagram: {e}")

    async def _offloaded_analyzer_node(self, state: TradeslyState) -> Dict[str, Any]:
        """Process state through external analyzer service (fire-and-forget)."""
        if not self.analyzer_client:
            logger.warning("Analyzer client not initialized, skipping analysis")
            return {}

        task = asyncio.create_task(self.analyzer_client.analyze_async(state))
        self.background_tasks.add(task)
        task.add_done_callback(lambda t: self._handle_analyzer_task_completion(t))

        return {}

    def _handle_analyzer_task_completion(self, task: asyncio.Task):
        """Handle completion of analyzer task."""
        self.background_tasks.discard(task)
        try:
            exception = task.exception()
            if exception:
                logger.error(
                    f"Analyzer task failed for chat {self.chat_id}: {exception}",
                    exc_info=(type(exception), exception, exception.__traceback__),
                )
        except asyncio.CancelledError:
            logger.debug(f"Analyzer task was cancelled for chat {self.chat_id}")
        except Exception as e:
            logger.error(f"Error retrieving task exception: {e}")

    async def orchestrator_node(self, state: TradeslyState) -> Dict[str, str]:
        """Determine which agent should be active based on conversation context."""
        prompt_template = self.config.orchestrator_prompt
        fmt_state = inject_agent_state(state)
        system_prompt = prompt_template.format(**fmt_state)

        formatted_conversation = format_conversation_history(state)
        system_prompt += f"\n\nConversation:\n{formatted_conversation}"

        llm_input = [("system", system_prompt)]
        model = get_model_for_agent(self.settings.DEFAULT_LLM_MODEL)
        response = await model.ainvoke(llm_input)

        suggested_agent = response.content.strip()
        agent_names = [agent.name for agent in self.config.agents]

        if not agent_names:
            logger.error("No agents available in configuration")
            return {"orchestrator_suggested_agent": self.NO_AGENT_PROVIDED}

        if suggested_agent not in agent_names:
            logger.warning(
                f"Orchestrator suggested invalid agent '{suggested_agent}', keeping current agent"
            )
            suggested_agent = state.get("active_agent", agent_names[0])

        logger.info(f"Orchestrator suggested agent: {suggested_agent}")
        return {"orchestrator_suggested_agent": suggested_agent}

    async def _build_graph_for_orchestrator_router(
        self, builder: StateGraph, tools_dict: Dict
    ) -> StateGraph:
        """Build agent graph for orchestrator router mode."""

        def create_agent_wrapper(agent_executable):
            async def agent_wrapper(state: TradeslyState, config: dict):
                result = await agent_executable.ainvoke(state, config)
                if isinstance(result, dict) and "orchestrator_suggested_agent" in result:
                    del result["orchestrator_suggested_agent"]

                messages = result.get("messages", [])
                if messages and self._is_completion_message(messages[-1]):
                    result["messages"] = messages[:-1]
                return result

            return agent_wrapper

        async def decision_node(state: TradeslyState):
            new_active_agent = state.get("orchestrator_suggested_agent")
            if not new_active_agent or new_active_agent == self.NO_AGENT_PROVIDED:
                logger.warning("No valid agent suggestion from orchestrator")
                return Command(goto=self.ANALYZER_NODE)

            current_active_agent = state["active_agent"]
            if new_active_agent != current_active_agent:
                return self._create_agent_switch_command(new_active_agent)
            else:
                logger.info(f"Keeping current agent {current_active_agent}")
                return Command(goto=self.ANALYZER_NODE)

        async def new_active_agent_node(state: TradeslyState):
            return Command(goto=state.get("active_agent"))

        # Create and wrap all agent nodes
        for agent in self.config.agents:
            tools = self._get_agent_tools(agent, tools_dict)
            lg_agent = await self._make_agent(
                agent.name,
                get_model_for_agent(agent.model_name or self.settings.DEFAULT_LLM_MODEL),
                tools=tools,
                global_system_prompt=self.config.global_system_prompt_template,
            )
            wrapped_agent = create_agent_wrapper(lg_agent)
            builder.add_node(agent.name, wrapped_agent)

        builder.add_node(self.ORCHESTRATOR_NODE, self.orchestrator_node)
        builder.add_node(self.ANALYZER_NODE, self._offloaded_analyzer_node)
        builder.add_node(self.DECISION_NODE, decision_node)
        builder.add_node(self.NEW_ACTIVE_AGENT_NODE, new_active_agent_node)

        self._setup_orchestrator_edges(builder)

        return builder

    def _is_completion_message(self, message) -> bool:
        """Check if message indicates completion."""
        return (
            isinstance(message, AIMessage)
            and "<complete>" in message.content.strip().lower()
        )

    def _create_agent_switch_command(self, new_active_agent: str):
        """Create command to switch to a new agent."""
        logger.info(f"Switching to {new_active_agent}")
        agent_config = next(
            (agent for agent in self.config.agents if agent.name == new_active_agent),
            None,
        )
        if not agent_config:
            logger.error(f"Agent configuration not found for {new_active_agent}")
            return Command(goto=self.ANALYZER_NODE)

        return Command(
            update={
                "active_agent": new_active_agent,
                "active_agent_prompt": agent_config.system_prompt,
            },
            goto=self.NEW_ACTIVE_AGENT_NODE,
        )

    def _get_agent_tools(self, agent: AgentConfig, tools_dict: Dict) -> List:
        """Get tools for an agent."""
        tools = []
        if agent.tools:
            for tool_name in agent.tools:
                if tool_name in tools_dict:
                    tools.append(tools_dict[tool_name])

        if "save_agent_state_fn" in tools_dict:
            tools.append(tools_dict["save_agent_state_fn"])

        return tools

    def _setup_orchestrator_edges(self, builder: StateGraph):
        """Setup edges for orchestrator router mode."""
        def start_router(state: TradeslyState) -> list[str]:
            active_agent = state["active_agent"]
            if active_agent == self.NO_AGENT_PROVIDED:
                active_agent = self.config.agents[0].name
                state["active_agent"] = active_agent

            logger.info(f"START router: routing to {active_agent} and orchestrator in parallel")
            return [active_agent, self.ORCHESTRATOR_NODE]

        builder.add_conditional_edges(
            START,
            start_router,
            {agent.name: agent.name for agent in self.config.agents}
            | {self.ORCHESTRATOR_NODE: self.ORCHESTRATOR_NODE},
        )

        for agent in self.config.agents:
            builder.add_edge(agent.name, self.DECISION_NODE)
        builder.add_edge(self.ORCHESTRATOR_NODE, self.DECISION_NODE)
        builder.add_edge(self.ANALYZER_NODE, END)

    async def _build_graph_for_handoff_router(
        self, builder: StateGraph, tools_dict: Dict
    ) -> StateGraph:
        """Build agent graph for handoff router mode."""
        for agent in self.config.agents:
            handoff_tools = self._get_handoff_tools(agent)
            regular_tools = self._get_regular_tools(agent, tools_dict)

            if "save_agent_state_fn" in tools_dict:
                regular_tools.append(tools_dict["save_agent_state_fn"])

            lg_agent = await self._make_agent(
                agent.name,
                get_model_for_agent(agent.model_name or self.settings.DEFAULT_LLM_MODEL),
                tools=regular_tools + handoff_tools,
                global_system_prompt=self.config.global_system_prompt_template,
            )

            builder.add_node(agent.name, lg_agent)

        builder.add_node(self.ANALYZER_NODE, self._offloaded_analyzer_node)

        def router(state: TradeslyState) -> str:
            active_agent = state["active_agent"]
            if active_agent == self.NO_AGENT_PROVIDED:
                active_agent = self.config.agents[0].name
            logger.debug(f"Routing request to active agent: {active_agent}")
            return active_agent

        routing_destinations = {agent.name: agent.name for agent in self.config.agents}
        builder.add_conditional_edges(START, router, routing_destinations)

        for agent in self.config.agents:
            builder.add_edge(agent.name, self.ANALYZER_NODE)

        builder.add_edge(self.ANALYZER_NODE, END)

        return builder

    def _get_handoff_tools(self, agent: AgentConfig) -> List:
        """Get handoff tools for an agent."""
        handoff_tools = []
        if agent.tools:
            for tool_name in agent.tools:
                if tool_name.startswith("hand_off_to_"):
                    target_agent = tool_name.replace("hand_off_to_", "")
                    handoff_tools.append(
                        make_handoff_tool(agent_name=target_agent, config=self.config)
                    )
        return handoff_tools

    def _get_regular_tools(self, agent: AgentConfig, tools_dict: Dict) -> List:
        """Get regular (non-handoff) tools for an agent."""
        regular_tools = []
        if agent.tools:
            for tool_name in agent.tools:
                if not tool_name.startswith("hand_off_to_"):
                    if tool_name in tools_dict:
                        regular_tools.append(tools_dict[tool_name])
                    else:
                        logger.warning(
                            f"Tool '{tool_name}' not found in tools_dict for agent '{agent.name}'"
                        )
        return regular_tools

    def _cleanup_sent_events(self):
        """Clean up sent_events dictionary to prevent memory leak."""
        if len(self.sent_events) > self.MAX_SENT_EVENTS:
            keys_to_remove = list(self.sent_events.keys())[: self.MAX_SENT_EVENTS // 2]
            for key in keys_to_remove:
                del self.sent_events[key]
            logger.debug(
                f"Cleaned up {len(keys_to_remove)} old event entries for chat {self.chat_id}"
            )

    def _is_valid_update(self, update: Any) -> bool:
        """Validate update structure from graph stream."""
        if not isinstance(update, tuple) or len(update) != 3:
            return False

        namespace, stream_mode, _ = update

        if stream_mode != "updates":
            return False

        if not isinstance(namespace, tuple) or len(namespace) == 0:
            return False

        if not any(isinstance(ns, str) and "_agent" in ns for ns in namespace):
            return False

        return True

    def _extract_agent_name(self, namespace: tuple) -> Optional[str]:
        """Extract agent name from namespace tuple."""
        try:
            last_component = namespace[-1]
            if not isinstance(last_component, str):
                logger.warning(f"Last namespace component is not string: {type(last_component)}")
                return None

            if "_agent" in last_component:
                return last_component.replace("_agent", "")

            return last_component.split("_")[0]

        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to extract agent name from namespace {namespace}: {e}")
            return None

    async def _parse_and_yield_chunks(
        self, msg: AIMessage | ToolMessage, agent_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse incoming message and yield individual response objects."""
        if hasattr(msg, "id") and msg.id in self.sent_events:
            return

        self._cleanup_sent_events()

        data = {"agent": agent_name, "id": msg.id if hasattr(msg, "id") else None}

        if isinstance(msg, AIMessage):
            async for item in self._handle_ai_message(msg, data):
                yield item
        elif isinstance(msg, ToolMessage):
            yield self._handle_tool_message(msg, data)

        if hasattr(msg, "id") and msg.id:
            self.sent_events[msg.id] = True

    async def _handle_ai_message(
        self, msg: AIMessage, data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle AI message processing."""
        has_content = hasattr(msg, "content") and msg.content
        original_content = msg.content if has_content else None
        sanitized_content, was_sanitized, removed_content = sanitize_message(
            original_content or "", self.config.sanitized_keywords
        )

        event_data = {
            **data,
            "content": original_content,
            "sanitized_content": sanitized_content if was_sanitized else None,
            "was_sanitized": was_sanitized,
            "removed_content": removed_content,
        }

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            event_data["tool_calls"] = msg.tool_calls
            yield {"type": "debug", "event": "AI message with tool calls", "data": event_data}

            tool_name = msg.tool_calls[0]["name"]
            if (
                not self.config.silence_tool_message
                and has_content
                and not tool_name.startswith("hand_off_")
            ):
                content = sanitized_content
                if FILLER_MESSAGE_BREAK_TIME is not None:
                    content += f'<break time="{FILLER_MESSAGE_BREAK_TIME}" />'
                yield {"type": "message", "content": content, "id": msg.id}
            else:
                logger.info(
                    f"Tool message from {tool_name} not sent to client. "
                    f"AI message: {sanitized_content if was_sanitized else original_content}"
                )
        else:
            yield {"type": "debug", "event": "AI Message", "data": event_data}
            if has_content:
                yield {"type": "message", "content": sanitized_content, "id": msg.id}

    def _handle_tool_message(self, msg: ToolMessage, data: dict) -> dict:
        """Handle tool message processing."""
        return {
            "type": "debug",
            "event": "Tool call result",
            "data": {
                **data,
                "tool_call_id": msg.tool_call_id if hasattr(msg, "tool_call_id") else None,
                "content": msg.content if hasattr(msg, "content") else None,
            },
        }

    async def _process_update(self, update: Any) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a single update from the graph stream."""
        if not self._is_valid_update(update):
            return

        namespace, stream_mode, chunk = update
        agent_name = self._extract_agent_name(namespace)
        if not agent_name:
            return

        if not isinstance(chunk, dict):
            logger.warning(f"Unexpected chunk type: {type(chunk)}")
            return

        agent_messages = chunk.get("agent", {}).get("messages", [])
        tool_messages = chunk.get("tools", {}).get("messages", [])

        for message in agent_messages + tool_messages:
            async for item in self._parse_and_yield_chunks(message, agent_name):
                yield item

    async def _process_message_internal(
        self, message: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Internal method to process a single message (without lock)."""
        if not self.graph:
            raise ValueError("Graph not initialized. Call initialize() first.")

        self.last_activity = datetime.now()

        try:
            input_state = {"messages": [("user", message)]}

            if self.is_first:
                input_state.update(self.state)
                self.is_first = False

            yield {"type": "debug", "event": "graph_execution_start"}

            response = self.graph.astream(
                input_state,
                self.langgraph_config,
                stream_mode=["updates"],
                subgraphs=True,
            )

            async for update in response:
                async for item in self._process_update(update):
                    yield item

            yield {"type": "debug", "event": "graph_execution_complete"}

        except Exception as e:
            logger.error(f"Error processing message in chat {self.chat_id}: {e}", exc_info=True)
            yield {
                "type": "error",
                "event": "graph_execution_error",
                "data": {"error": str(e)},
            }

    async def cleanup(self):
        """Properly clean up resources."""
        logger.info(f"Cleaning up resources for chat {self.chat_id}")

        if self.background_tasks:
            logger.info(f"Cancelling {len(self.background_tasks)} background tasks")
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()

            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

        if self.analyzer_client:
            try:
                await self.analyzer_client.stop()
                logger.info("Analyzer client stopped")
            except Exception as e:
                logger.error(f"Error stopping analyzer client: {e}")

        self.sent_events.clear()
        self.background_tasks.clear()

        logger.info(f"Cleanup complete for chat {self.chat_id}")


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manages multiple orchestrators with message queuing for concurrency safety."""

    def __init__(self):
        self.orchestrators: Dict[str, AgentOrchestrator] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.configs: Dict[str, Tuple[AgentsConfig, TradeslyState]] = {}
        self._shutdown = False

    async def initialize(self):
        """Initialize the conversation manager."""
        logger.info("ConversationManager initialized")

    async def register_conversation(
        self, chat_id: str, config: AgentsConfig, initial_context: TradeslyState
    ) -> bool:
        """Register a conversation and eagerly initialize orchestrator."""
        if chat_id in self.configs:
            logger.warning(f"Conversation {chat_id} already registered")
            return True

        try:
            self.configs[chat_id] = (config, initial_context)

            orchestrator = AgentOrchestrator(chat_id, config, initial_context)
            initialized = await orchestrator.initialize()

            if not initialized:
                logger.error(f"Failed to initialize orchestrator for {chat_id}")
                del self.configs[chat_id]
                return False

            self.orchestrators[chat_id] = orchestrator
            self.message_queues[chat_id] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

            task = asyncio.create_task(self._process_queue(chat_id, orchestrator))
            self.processing_tasks[chat_id] = task

            logger.info(f"Conversation {chat_id} registered and orchestrator initialized")
            return True

        except Exception as e:
            logger.error(f"Error registering conversation {chat_id}: {e}", exc_info=True)
            await self._cleanup_failed_registration(chat_id)
            return False

    async def _cleanup_failed_registration(self, chat_id: str):
        """Clean up after failed registration."""
        if chat_id in self.configs:
            del self.configs[chat_id]
        if chat_id in self.orchestrators:
            await self.orchestrators[chat_id].cleanup()
            del self.orchestrators[chat_id]

    async def send_message(
        self, chat_id: str, message: str, timeout: float = DEFAULT_MESSAGE_TIMEOUT
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Queue message and stream responses."""
        if self._shutdown:
            yield {
                "type": "error",
                "event": "manager_shutdown",
                "data": {"error": "ConversationManager is shutting down"},
            }
            return

        if chat_id not in self.orchestrators:
            yield {
                "type": "error",
                "event": "conversation_not_registered",
                "data": {
                    "error": f"Conversation {chat_id} not registered. "
                    "Call register_conversation first."
                },
            }
            return

        if chat_id not in self.message_queues:
            yield {
                "type": "error",
                "event": "queue_not_found",
                "data": {"error": f"Message queue for {chat_id} not found."},
            }
            return

        response_queue = asyncio.Queue()

        try:
            await self._enqueue_message(chat_id, message, response_queue, timeout)

            async for response in self._read_responses(response_queue):
                yield response

        except Exception as e:
            logger.error(f"Error in send_message for chat {chat_id}: {e}", exc_info=True)
            yield {
                "type": "error",
                "event": "send_message_error",
                "data": {"error": str(e)},
            }

    async def _enqueue_message(
        self, chat_id: str, message: str, response_queue: asyncio.Queue, timeout: float
    ):
        """Enqueue a message for processing."""
        try:
            await asyncio.wait_for(
                self.message_queues[chat_id].put((message, response_queue, timeout)),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Message queue is full. Please try again later.")

    async def _read_responses(self, response_queue: asyncio.Queue):
        """Read responses from the response queue."""
        while True:
            try:
                response = await response_queue.get()
                if response is None:
                    break
                yield response
            except Exception as e:
                logger.error(f"Error getting response from queue: {e}")
                yield {
                    "type": "error",
                    "event": "response_queue_error",
                    "data": {"error": str(e)},
                }
                break

    async def _process_queue(self, chat_id: str, orchestrator: AgentOrchestrator):
        """Process messages sequentially for a conversation."""
        queue = self.message_queues[chat_id]
        logger.info(f"Started queue processor for chat {chat_id}")

        while not self._shutdown:
            try:
                message_data = await self._wait_for_message(queue, chat_id)
                if message_data is None:
                    break

                message, response_queue, timeout = message_data
                await self._process_single_message(
                    chat_id, orchestrator, message, response_queue, timeout
                )

            except asyncio.CancelledError:
                logger.info(f"Queue processor for chat {chat_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processor for chat {chat_id}: {e}", exc_info=True)

        logger.info(f"Queue processor stopped for chat {chat_id}")

    async def _wait_for_message(self, queue: asyncio.Queue, chat_id: str):
        """Wait for a message or handle idle timeout."""
        try:
            return await asyncio.wait_for(queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            if self._should_cleanup_conversation(chat_id):
                logger.info(f"Cleaning up idle conversation {chat_id}")
                await self._cleanup_conversation(chat_id)
                return None
            return None

    async def _process_single_message(
        self,
        chat_id: str,
        orchestrator: AgentOrchestrator,
        message: str,
        response_queue: asyncio.Queue,
        timeout: float,
    ):
        """Process a single message and handle responses."""
        logger.info(f"Processing message for chat {chat_id}: {message}")

        try:
            async with asyncio.timeout(timeout):
                async for response in orchestrator._process_message_internal(message):
                    await response_queue.put(response)

        except asyncio.TimeoutError:
            logger.error(f"Message processing timed out for chat {chat_id}")
            await response_queue.put(
                {
                    "type": "error",
                    "event": "processing_timeout",
                    "data": {"error": f"Processing timed out after {timeout}s"},
                }
            )
        except Exception as e:
            logger.error(f"Error processing message for chat {chat_id}: {e}", exc_info=True)
            await response_queue.put(
                {
                    "type": "error",
                    "event": "processing_error",
                    "data": {"error": str(e)},
                }
            )
        finally:
            await response_queue.put(None)

    def _should_cleanup_conversation(self, chat_id: str) -> bool:
        """Check if conversation should be cleaned up due to inactivity."""
        if chat_id not in self.orchestrators:
            return False

        orchestrator = self.orchestrators[chat_id]
        idle_time = (datetime.now() - orchestrator.last_activity).total_seconds()

        return idle_time > DEFAULT_IDLE_TIMEOUT

    async def _cleanup_conversation(self, chat_id: str):
        """Clean up resources for a specific conversation."""
        logger.info(f"Cleaning up conversation {chat_id}")

        if chat_id in self.processing_tasks:
            task = self.processing_tasks[chat_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.processing_tasks[chat_id]

        if chat_id in self.orchestrators:
            await self.orchestrators[chat_id].cleanup()
            del self.orchestrators[chat_id]

        if chat_id in self.message_queues:
            del self.message_queues[chat_id]

        if chat_id in self.configs:
            del self.configs[chat_id]

        logger.info(f"Conversation {chat_id} cleaned up")

    async def cleanup_all(self):
        """Clean up all conversations and shut down the manager."""
        logger.info("Shutting down ConversationManager")
        self._shutdown = True

        chat_ids = list(self.orchestrators.keys())
        for chat_id in chat_ids:
            try:
                await self._cleanup_conversation(chat_id)
            except Exception as e:
                logger.error(f"Error cleaning up conversation {chat_id}: {e}")

        logger.info("ConversationManager shutdown complete")

    async def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        return list(self.orchestrators.keys())

    async def cleanup_idle_conversations(self):
        """Clean up conversations that have been idle too long."""
        chat_ids = list(self.orchestrators.keys())
        for chat_id in chat_ids:
            if self._should_cleanup_conversation(chat_id):
                await self._cleanup_conversation(chat_id)


async def get_conversation_manager() -> ConversationManager:
    """Get or create the global ConversationManager singleton."""
    global _conversation_manager

    async with _manager_lock:
        if _conversation_manager is None:
            _conversation_manager = ConversationManager()
            await _conversation_manager.initialize()
            logger.info("Global ConversationManager initialized")

        return _conversation_manager


async def cleanup_conversation_manager():
    """Cleanup the global ConversationManager."""
    global _conversation_manager

    async with _manager_lock:
        if _conversation_manager is not None:
            await _conversation_manager.cleanup_all()
            _conversation_manager = None
            logger.info("Global ConversationManager cleaned up")


# ============================================================================
# LANGGRAPH FRAMEWORK
# ============================================================================

class LanggraphFramework:
    """Framework for managing Langgraph agents with concurrency safety."""

    def __init__(self, config: Required[AgentsConfig]):
        self.config = config
        self.chat_id = None
        self.initial_context = None
        self._manager = None
        self._is_registered = False

    async def validate_setup(self) -> bool:
        """Check if Langgraph is properly configured."""
        if not self._manager or not self._is_registered:
            return False

        active_conversations = await self._manager.get_active_conversations()
        return self.chat_id in active_conversations

    async def start_session(self, chat_id: str, state: TradeslyState) -> None:
        """Initialize a new chat session with eager orchestrator initialization."""
        self.chat_id = chat_id
        self.initial_context = state

        self._manager = await get_conversation_manager()

        logger.info(f"Registering and initializing conversation {chat_id}...")
        success = await self._manager.register_conversation(
            chat_id=chat_id, config=self.config, initial_context=state
        )

        if success:
            self._is_registered = True
            logger.info(f"Conversation {chat_id} ready (orchestrator pre-initialized)")
        else:
            logger.error(f"Failed to register conversation {chat_id}")
            raise ValueError(f"Failed to register conversation {chat_id}")

    async def process_stream(
        self, chat_id: str, message: str, timeout: float = 300.0
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Process messages using Langgraph with streaming support."""
        try:
            if not self._manager or not self._is_registered:
                raise ValueError("Session not initialized. Call start_session first.")

            async for response in self._manager.send_message(
                chat_id=chat_id, message=message, timeout=timeout
            ):
                yield response

        except Exception as e:
            logger.error(f"Error processing message with Langgraph: {str(e)}", exc_info=True)
            error_trace = traceback.format_exc()
            logger.error(f"Detailed error traceback: {error_trace}")

            yield {
                "type": "debug",
                "event": "error",
                "data": {"error": str(e), "traceback": error_trace},
            }

            yield f"Error processing message: {str(e)}"

    async def cleanup_chat(self, chat_id: str) -> None:
        """Mark chat session for cleanup."""
        if self._manager and self._is_registered:
            self._is_registered = False
            logger.info(f"Chat session {chat_id} marked for cleanup")


# ============================================================================
# CONFIG MANAGER FACTORY
# ============================================================================

class ConfigManagerFactory:
    """Factory for creating config managers."""

    @staticmethod
    def get_config_manager():
        """Create and return the appropriate config manager."""
        use_redis = os.getenv("USE_REDIS_CONFIG_STORAGE", "false").lower() == "true"

        if use_redis:
            from .redis_config_manager import RedisConfig, RedisConfigManager
            redis_config = RedisConfig()
            return RedisConfigManager(redis_config)
        else:
            from .agent_config_manager import AgentConfigManager
            config_dir = os.getenv("AGENT_CONFIG_DIR", "agents-config")
            return AgentConfigManager(config_dir)


agent_config_manager = ConfigManagerFactory.get_config_manager()


# ============================================================================
# TRANSCRIPTION HANDLER
# ============================================================================

class TranscriptionHandler:
    """Handles transcription events and publishes to Redis."""

    def __init__(
        self,
        room_name: str,
        call_id: str,
        lk_job_id: str,
        redis_client: redis.StrictRedis,
    ):
        self.room_name = room_name
        self.redis_client = redis_client
        self.redis_transcript_channel = f"transcript:{room_name}"
        self.last_published_message = None
        self.call_id = call_id
        self.lk_job_id = lk_job_id

    def _publish_transcript(
        self, participant: str, text: str, is_complete: bool = True
    ):
        """Helper method to publish transcript to Redis."""
        try:
            transcript_json = json.dumps(
                {
                    "type": "transcript",
                    "data": {
                        "room_name": self.room_name,
                        "participant": participant,
                        "text": text,
                        "timestamp": datetime.now(datetime.UTC).isoformat(),
                        "call_id": self.call_id,
                        "lk_job_id": self.lk_job_id,
                        "is_complete": is_complete,
                    },
                }
            )

            self.redis_client.publish(self.redis_transcript_channel, transcript_json)
            self.publish_new_transcript()

        except Exception as e:
            logger.error(f"Error publishing transcript to Redis: {str(e)}")

    def handle_user_transcript(self, transcript):
        """Handle a user transcript event."""
        text = (
            transcript.content.strip()
            if hasattr(transcript, "content")
            else str(transcript)
        )
        logger.info(f"[Human][TRANSCRIPTION]: {text}")
        self._publish_transcript("User", text)

    def handle_agent_transcript(self, text: str):
        """Handle an agent transcript event."""
        logger.info(f"[Agent][TRANSCRIPTION]: {text}")
        self._publish_transcript("AI", text)

    def handle_agent_transcript_chunk(self, text: str):
        """Handle an agent transcript chunk (not complete)."""
        self._publish_transcript("AI", text, is_complete=False)

    def handle_system_transcript(self, text: str):
        """Handle a system transcript event."""
        logger.info(f"[System][TRANSCRIPTION]: {text}")
        self._publish_transcript("System", text)

    def handle_participant_transcript(
        self, participant_identity: str, text: str, is_final: bool = True
    ):
        """Handle transcript from any participant."""
        try:
            participant_type = self._get_participant_type(participant_identity)

            transcript_json = json.dumps(
                {
                    "type": "transcript",
                    "data": {
                        "room_name": self.room_name,
                        "participant": participant_type,
                        "participant_identity": participant_identity,
                        "text": text,
                        "timestamp": datetime.now(datetime.UTC).isoformat(),
                        "call_id": self.call_id,
                        "lk_job_id": self.lk_job_id,
                        "is_final": is_final,
                    },
                }
            )

            logger.info(f"[Participant][TRANSCRIPTION]: {text}")
            self.redis_client.publish(self.redis_transcript_channel, transcript_json)
            self.publish_new_transcript()

        except Exception as e:
            logger.error(f"Error publishing participant transcript to Redis: {str(e)}")

    def _get_participant_type(self, participant_identity: str) -> str:
        """Determine participant type based on identity."""
        identity_lower = participant_identity.lower()

        if "sip" in identity_lower or "phone" in identity_lower:
            return "User"
        elif "csr" in identity_lower:
            return "csr"
        elif "agent" in identity_lower or "ai" in identity_lower:
            return "AI"
        else:
            return "Participant"

    def publish_new_transcript(self):
        """Publish a message containing the room name to Redis."""
        try:
            message = json.dumps(
                {"type": "new_transcript", "data": {"room_name": self.room_name}}
            )

            if message == self.last_published_message:
                return

            self.redis_client.publish(self.redis_transcript_channel, message)
            self.last_published_message = message

        except Exception as e:
            logger.error(f"Error publishing new transcript message to Redis: {str(e)}")

    async def cleanup_transcriptions(self):
        """Clean up transcription resources."""
        logger.info("TranscriptionHandler cleanup completed")


# ============================================================================
# LANGGRAPH AGENT
# ============================================================================

class LangGraphAgent(Agent):
    """LiveKit Agent implementation using LangGraph framework."""

    def __init__(
        self,
        room_name: str,
        redis_client,
        transcription_handler: TranscriptionHandler,
        call_context: Optional[TradeslyState] = None,
    ) -> None:
        super().__init__(instructions="")

        self._chat_id = f"lk_{str(uuid.uuid4())}"
        self._room_name = room_name
        self._company_phone_number = call_context.get("company_phone_number")
        self._call_id = call_context["call_id"]
        self._ticket_id = call_context["ticket_id"]
        self._lead_id = call_context["lead_id"]
        self._customer_id = call_context["customer_id"]
        self._caller_phone_number = call_context["caller_phone_number"]
        self._job_id = call_context["job_id"]
        self._call_context = call_context
        self._turn_number = 0

        self._ttft_tracking = os.getenv("TTFT_TRACKING", "true").lower() == "true"

        if self._ttft_tracking:
            self._ttft_handlers: Set[Callable[[MetricsEvent], Awaitable[None]]] = set()
            queue_size = int(os.getenv("TTFT_QUEUE_SIZE", "100"))
            self._ttft_queue: asyncio.Queue[MetricsEvent] = asyncio.Queue(maxsize=queue_size)
            self._queue_overflow_count = 0
            logger.info(f"TTFT queue initialized with maxsize={queue_size}")
        else:
            self._ttft_handlers = set()
            self._ttft_queue = None
            self._queue_overflow_count = 0

        self._transcription_handler = transcription_handler

        logger.info(f"Company phone number: {self._company_phone_number}")
        config = agent_config_manager.load_config_for_chat(self._company_phone_number)
        self._langgraph = LanggraphFramework(config=config)

    async def initialize(self):
        """Async initialization with eager orchestrator setup."""
        await self._langgraph.start_session(self._chat_id, self._call_context)
        logger.info(f"Agent {self._chat_id} initialized and ready for messages")
        return self

    def get_latest_msg(self, chat_ctx: ChatContext) -> str:
        """Extract the latest user message from chat context."""
        logger.info(f"[INSIDE get_latest_message] ChatContext: {chat_ctx.items}")

        for item in reversed(chat_ctx.items):
            if item.type == "message" and item.role == "user":
                return item.content[0]

        logger.warning("No user message found in chat context")
        return None

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Handle user turn completion."""
        if not new_message.text_content:
            logger.debug("[END TURN] Stop response")
            raise StopResponse()

    async def local_langgraph_llm_node(
        self, chat_ctx: llm.ChatContext
    ) -> AsyncIterable[FunctionTool | str | llm.ChatChunk]:
        """Process message through LangGraph and yield responses."""
        latest_msg = self.get_latest_msg(chat_ctx)
        if not latest_msg:
            return

        if self._ttft_tracking:
            logger.info("TTFT tracking is enabled")
            start_time = time.time()
            first_token_time = None

        async for response in self._langgraph.process_stream(self._chat_id, latest_msg):
            if isinstance(response, dict) and response.get("type") == "message":
                content = response["content"].strip()
                if content:
                    if self._ttft_tracking and first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        await self._emit_ttft(ttft)
                    logger.debug(f"[LLM Node][Processing content]: {content}")
                    yield content
            logger.debug(f"Received response: {response}")

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: List[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[FunctionTool | str | llm.ChatChunk]:
        """LLM node implementation."""
        return self.local_langgraph_llm_node(chat_ctx)

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        """Handle transcription with logging."""
        response_stream = super().transcription_node(text, model_settings)
        async for chunk in response_stream:
            stripped_chunk = chunk.strip()
            if stripped_chunk:
                yield stripped_chunk
                self._transcription_handler.handle_agent_transcript(stripped_chunk)

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Synthesize text to speech."""
        activity = self._get_activity_or_raise()

        if not activity.tts.capabilities.streaming:
            async for frame in super().tts_node(text, model_settings):
                yield frame
            return

        original_conn_options = activity.session.conn_options.tts_conn_options

        async for chunk in text:
            if not chunk or not chunk.strip():
                logger.debug("Skipping empty chunk")
                continue

            try:
                async with activity.tts.stream(conn_options=original_conn_options) as stream:
                    logger.debug(f"[PUSH_TEXT] Generating TTS for chunk: {chunk!r}")
                    stream.push_text(chunk)
                    stream.end_input()

                    async for ev in stream:
                        yield ev.frame

            except asyncio.CancelledError:
                logger.info("TTS stream cancelled")
                raise
            except Exception as e:
                logger.error(f"Unexpected TTS error: {type(e).__name__}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break

    # TTFT Tracking Methods
    def on_ttft(self, handler: Callable[[MetricsEvent], Awaitable[None]]) -> None:
        """Register a handler for TTFT events."""
        self._ttft_handlers.add(handler)

    def off_ttft(self, handler: Callable[[MetricsEvent], Awaitable[None]]) -> None:
        """Unregister a handler for TTFT events."""
        self._ttft_handlers.discard(handler)

    async def _emit_ttft(self, ttft: float) -> None:
        """Emit a TTFT event to all registered handlers."""
        try:
            metrics = LLMMetrics(
                timestamp=time.time(),
                request_id=self._chat_id,
                ttft=ttft,
                duration=0,
                label=getattr(self._llm, "_label", "unknown"),
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                tokens_per_second=0.0,
                error=None,
            )
            event = MetricsEvent(metrics=metrics)

            current_size = self._ttft_queue.qsize()
            try:
                self._ttft_queue.put_nowait(event)
                if current_size % 10 == 0:
                    logger.debug(
                        f"TTFT queue size: {current_size + 1}/{self._ttft_queue.maxsize}"
                    )
            except asyncio.QueueFull:
                self._queue_overflow_count += 1
                logger.warning(
                    f"TTFT queue full! Size: {current_size}, "
                    f"Overflow count: {self._queue_overflow_count}"
                )
                try:
                    self._ttft_queue.get_nowait()
                    self._ttft_queue.put_nowait(event)
                    logger.debug(f"Replaced oldest event, queue size: {self._ttft_queue.qsize()}")
                except asyncio.QueueEmpty:
                    self._ttft_queue.put_nowait(event)

            for handler in self._ttft_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in TTFT handler: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error emitting TTFT event: {e}", exc_info=True)

    async def wait_for_ttft(self) -> MetricsEvent:
        """Wait for and get the next TTFT value."""
        try:
            return await self._ttft_queue.get()
        except asyncio.CancelledError:
            logger.info("TTFT wait was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error waiting for TTFT: {e}", exc_info=True)
            raise

    def get_ttft_queue_stats(self) -> dict:
        """Get statistics about TTFT queue usage."""
        return {
            "current_size": self._ttft_queue.qsize(),
            "max_size": self._ttft_queue.maxsize,
            "overflow_count": self._queue_overflow_count,
            "utilization_percent": (self._ttft_queue.qsize() / self._ttft_queue.maxsize) * 100,
        }

    async def __aexit__(self, *args, **kwargs) -> None:
        """Clean up resources when the agent is destroyed."""
        try:
            await self._langgraph.cleanup_chat(self._chat_id)
        except Exception as e:
            logger.error(f"Error cleaning up Langgraph session: {e}")

        if self._ttft_tracking:
            stats = self.get_ttft_queue_stats()
            logger.info(f"TTFT queue stats at cleanup: {stats}")

            self._ttft_handlers.clear()

            drained_count = 0
            while not self._ttft_queue.empty():
                try:
                    self._ttft_queue.get_nowait()
                    drained_count += 1
                except asyncio.QueueEmpty:
                    break

            if drained_count > 0:
                logger.info(f"Drained {drained_count} events from TTFT queue during cleanup")


# ============================================================================
# SESSION EVENT HANDLER
# ============================================================================

class SessionEventHandler:
    """Handles various session events from the LiveKit pipeline."""

    def __init__(
        self,
        room_name: str,
        transcription_handler: TranscriptionHandler,
        redis_event_handler: RedisEventHandler,
        primary_caller=None,
        cleanup_callback=None,
        is_outbound=False,
        is_web_call=False,
        user_away_timeout=60.0,
        use_multi_user_transcript=True,
    ):
        self.room_name = room_name
        self.transcription_handler = transcription_handler
        self.redis_event_handler = redis_event_handler
        self.primary_caller = primary_caller
        self.cleanup_callback = cleanup_callback
        self.is_outbound = is_outbound
        self.is_web_call = is_web_call
        self.user_away_timeout = user_away_timeout
        self.use_multi_user_transcript = use_multi_user_transcript
        self.session = None
        self.has_asked_are_you_there = False
        self.timeout_task = None

        self.current_cycle_metrics = {
            "eou_metrics": None,
            "llm_metrics": None,
            "tts_metrics": None,
        }

    def publish_event(self, event_type: str, data: dict):
        """Publish an event to Redis for monitoring and analysis."""
        self.redis_event_handler.publish_event(event_type, data)

    def calculate_total_latency(self, event):
        """Calculate and log total latency when all metrics are available."""
        self.current_cycle_metrics[event.metrics.type] = event.metrics

        if all(self.current_cycle_metrics.values()):
            total_latency = (
                self.current_cycle_metrics["eou_metrics"].end_of_utterance_delay
                + self.current_cycle_metrics["llm_metrics"].ttft
                + self.current_cycle_metrics["tts_metrics"].ttfb
            )

            total_metrics = TotalLatencyMetrics(
                total=total_latency,
                eou=self.current_cycle_metrics["eou_metrics"].end_of_utterance_delay,
                llm=self.current_cycle_metrics["llm_metrics"].ttft,
                tts=self.current_cycle_metrics["tts_metrics"].ttfb,
            )

            metrics_msg = format_metrics("total_latency", total_metrics, self.room_name)
            logger.info(json.dumps(metrics_msg))

            self.transcription_handler.handle_system_transcript(
                f"{total_latency:.3f} (EOU: {self.current_cycle_metrics['eou_metrics'].end_of_utterance_delay:.3f}, "
                f"LLM: {self.current_cycle_metrics['llm_metrics'].ttft:.3f}, "
                f"TTS: {self.current_cycle_metrics['tts_metrics'].ttfb:.3f})"
            )

            self.current_cycle_metrics = {
                "eou_metrics": None,
                "llm_metrics": None,
                "tts_metrics": None,
            }

    async def _ask_are_you_there(self):
        """Ask 'Are you still there?' to the user."""
        try:
            await self.session.say("Are you still there?")
        except Exception as e:
            logger.error(e)
            if self.cleanup_callback:
                asyncio.create_task(self.cleanup_callback("User inactive timeout reached"))

    async def _second_timeout(self):
        """Handle the second timeout after asking 'Are you there?'."""
        try:
            await asyncio.sleep(self.user_away_timeout + 5)
            if self.has_asked_are_you_there and self.redis_event_handler.ai_voice_enabled is True:
                logger.info(f"Second timeout reached, hanging up in room: {self.room_name}")
                if self.primary_caller is not None:
                    self.redis_event_handler.on_participant_disconnect(self.primary_caller)
                if self.cleanup_callback:
                    asyncio.create_task(self.cleanup_callback("User inactive timeout reached"))
        except asyncio.CancelledError:
            logger.error(f"Second timeout cancelled in room: {self.room_name}")

    def reset_are_you_there_flag(self):
        """Reset the 'Are you there?' flag and cancel the timeout task when user responds."""
        if self.has_asked_are_you_there:
            logger.info(f"Resetting 'Are you there?' flag in room: {self.room_name}")
            self.has_asked_are_you_there = False
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                logger.info(
                    f"Cancelled timeout task due to user response in room: {self.room_name}"
                )

    def setup_session_event_handlers(self, current_session):
        """Set up event handlers for various LiveKit pipeline events."""
        logger.info(f"[SETUP DEBUG] Setting up event handlers for {self.room_name}")
        self.session = current_session

        @current_session.on(LiveKitPipelineEvent.USER_INPUT_TRANSCRIBED.value)
        def on_user_transcription(event):
            if event.is_final:
                logger.info(f"[Human][Final]: {event.transcript}")
                self.transcription_handler.handle_user_transcript(event.transcript)
                self.reset_are_you_there_flag()
            else:
                logger.info(f"[Human][Partial]: {event.transcript}")

        @current_session.on(LiveKitPipelineEvent.FUNCTION_TOOLS_EXECUTED.value)
        def on_tools_executed(tools):
            logger.info(f"=========> Function tools executed: {tools}")

        @current_session.on(LiveKitPipelineEvent.METRICS_COLLECTED.value)
        def on_metrics_collected(event):
            metrics_msg = format_metrics(event.metrics.type, event.metrics, self.room_name)

            metric_attributes = {
                "metrics.room_name": self.room_name,
                "metrics.metric_type": event.metrics.type,
                "metrics.is_outbound": self.is_outbound,
                "metrics.is_web_call": self.is_web_call,
                "metrics.primary_caller": self.primary_caller.identity
                if self.primary_caller
                else "unknown",
            }

            if event.metrics.type == "eou_metrics":
                metric_attributes["metrics.eou_latency_seconds"] = event.metrics.end_of_utterance_delay
            elif event.metrics.type == "llm_metrics":
                metric_attributes["metrics.llm_latency_seconds"] = event.metrics.ttft
            elif event.metrics.type == "tts_metrics":
                metric_attributes["metrics.tts_latency_seconds"] = event.metrics.ttfb
            elif event.metrics.type == "stt_metrics":
                metric_attributes["metrics.stt_latency_seconds"] = getattr(event.metrics, "latency", 0)

            if event.metrics.type != "vad_metrics":
                logger.info(json.dumps(metrics_msg))

            self.calculate_total_latency(event)

        @current_session.on("user_state_changed")
        def on_user_state_changed(event):
            if event.new_state == "away" and self.is_outbound is False:
                logger.info(f"User state changed to: {event.new_state} in room: {self.room_name}")
                if not self.has_asked_are_you_there:
                    if self.redis_event_handler.ai_voice_enabled is True:
                        logger.info(
                            f"User inactive, asking 'Are you there?' in room: {self.room_name}"
                        )
                        self.has_asked_are_you_there = True
                        asyncio.create_task(self._ask_are_you_there())
                        self.timeout_task = asyncio.create_task(self._second_timeout())
                    else:
                        logger.info(
                            f"TTS is disabled, skipping 'Are you there?' check in room: {self.room_name}"
                        )
            else:
                logger.info(f"User state changed to: {event.new_state} in room: {self.room_name}")


# ============================================================================
# AGENT ENTRYPOINT
# ============================================================================

class AgentEntrypoint:
    """Main entry point for the LiveKit agent."""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.session = None
        self.redis_client = None
        self.prompt_loader = None
        self.system_prompt = None
        self.greeting = None
        self.company_phone_number = None
        self.prompt_id = None

        # CRM-related attributes
        self.call_id = None
        self.ticket_id = None
        self.lead_id = None
        self.customer_id = None
        self.job_id = None
        self.call_context = None
        self.organization_id = None
        self.multi_user_transcriber = None

        self._load_configuration()
        self._initialize_room_info()
        self._initialize_redis()

    def _load_configuration(self):
        """Load configuration from environment variables."""
        self.tts_provider = os.getenv("TTS_PROVIDER", "cartesia")
        self.cartesia_voice_id = os.getenv("CARTESIA_VOICE_ID")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.enable_tts = os.getenv("ENABLE_TTS", "true").lower() == "true"
        self.use_langgraph = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
        self.use_livekit_multi_agent = (
            os.getenv("USE_LIVEKIT_MULTI_AGENT", "false").lower() == "true"
        )
        self.allow_tools = os.getenv("ALLOW_TOOLS", "true").lower() == "true"
        self.skip_human = os.getenv("SKIP_HUMAN", "false").lower() == "true"
        self.use_multi_user_transcript = (
            os.getenv("USE_MULTI_USER_TRANSCRIPT", "false").lower() == "true"
        )

    def _initialize_room_info(self):
        """Initialize room information."""
        self.room_name = self.ctx.room.name
        self.is_outbound = self.room_name.startswith("outbound")
        self.is_web_call = self.room_name.startswith("web")

        self.default_company_phone_number = Utils.get_company_number_from_room_name(
            self.ctx.room.name
        )
        self.default_caller_number = Utils.get_caller_number_from_room_name(self.ctx.room.name)

        if callable(self.room_name) and not isinstance(self.room_name, str):
            random_number = str(random.randint(100000, 999999))
            self.room_name = f"web-{self.default_company_phone_number}-{random_number}"

    def _initialize_redis(self):
        """Initialize Redis client and event handler."""
        self.redis_client = Utils.get_redis_client()
        self.redis_event_handler = RedisEventHandler(
            self.room_name,
            self.redis_client,
            self.ctx.room,
            cleanup_callback=self.cleanup_and_shutdown,
        )

    @staticmethod
    def prewarm(proc: JobProcess):
        """Preload the VAD model for better performance."""
        proc.userdata["vad"] = silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.55,
            prefix_padding_duration=0.5,
            max_buffered_speech=60.0,
            activation_threshold=0.5,
        )

    def get_room_input_options(self) -> RoomInputOptions:
        """Get room input options configuration."""
        async def custom_text_input_cb(sess: AgentSession, ev: TextInputEvent) -> None:
            logger.info(f"[Human] (text input): {ev.text}")
            try:
                self.transcription_handler.handle_user_transcript(ev.text)
                await sess.generate_reply(user_input=ev.text)
            except Exception as e:
                logger.error(f"Error generating reply: {e}")

        nc = self._get_noise_cancellation()

        return RoomInputOptions(
            text_enabled=True,
            noise_cancellation=nc,
            video_enabled=False,
            audio_sample_rate=24000,
            audio_num_channels=1,
            text_input_cb=custom_text_input_cb,
        )

    def _get_noise_cancellation(self):
        """Get noise cancellation configuration."""
        try:
            if os.getenv("ENABLE_NOISE_CANCELLATION", "true").lower() == "true":
                if os.getenv("USE_BVC_TELEPHONY", "true").lower() == "true":
                    logger.info("=========> Using BVC Telephony")
                    return noise_cancellation.BVCTelephony()
                else:
                    logger.info("=========> Using BVC")
                    return noise_cancellation.BVC()
        except Exception as e:
            logger.error(f"Failed to initialize noise cancellation: {e}")

        return None

    @staticmethod
    def get_room_output_options(enable_tts: bool = True) -> RoomOutputOptions:
        """Get the default room output options configuration."""
        return RoomOutputOptions(
            audio_enabled=enable_tts, audio_sample_rate=24000, audio_num_channels=1
        )

    def get_llm(self):
        """Get the appropriate LLM based on configuration."""
        if self.is_outbound:
            logger.info("=========> Outbound agent, using MinimalLLM")
            return MinimalLLM()

        llm_service = os.getenv("LLM_SERVICE", "openai").lower()
        logger.info(f"Inbound Agent - LLM service: {llm_service}")

        llm_providers = {
            "openai": lambda: openai.LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o")),
            "llama": lambda: groq.LLM(model=os.getenv("GROQ_MODEL", "llama3-8b-8192")),
            "google": lambda: google.LLM(model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite")),
            "deepseek": lambda: groq.LLM(model=os.getenv("GROQ_MODEL", "deepseek-chat")),
        }

        provider = llm_providers.get(llm_service)
        if provider:
            return provider()

        # Default to OpenAI if unknown service
        logger.warning(f"Unknown LLM service: {llm_service}, defaulting to OpenAI")
        return openai.LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    def get_stt(self):
        """Get the appropriate STT based on configuration."""
        stt_provider = os.getenv("STT_PROVIDER", "deepgram").lower()

        if stt_provider == "cartesia":
            model = os.getenv("CARTESIA_STT_MODEL", "ink-whisper")
            logger.info(f"=========> Initializing Cartesia STT with model: {model}")
            return cartesia.STT(model=model)

        elif stt_provider == "assemblyai":
            logger.info("=========> Initializing AssemblyAI STT")
            return assemblyai.STT()

        elif stt_provider == "openai":
            return self._get_openai_stt()

        elif stt_provider == "deepgram_flux":
            return self._get_deepgram_flux_stt()

        else:
            model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
            language = os.getenv("DEEPGRAM_STT_LANGUAGE", "en-US")
            logger.info(f"=========> Initializing Deepgram STT: {model}")
            return deepgram.STT(model=model, language=language)

    def _get_openai_stt(self):
        """Get OpenAI STT configuration."""
        use_realtime = os.getenv("OPENAI_STT_USE_REALTIME", "false").lower() == "true"
        model = (
            os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
            if use_realtime
            else os.getenv("OPENAI_STT_MODEL", "whisper-1")
        )

        language = os.getenv("OPENAI_STT_LANGUAGE", "en")
        base_url = os.getenv("OPENAI_STT_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_STT_API_KEY") or os.getenv("OPENAI_API_KEY")

        logger.info(
            f"=========> Initializing OpenAI STT with model: {model}, "
            f"base_url: {base_url}, use_realtime: {use_realtime}"
        )
        return openai.STT(
            model=model,
            language=language,
            base_url=base_url,
            api_key=api_key,
            use_realtime=use_realtime,
        )

    def _get_deepgram_flux_stt(self):
        """Get Deepgram Flux STT configuration."""
        logger.info("=========> Initializing Deepgram Flux STT")
        if hasattr(deepgram, "STTv2"):
            return deepgram.STTv2(model="flux-general-en", eager_eot_threshold=0.4)
        else:
            logger.info("=========> STTv2 not found, falling back to STT")
            model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
            language = os.getenv("DEEPGRAM_STT_LANGUAGE", "en-US")
            return deepgram.STT(model=model, language=language)

    def get_tts(self):
        """Get the appropriate TTS based on configuration."""
        try:
            logger.info(f"=========> Initializing TTS with provider: {self.tts_provider}")
            if self.is_outbound:
                logger.info("=========> Outbound agent, skipping TTS")
                return MinimalTTS()

            if self.tts_provider == "cartesia":
                return self._get_cartesia_tts()
            elif self.tts_provider == "eleven_labs":
                return self._get_elevenlabs_tts()

        except Exception as e:
            logger.error(f"=========> Error creating TTS: {e}")
            raise

    def _get_cartesia_tts(self):
        """Get Cartesia TTS configuration."""
        logger.info("=========> Initializing Cartesia TTS")

        api_key = os.getenv("CARTESIA_API_KEY")
        if not api_key:
            logger.error("=========> CARTESIA_API_KEY environment variable is not set")
            raise ValueError("CARTESIA_API_KEY environment variable is required")

        cartesia_model = os.getenv("CARTESIA_MODEL", "sonic-2")
        logger.info("=========> Cartesia TTS Configuration:")
        logger.info(f"  Model: {cartesia_model}")
        logger.info(f"  Voice ID: {self.cartesia_voice_id}")
        logger.info(f"  API Key Length: {len(api_key) if api_key else 0}")

        return cartesia.TTS(
            model=cartesia_model,
            voice=self.cartesia_voice_id,
            speed="normal",
            api_key=api_key,
            sample_rate=44100,
        )

    def _get_elevenlabs_tts(self):
        """Get ElevenLabs TTS configuration."""
        logger.info("=========> Initializing ElevenLabs TTS")

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.error("=========> ELEVENLABS_API_KEY environment variable is not set")
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
        return elevenlabs.TTS(
            api_key=api_key,
            voice_id=self.elevenlabs_voice_id,
            model=elevenlabs_model,
        )

    def setup_session_event_handlers(self, current_session):
        """Setup all event handlers for the session."""
        self.session_event_handler = SessionEventHandler(
            self.room_name,
            self.transcription_handler,
            self.redis_event_handler,
            primary_caller=self.primary_caller,
            cleanup_callback=self.cleanup_and_shutdown,
            is_outbound=self.is_outbound,
            is_web_call=self.is_web_call,
            user_away_timeout=self.user_away_timeout,
            use_multi_user_transcript=self.use_multi_user_transcript,
        )
        self.session_event_handler.setup_session_event_handlers(current_session)

    def get_span_name(self):
        """Get the span name for tracing."""
        return f"start_session_{self.room_name}"

    def _handle_participant_disconnect(self, participant):
        """Handle participant disconnection and shutdown room if needed."""
        self.redis_event_handler.on_participant_disconnect(participant)

        if (
            hasattr(self, "primary_caller")
            and participant.identity == self.primary_caller.identity
        ):
            logger.info(f"Primary caller {participant.identity} disconnected, shutting down room")
            asyncio.create_task(self.cleanup_and_shutdown("Primary caller disconnected"))
        else:
            logger.info(
                f"Non-primary participant {participant.identity} disconnected, continuing session"
            )

    async def handle_ttft(self, event: MetricsEvent):
        """Handle Time To First Token (TTFT) events."""
        logger.info(f"=========> TTFT for room {self.room_name}: {event.metrics.ttft:.3f}s")
        self.session_event_handler.calculate_total_latency(event)

    async def start_session(self, _span=None):
        """Start the agent session."""
        logger.info(f"=========> room_name: {self.room_name}")

        self.session_start_time = time.time()

        await self._connect_to_room()

        if self.is_outbound:
            self.primary_caller = await self.dial_and_wait_for_participant()
        else:
            self.primary_caller = await self.ctx.wait_for_participant()

        self.redis_event_handler.primary_caller = self.primary_caller

        await self._initialize_session_components()
        await self._setup_room_event_handlers()
        await self._initialize_multi_user_transcription()

        session = await self._create_agent_session()

        await self._initialize_crm()
        await self._initialize_transcription_handler()
        await self._setup_agent()

        if not self.is_outbound:
            await self._publish_room_started_event()

        await self._setup_room_listeners()

        self.call_answered_events = await self.get_call_answered_events()

        if not self.is_outbound and not self.skip_human:
            logger.info("Waiting for human agent to join room")
            await Utils.wait_for_human_agent(self.ctx)

        if self.use_multi_user_transcript:
            await self._setup_multi_user_transcription()

        await self._start_session_with_audio(session)

    async def _connect_to_room(self):
        """Connect to the LiveKit room."""
        try:
            logger.info("ctx.connect starts")
            start_time = time.time()
            await self.ctx.connect()
            duration = time.time() - start_time
            logger.info(f"ctx.connect completed in {duration:.3f} seconds")

        except Exception as e:
            logger.error(f"Failed to connect to LiveKit room: {str(e)}")
            error_msg = self._get_connection_error_message(e)
            self.cleanup_and_shutdown(error_msg)
            return None

    def _get_connection_error_message(self, error: Exception) -> str:
        """Get appropriate error message for connection failures."""
        error_str = str(error)
        if "token is expired" in error_str:
            return "Connection failed: Authentication token has expired. Please try reconnecting."
        elif "401 Unauthorized" in error_str:
            return "Connection failed: Invalid authentication credentials."
        return f"Connection failed: {error_str}"

    async def _initialize_session_components(self):
        """Initialize session components like phone numbers and prompts."""
        trunk_phone = self.primary_caller.attributes.get("sip.trunkPhoneNumber")
        self.company_phone_number = (
            trunk_phone if trunk_phone and isinstance(trunk_phone, str)
            else self.default_company_phone_number
        )
        self.company_phone_number = self.company_phone_number.replace("+", "")

        caller_phone = self.primary_caller.attributes.get("sip.phoneNumber")
        self.caller_number = (
            caller_phone if caller_phone and isinstance(caller_phone, str)
            else self.default_caller_number
        )

        self.prompt_loader = PromptLoader(
            self.redis_client,
            self.company_phone_number,
            use_langgraph=self.use_langgraph,
            use_livekit_multi_agent=self.use_livekit_multi_agent,
        )
        self.system_prompt, self.greeting, self.agents = self.prompt_loader.load_prompt(
            self.is_outbound, self.prompt_id
        )

    async def _setup_room_event_handlers(self):
        """Setup room event handlers."""
        pass  # Placeholder for future implementation

    async def _initialize_multi_user_transcription(self):
        """Initialize multi-user transcription if enabled."""
        pass  # Placeholder for future implementation

    async def _create_agent_session(self) -> AgentSession:
        """Create and configure the agent session."""
        turn_detection = self._get_turn_detection_model()

        self.user_away_timeout = float(os.getenv("USER_AWAY_TIMEOUT", "60.0"))
        if self.is_web_call:
            self.user_away_timeout = self.user_away_timeout * 5

        preemptive_generation = True
        logger.info(f"------- Preemptive generation: {preemptive_generation}")

        return AgentSession(
            stt=self.get_stt(),
            llm=self.get_llm(),
            tts=self.get_tts(),
            vad=self.ctx.proc.userdata["vad"],
            turn_detection=turn_detection,
            max_endpointing_delay=float(os.getenv("MAX_ENDPOINTING_DELAY", "6.0")),
            min_endpointing_delay=float(os.getenv("MIN_ENDPOINTING_DELAY", "0.5")),
            user_away_timeout=self.user_away_timeout,
            min_interruption_duration=float(os.getenv("MIN_INTERRUPTION_DURATION", "0.5")),
            min_interruption_words=int(os.getenv("MIN_INTERRUPTION_WORDS", "0")),
            preemptive_generation=preemptive_generation,
            use_tts_aligned_transcript=False,
        )

    def _get_turn_detection_model(self):
        """Get turn detection model with fallback."""
        try:
            return EnglishModel()
        except Exception as e:
            logger.error(f"Failed to initialize English turn detection model: {e}")
            try:
                return MultilingualModel()
            except Exception as e2:
                logger.error(f"Failed to initialize fallback Multilingual model: {e2}")
                raise RuntimeError("Failed to initialize any turn detection model") from e2

    async def _initialize_crm(self):
        """Initialize CRM client and call records."""
        self.crm_api_client = CRMApiClient()

        if self.is_outbound:
            self.crm_init_outbound_call()
        else:
            await self.crm_init_inbound_call()

    async def _initialize_transcription_handler(self):
        """Initialize transcription handler."""
        self.lk_job_id = self.ctx.job.id
        self.transcription_handler = TranscriptionHandler(
            self.room_name, self.call_id, self.lk_job_id, self.redis_client
        )

    async def _setup_agent(self):
        """Setup the agent based on configuration."""
        if self.use_langgraph and not self.is_outbound:
            self.agent = LangGraphAgent(
                room_name=self.room_name,
                redis_client=self.redis_client,
                transcription_handler=self.transcription_handler,
                call_context=self.call_context,
            )
            await self.agent.initialize()
            self.agent.on_ttft(self.handle_ttft)

    async def _publish_room_started_event(self):
        """Publish room started event to Redis."""
        self.redis_event_handler.publish_event(
            f"{os.getenv('REDIS_EVENT_CHANNEL', 'livekit-events')}",
            LiveKitRoomEvent.ROOM_STARTED.value,
            {
                "room_name": str(self.room_name),
                "company_number": str(self.company_phone_number),
                "customer_number": str(self.caller_number),
                "direction": "inbound" if not self.is_outbound else "outbound",
                "call_id": str(self.call_id) if self.call_id else None,
                "lk_job_id": str(self.lk_job_id) if self.lk_job_id else None,
                "ticket_id": str(self.ticket_id) if self.ticket_id else None,
                "lead_id": str(self.lead_id) if self.lead_id else None,
                "customer_id": str(self.customer_id) if self.customer_id else None,
                "organization_id": str(self.organization_id) if self.organization_id else None,
            },
        )

    async def _setup_room_listeners(self):
        """Setup room event listeners."""
        self.ctx.room.on(
            LiveKitRoomEvent.PARTICIPANT_CONNECTED.value,
            lambda remote_participant: self.redis_event_handler.on_participant_connect(
                remote_participant,
                self.primary_caller,
                self.ticket_id,
                self.call_id,
                self.crm_api_client,
            ),
        )

        self.ctx.room.on(
            LiveKitRoomEvent.PARTICIPANT_DISCONNECTED.value,
            lambda participant: self._handle_participant_disconnect(participant),
        )

        self.ctx.room.on(
            LiveKitRoomEvent.TRACK_PUBLISHED.value,
            lambda publication, participant: self.redis_event_handler.on_track_published(
                publication, participant
            ),
        )

        await self.redis_event_handler.listen_to_redis_events()

    async def _setup_multi_user_transcription(self):
        """Setup multi-user transcription."""
        logger.info("Multi-user transcription is ENABLED")
        self.multi_user_transcriber = MultiUserTranscriber(
            self.ctx, self.transcription_handler, self.stt_engine
        )
        self.multi_user_transcriber.start()
        await self.multi_user_transcriber.setup_existing_participants()

    async def _start_session_with_audio(self, session: AgentSession):
        """Start session and handle audio setup."""
        enable_audio = self.enable_tts and not self.is_outbound

        await session.start(
            room=self.ctx.room,
            agent=self.agent if hasattr(self, 'agent') else None,
            room_input_options=self.get_room_input_options(),
            room_output_options=self.get_room_output_options(enable_audio),
        )

        await self._wait_for_audio_track_ready()
        await self._setup_background_audio(session)
        await self._play_greeting(session)

    async def _wait_for_audio_track_ready(self):
        """Wait for audio track to be properly published."""
        audio_track_ready = False
        max_wait_time = 5.0
        start_time = time.time()

        while not audio_track_ready and (time.time() - start_time) < max_wait_time:
            for track_pub in self.ctx.room.local_participant.track_publications.values():
                if (
                    track_pub.kind == rtc.TrackKind.KIND_AUDIO
                    and track_pub.track is not None
                    and hasattr(track_pub, "sid")
                    and track_pub.sid is not None
                ):
                    audio_track_ready = True
                    logger.info(f"Audio track ready: {track_pub.sid}")
                    break

            if not audio_track_ready:
                await asyncio.sleep(0.1)

        if not audio_track_ready:
            logger.warning("Audio track not ready after waiting, proceeding anyway")
        else:
            await asyncio.sleep(0.2)

    async def _setup_background_audio(self, session: AgentSession):
        """Setup background audio player."""
        if self.redis_event_handler.ai_voice_enabled and not self.is_outbound:
            self.background_audio_player = BackgroundAudioPlayer(
                ambient_sound=AudioConfig(
                    "audio/office-ambience-6322.mp3", volume=0.25, probability=1.0
                ),
                thinking_sound=AudioConfig(
                    BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.4, probability=0.5
                ),
            )

            logger.info("Starting background audio player with office ambiance and typing sounds")
            await self.background_audio_player.start(room=self.ctx.room, agent_session=session)
        else:
            self.background_audio_player = None
            logger.info("Background audio player is disabled or not initialized")

    async def _play_greeting(self, session: AgentSession):
        """Play greeting message."""
        if not self.is_outbound:
            await asyncio.sleep(0.75)
            await session.say(self.greeting)
        else:
            logger.info(f"Outbound call - skipping TTS greeting: {self.greeting}")

    async def crm_init_inbound_call(self):
        """Initialize CRM API client and create inbound call record."""
        self.call_context = None

        try:
            request = InitInboundCallRequest(
                company_phone_number=self.company_phone_number,
                caller_phone_number=self.caller_number,
            )
            self.call_context = await init_inbound_call(request)
            self.call_context["room_name"] = self.room_name

            self.call_id = self.call_context["call_id"]
            self.ticket_id = self.call_context["ticket_id"]
            self.lead_id = self.call_context["lead_id"]
            self.customer_id = self.call_context["customer_id"]
            self.job_id = self.call_context["job_id"]
            self.organization_id = self.call_context["organization_id"]

        except Exception as e:
            logger.error(f"Error retrieving call context: {e}")
            self._initialize_empty_call_context()

    def _initialize_empty_call_context(self):
        """Initialize empty call context on CRM failure."""
        self.call_id = None
        self.ticket_id = None
        self.lead_id = None
        self.customer_id = None
        self.job_id = None
        self.organization_id = None
        self.call_context = None

    def crm_init_outbound_call(self):
        """Initialize CRM API client and create outbound call record."""
        try:
            outbound_call = self.crm_api_client.initiate_outbound_call(
                ticket_id=self.ticket_id, notes="Outbound call initiated by AI agent"
            )

            self.call_id = outbound_call.get("callId")

            logger.info(
                f"Created outbound call: Call ID {self.call_id}, Ticket ID {self.ticket_id}"
            )

        except Exception as e:
            logger.error(f"Error creating outbound call: {e}")
            self.call_id = None
            self.ticket_id = None

    async def hangup_call(self):
        """Hang up the current call by deleting the room."""
        ctx = get_job_context()
        if ctx is None:
            return

        try:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
            logger.info(f"Successfully deleted room: {ctx.room.name}")
        except Exception as e:
            logger.info(
                f"Failed to delete room {ctx.room.name} during cleanup: {e}. "
                "Room deletion failed, but this is expected when connection is already closed"
            )

    async def get_call_answered_events(self):
        """Get a list of call_answered events from Redis."""
        call_answered_keys = self.redis_client.keys("livekit:call_answered:*")

        call_answered_events = []
        for key in call_answered_keys:
            try:
                call_answered_data = self.redis_client.get(key)
                if call_answered_data:
                    call_answered_event = json.loads(call_answered_data)
                    call_answered_events.append({"key": key, "data": call_answered_event})
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse call_answered event from key {key}: {e}")
                continue

        logger.info(f"Found {len(call_answered_events)} active call sessions")
        return call_answered_events

    async def cleanup_and_shutdown(self, reason: str = "") -> None:
        """Clean up resources and shutdown the agent."""
        logger.info(f"cleanup_and_shutdown - reason: {reason} (room:{self.ctx.room.name})")

        if hasattr(self, "session_start_time"):
            duration = time.time() - self.session_start_time
            await self._update_call_log(duration)

        await self._cleanup_call_answered_events()
        await self._cleanup_components()
        await self.hangup_call()

        self.redis_client.close()
        self.ctx.shutdown(reason)

    async def _update_call_log(self, duration: float):
        """Update call log in CRM."""
        if hasattr(self, "crm_api_client") and self.crm_api_client:
            asyncio.create_task(
                self.crm_api_client.update_call_log(
                    call_id=self.call_id,
                    call_status="Ended",
                    call_duration=duration,
                    notes=f"Call finished in {duration:.3f}s",
                )
            )
        else:
            logger.warning("CRM API client not found, skipping call log update")

    async def _cleanup_call_answered_events(self):
        """Clean up call answered events from Redis."""
        calls_answered_events = await self.get_call_answered_events()
        logger.info(
            f"Shutting down room {self.ctx.room.name} "
            f"(calls_answered_events count: {len(calls_answered_events)})"
        )

        for call_answered_event in calls_answered_events:
            if call_answered_event["data"].get("data", {}).get("roomName") == self.ctx.room.name:
                room_name = call_answered_event["data"]["data"]["roomName"]
                logger.info(
                    f"Deleting call_answered_event in room: {room_name}. "
                    f"Key: {call_answered_event['key']}"
                )
                self.redis_client.delete(call_answered_event["key"])

    async def _cleanup_components(self):
        """Clean up various agent components."""
        if hasattr(self, "background_audio_player") and self.background_audio_player is not None:
            await self.background_audio_player.aclose()

        if hasattr(self, "multi_user_transcriber") and self.multi_user_transcriber:
            await self.multi_user_transcriber.aclose()

        if hasattr(self, "transcription_handler") and self.transcription_handler:
            await self.transcription_handler.cleanup_transcriptions()

        if hasattr(self, "agent") and self.agent:
            if hasattr(self.agent, "off_ttft"):
                self.agent.off_ttft(self.handle_ttft)

            if hasattr(self.agent, "__aexit__"):
                try:
                    await self.agent.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error during agent cleanup: {e}", exc_info=True)

        for task in self.ctx._participant_tasks.values():
            task.cancel()


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the agent."""
    entrypoint_instance = AgentEntrypoint(ctx)
    await entrypoint_instance.start_session()


if __name__ == "__main__":
    import sys

    logger.info(f"AGENT_NAME: {os.getenv('AGENT_NAME')}")

    if len(sys.argv) == 1:
        sys.argv.append("dev")
        logger.info("No command specified, defaulting to 'dev' mode")

    os.environ["AGENT_NAME"] = os.getenv("INBOUND_AGENT_WORKER_NAME")

    async def run_startup_tests():
        """Run startup connection tests."""
        skip_connection_tests = (
            os.getenv("SKIP_STARTUP_CONNECTION_TESTS", "false").lower() == "true"
        )

        if skip_connection_tests:
            logger.info("Skipping startup connection tests (SKIP_STARTUP_CONNECTION_TESTS=true)")
            return

        logger.info("=========> Running startup connection tests...")
        try:
            connection_results = await run_startup_connection_tests()

            logger.info("=========> Connection test results:")
            for service, result in connection_results.items():
                status = "SUCCESS" if result.success else "FAILED"
                time_str = f" ({result.response_time:.3f}s)" if result.response_time else ""
                logger.info(f"  {status} {result.service_name}{time_str}")
                if not result.success and result.error:
                    logger.error(f"    Error: {result.error}")

            failed_services = [
                service for service, result in connection_results.items() if not result.success
            ]

            if failed_services:
                logger.error(
                    f"CRITICAL: Connection tests failed for services: {', '.join(failed_services)}"
                )
                logger.error("Agent will continue but may experience issues with failed services")

                for service, result in connection_results.items():
                    if not result.success:
                        logger.error(f"  {service}: {result.error}")
                        logger.error(f"    Details: {result.details}")
            else:
                logger.info("All startup connection tests passed successfully")

        except Exception as e:
            logger.error(f"Failed to run startup connection tests: {e}", exc_info=True)
            logger.error("Agent will continue but connection status is unknown")

    asyncio.run(run_startup_tests())

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME"),
            prewarm_fnc=AgentEntrypoint.prewarm,
            initialize_process_timeout=30,
            job_memory_warn_mb=float(os.getenv("MEMORY_WARN_MB", "2500")),
            job_memory_limit_mb=float(os.getenv("MEMORY_LIMIT_MB", "3000")),
        ),
    )