"""
Base agent class and communication protocol.

All agents (HypothesisGenerator, ExperimentDesigner, DataAnalyst, etc.) inherit from this base.
"""

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent lifecycle status."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class AgentMessage(BaseModel):
    """
    Message for inter-agent communication.

    Example:
        ```python
        msg = AgentMessage(
            type=MessageType.REQUEST,
            from_agent="hypothesis_generator",
            to_agent="literature_analyzer",
            content={"query": "dark matter papers"},
            correlation_id="exp-123"
        )
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    from_agent: str
    to_agent: str
    content: dict[str, Any]
    correlation_id: str | None = None  # For tracking request/response pairs
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AgentState(BaseModel):
    """Agent state for persistence."""

    agent_id: str
    agent_type: str
    status: AgentStatus
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BaseAgent:
    """
    Base class for all Kosmos agents.

    Provides:
    - Lifecycle management (start, stop, pause, resume)
    - Message passing for inter-agent communication
    - State persistence
    - Health checks
    - Logging

    Subclasses should implement:
    - process_message(): Handle incoming messages
    - execute(): Main agent logic
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_type: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent ID (auto-generated if None)
            agent_type: Agent type name (uses class name if None)
            config: Optional configuration dict
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type or self.__class__.__name__
        self.config = config or {}

        self.status = AgentStatus.CREATED
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Message handling
        self.message_queue: list[AgentMessage] = []
        self.message_handlers: dict[str, Callable] = {}
        self._message_router: Callable[[AgentMessage], None] | None = None

        # State management
        self.state_data: dict[str, Any] = {}

        # Statistics
        self.messages_received = 0
        self.messages_sent = 0
        self.tasks_completed = 0
        self.errors_encountered = 0

        logger.info(f"Agent {self.agent_type} ({self.agent_id}) created")

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    def start(self):
        """Start the agent."""
        if self.status != AgentStatus.CREATED:
            logger.warning(f"Agent {self.agent_id} already started")
            return

        self.status = AgentStatus.STARTING
        logger.info(f"Starting agent {self.agent_id}")

        try:
            self._on_start()
            self.status = AgentStatus.RUNNING
            logger.info(f"Agent {self.agent_id} started successfully")
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Failed to start agent {self.agent_id}: {e}")
            raise

    def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")

        try:
            self._on_stop()
            self.status = AgentStatus.STOPPED
            logger.info(f"Agent {self.agent_id} stopped")
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}: {e}")
            raise

    def pause(self):
        """Pause the agent."""
        if self.status != AgentStatus.RUNNING:
            logger.warning(f"Cannot pause agent {self.agent_id} in status {self.status}")
            return

        self.status = AgentStatus.PAUSED
        logger.info(f"Agent {self.agent_id} paused")

    def resume(self):
        """Resume the agent."""
        if self.status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent {self.agent_id} in status {self.status}")
            return

        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.agent_id} resumed")

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self.status == AgentStatus.RUNNING

    def is_healthy(self) -> bool:
        """
        Health check.

        Subclasses can override for custom health checks.
        """
        return self.status in [AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.WORKING]

    def get_status(self) -> dict[str, Any]:
        """
        Get agent status information.

        Returns:
            dict: Status including state, statistics, health
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "is_healthy": self.is_healthy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "statistics": {
                "messages_received": self.messages_received,
                "messages_sent": self.messages_sent,
                "tasks_completed": self.tasks_completed,
                "errors_encountered": self.errors_encountered,
            },
            "message_queue_length": len(self.message_queue),
        }

    # ========================================================================
    # MESSAGE PASSING
    # ========================================================================

    def send_message(
        self,
        to_agent: str,
        content: dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        correlation_id: str | None = None,
    ) -> AgentMessage:
        """
        Send message to another agent.

        Args:
            to_agent: Target agent ID
            content: Message content
            message_type: Type of message
            correlation_id: Optional correlation ID

        Returns:
            AgentMessage: The sent message
        """
        message = AgentMessage(
            type=message_type,
            from_agent=self.agent_id,
            to_agent=to_agent,
            content=content,
            correlation_id=correlation_id,
        )

        self.messages_sent += 1
        logger.debug(f"Agent {self.agent_id} sending message to {to_agent}: {message_type}")

        # Use message router if available to actually deliver the message
        if self._message_router is not None:
            try:
                self._message_router(message)
                logger.debug(f"Message routed from {self.agent_id} to {to_agent}")
            except Exception as e:
                logger.error(f"Failed to route message from {self.agent_id} to {to_agent}: {e}")

        return message

    def receive_message(self, message: AgentMessage):
        """
        Receive message from another agent.

        Args:
            message: Incoming message
        """
        self.messages_received += 1
        self.message_queue.append(message)

        logger.debug(
            f"Agent {self.agent_id} received message from {message.from_agent}: {message.type}"
        )

        # Process message
        try:
            self.process_message(message)
        except Exception as e:
            self.errors_encountered += 1
            logger.error(f"Error processing message in {self.agent_id}: {e}")
            # Send error response
            if message.type == MessageType.REQUEST:
                self.send_message(
                    to_agent=message.from_agent,
                    content={"error": str(e)},
                    message_type=MessageType.ERROR,
                    correlation_id=message.id,
                )

    def process_message(self, message: AgentMessage):
        """
        Process incoming message.

        Subclasses should override this to implement message handling logic.

        Args:
            message: Message to process
        """
        logger.warning(
            f"Agent {self.agent_id} received message but process_message() not implemented"
        )

    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register handler for specific message type.

        Args:
            message_type: Type of message to handle
            handler: Callable that takes AgentMessage
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type} in agent {self.agent_id}")

    def set_message_router(self, router: Callable[[AgentMessage], None]):
        """
        Set the message router callback for delivering messages.

        When set, send_message() will use this callback to actually deliver
        messages to their target agents. This enables proper message routing
        through a central registry or message broker.

        Args:
            router: Callable that takes an AgentMessage and delivers it
        """
        self._message_router = router
        logger.debug(f"Message router set for agent {self.agent_id}")

    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================

    def get_state(self) -> AgentState:
        """
        Get current agent state for persistence.

        Returns:
            AgentState: Current state
        """
        self.updated_at = datetime.utcnow()
        return AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=self.status,
            data=self.state_data,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def restore_state(self, state: AgentState):
        """
        Restore agent from saved state.

        Args:
            state: Saved state to restore
        """
        self.agent_id = state.agent_id
        self.agent_type = state.agent_type
        self.status = state.status
        self.state_data = state.data
        self.created_at = state.created_at
        self.updated_at = state.updated_at

        logger.info(f"Restored agent {self.agent_id} from saved state")

    def save_state_data(self, key: str, value: Any):
        """Save data to agent state."""
        self.state_data[key] = value
        self.updated_at = datetime.utcnow()

    def get_state_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from agent state."""
        return self.state_data.get(key, default)

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Execute agent task.

        Subclasses should override this to implement main agent logic.

        Args:
            task: Task specification

        Returns:
            dict: Task result
        """
        raise NotImplementedError(f"execute() not implemented for {self.agent_type}")

    # ========================================================================
    # LIFECYCLE HOOKS (for subclasses to override)
    # ========================================================================

    def _on_start(self):
        """Hook called when agent starts. Override in subclasses."""
        pass

    def _on_stop(self):
        """Hook called when agent stops. Override in subclasses."""
        pass

    def _on_pause(self):
        """Hook called when agent pauses. Override in subclasses."""
        pass

    def _on_resume(self):
        """Hook called when agent resumes. Override in subclasses."""
        pass

    # ========================================================================
    # UTILITY
    # ========================================================================

    def __repr__(self) -> str:
        return f"<{self.agent_type} id={self.agent_id} status={self.status}>"

    def __str__(self) -> str:
        return f"{self.agent_type}({self.agent_id})"
