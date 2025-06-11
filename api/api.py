import asyncio
import platform
import secrets
import socket
import string
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from time import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from uuid import uuid4

import psutil
import pytz
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from litellm import model_list
from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, SwarmRouter
from swarms.utils.any_to_str import any_to_str
from swarms.utils.litellm_tokenizer import count_tokens
from swarms.utils.index import (
    format_dict_to_string,
    format_data_structure,
)

# Literal of output types
OutputType = Literal[
    "all",
    "final",
    "list",
    "dict",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".toml",
    "string",
    "str",
]

SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SpreadSheetSwarm",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "GroupChat",
    "MultiAgentRouter",
    "AutoSwarmBuilder",
    "HiearchicalSwarm",
    "auto",
    "MajorityVoting",
    "MALT",
    "DeepResearchSwarm",
]

# Use the OutputType for type annotations
output_type: OutputType


load_dotenv()

# Define rate limit parameters
RATE_LIMIT = 100  # Max requests
TIME_WINDOW = 60  # Time window in seconds

# In-memory store for tracking requests
request_counts = defaultdict(lambda: {"count": 0, "start_time": time()})


class AgentSpec(BaseModel):
    agent_name: Optional[str] = Field(
        # default=None,
        description="The unique name assigned to the agent, which identifies its role and functionality within the swarm.",
    )
    description: Optional[str] = Field(
        default=None,
        description="A detailed explanation of the agent's purpose, capabilities, and any specific tasks it is designed to perform.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="The initial instruction or context provided to the agent, guiding its behavior and responses during execution.",
    )
    model_name: Optional[str] = Field(
        default="gpt-4o-mini",
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini",
    )
    auto_generate_prompt: Optional[bool] = Field(
        default=False,
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements.",
    )
    max_tokens: Optional[int] = Field(
        default=8192,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses.",
    )
    role: Optional[str] = Field(
        default="worker",
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents.",
    )
    max_loops: Optional[int] = Field(
        default=1,
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary.",
    )
    tools_list_dictionary: Optional[List[Dict[Any, Any]]] = Field(
        default=None,
        description="A dictionary of tools that the agent can use to complete its task.",
    )
    mcp_url: Optional[str] = Field(
        default=None,
        description="The URL of the MCP server that the agent can use to complete its task.",
    )

    class Config:
        arbitrary_types_allowed = True


class AgentCompletion(BaseModel):
    agent_config: Optional[AgentSpec] = Field(
        None,
        description="The configuration of the agent to be completed.",
    )
    task: Optional[str] = Field(
        None, description="The task to be completed by the agent."
    )
    history: Optional[Union[Dict[Any, Any], List[Dict[str, str]]]] = Field(
        default=None,
        description="The history of the agent's previous tasks and responses. Can be either a dictionary or a list of message objects.",
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }


class Agents(BaseModel):
    """Configuration for a collection of agents that work together as a swarm to accomplish tasks."""

    agents: List[AgentSpec] = Field(
        description="A list containing the specifications of each agent that will participate in the swarm, detailing their roles and functionalities."
    )


class SwarmSpec(BaseModel):
    name: Optional[str] = Field(
        None,
        description="The name of the swarm, which serves as an identifier for the group of agents and their collective task.",
        max_length=100,
    )
    description: Optional[str] = Field(
        None,
        description="A comprehensive description of the swarm's objectives, capabilities, and intended outcomes.",
    )
    agents: Optional[List[AgentSpec]] = Field(
        None,
        description="A list of agents or specifications that define the agents participating in the swarm.",
    )
    max_loops: Optional[int] = Field(
        default=1,
        description="The maximum number of execution loops allowed for the swarm, enabling repeated processing if needed.",
    )
    swarm_type: Optional[SwarmType] = Field(
        None,
        description="The classification of the swarm, indicating its operational style and methodology.",
    )
    rearrange_flow: Optional[str] = Field(
        None,
        description="Instructions on how to rearrange the flow of tasks among agents, if applicable.",
    )
    task: Optional[str] = Field(
        None,
        description="The specific task or objective that the swarm is designed to accomplish.",
    )
    img: Optional[str] = Field(
        None,
        description="An optional image URL that may be associated with the swarm's task or representation.",
    )
    return_history: Optional[bool] = Field(
        True,
        description="A flag indicating whether the swarm should return its execution history along with the final output.",
    )
    rules: Optional[str] = Field(
        None,
        description="Guidelines or constraints that govern the behavior and interactions of the agents within the swarm.",
    )
    tasks: Optional[List[str]] = Field(
        None,
        description="A list of tasks that the swarm should complete.",
    )
    messages: Optional[Union[List[Dict[Any, Any]], Dict[Any, Any]]] = Field(
        None,
        description="A list of messages that the swarm should complete.",
    )
    stream: Optional[bool] = Field(
        False,
        description="A flag indicating whether the swarm should stream its output.",
    )
    service_tier: Optional[str] = Field(
        "standard",
        description="The service tier to use for processing. Options: 'standard' (default) or 'flex' for lower cost but slower processing.",
    )

    class Config:
        arbitrary_types_allowed = True


def generate_key(prefix: str = "swarms") -> str:
    """
    Generates an API key similar to OpenAI's format (sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX).

    Args:
        prefix (str): The prefix for the API key. Defaults to "sk".

    Returns:
        str: An API key string in format: prefix-<48 random characters>
    """
    # Create random string of letters and numbers
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(28))
    return f"{prefix}-{random_part}"


def rate_limiter(request: Request):
    client_ip = request.client.host
    current_time = time()
    client_data = request_counts[client_ip]

    # Reset count if time window has passed
    if current_time - client_data["start_time"] > TIME_WINDOW:
        client_data["count"] = 0
        client_data["start_time"] = current_time

    # Increment request count
    client_data["count"] += 1

    # Check if rate limit is exceeded
    if client_data["count"] > RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )


def check_model_name(model_name: str) -> None:
    if model_name not in model_list:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not available. Check https://litellm.io for available models.",
        )


async def capture_telemetry(request: Request) -> Dict[str, Any]:
    """
    Captures comprehensive telemetry data from incoming requests including:
    - Request metadata (method, path, headers)
    - Client information (IP, user agent string)
    - Server information (hostname, platform)
    - System metrics (CPU, memory)
    - Timing data

    Args:
        request (Request): The FastAPI request object

    Returns:
        Dict[str, Any]: Dictionary containing telemetry data
    """
    try:
        # Get request headers
        headers = dict(request.headers)
        user_agent_string = headers.get("user-agent", "")

        # Get client IP, handling potential proxies
        client_ip = request.client.host
        forwarded_for = headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0]

        # Basic system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        telemetry = {
            "request_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            # Request data
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            # Headers and user agent info
            "headers": headers,
            "user_agent": user_agent_string,
            # Server information
            "server": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor() or "unknown",
            },
            # System metrics
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
            },
        }

        return telemetry

    except Exception as e:
        logger.error(f"Error capturing telemetry: {str(e)}")
        return {
            "error": "Failed to capture complete telemetry",
            "timestamp": datetime.now(UTC).isoformat(),
        }


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Dependency to verify the API key.

    Args:
        x_api_key (str): The API key to verify from the request header

    Raises:
        HTTPException: If the API key is invalid or verification fails
    """
    try:
        # Placeholder for the removed Supabase check
        pass
    except HTTPException as e:
        raise HTTPException(
            status_code=403,
            detail=f"API key verification failed: {e.detail}. Please ensure you have a valid API key from https://swarms.world",
        )


def validate_swarm_spec(
    swarm_spec: SwarmSpec,
) -> tuple[str, Optional[List[str]]]:
    """
    Validates the swarm specification and returns the task(s) to be executed.

    Args:
        swarm_spec: The swarm specification to validate

    Returns:
        tuple containing:
            - task string to execute (or stringified messages)
            - list of tasks if batch processing, None otherwise

    Raises:
        HTTPException: If validation fails
    """
    # Early validation
    if not any([swarm_spec.task, swarm_spec.tasks, swarm_spec.messages]):
        raise HTTPException(
            status_code=400,
            detail="There is no task or tasks or messages provided. Please provide a valid task description to proceed.",
        )

    # Determine task/tasks
    task = None
    tasks = None

    if swarm_spec.task is not None:
        task = swarm_spec.task
    elif swarm_spec.messages is not None:
        task = format_data_structure(swarm_spec.messages)
    elif swarm_spec.task and swarm_spec.messages is not None:
        task = (
            f"{format_data_structure(swarm_spec.messages)} \n\n User: {swarm_spec.task}"
        )
    elif swarm_spec.tasks is not None:
        tasks = swarm_spec.tasks

    # Validate agents if present
    if swarm_spec.agents:
        for agent in swarm_spec.agents:
            check_model_name(agent.model_name)
            # Safely concatenate strings, handling None values
            prompt_parts = [
                agent.system_prompt or "",
                agent.description or "",
                agent.agent_name or "",
            ]
            combined_prompt = "".join(prompt_parts)
            count_and_validate_prompts(combined_prompt)

    return task, tasks


def count_and_validate_prompts(prompt: str) -> None:
    if count_tokens(prompt) > 200_000:
        raise HTTPException(
            status_code=400,
            detail="""
            Prompt is too long. Please provide a prompt that is less than 10000 tokens. 
            Upgrade to a higher tier to use longer prompts at https://swarms.world/account.
            If you are using a custom model, please check the token limit of the model.
            """,
        )


def add_up_all_inputs(swarm_spec: SwarmSpec) -> int:
    # Use sum() with generator expression for better performance
    return sum(
        count_tokens(x)
        for x in (
            swarm_spec.task,
            format_data_structure(swarm_spec.messages),
            format_data_structure(swarm_spec.tasks),
            swarm_spec.rules,
            swarm_spec.description,
            swarm_spec.name,
        )
        if x is not None
    )


def create_single_agent(agent_spec: Union[AgentSpec, dict]) -> Agent:
    """
    Creates a single agent.

    Args:
        agent_spec: Agent specification (either AgentSpec object or dict)

    Returns:
        Created Agent instance

    Raises:
        HTTPException: If agent creation fails
    """
    try:
        # Convert dict to AgentSpec if needed
        if isinstance(agent_spec, dict):
            agent_spec = AgentSpec(**agent_spec)

        # Validate required fields
        if not agent_spec.agent_name:
            raise ValueError("Agent name is required.")
        if not agent_spec.model_name:
            raise ValueError("Model name is required.")

        # Create the agent
        agent = Agent(
            agent_name=agent_spec.agent_name,
            description=agent_spec.description,
            system_prompt=agent_spec.system_prompt,
            model_name=agent_spec.model_name or "gpt-4o-mini",
            auto_generate_prompt=agent_spec.auto_generate_prompt or False,
            max_tokens=agent_spec.max_tokens or 8192,
            temperature=agent_spec.temperature or 0.5,
            role=agent_spec.role or "worker",
            max_loops=agent_spec.max_loops or 1,
            dynamic_temperature_enabled=True,
            tools_list_dictionary=agent_spec.tools_list_dictionary,
            output_type="str-all-except-first",
            mcp_url=agent_spec.mcp_url,
        )

        logger.info("Successfully created agent: {}", agent_spec.agent_name)
        return agent

    except ValueError as ve:
        logger.error(
            "Validation error for agent {}: {}",
            getattr(agent_spec, "agent_name", "unknown"),
            str(ve),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Agent validation error: {str(ve)}",
        )
    except Exception as e:
        logger.error(
            "Error creating agent {}: {}",
            getattr(agent_spec, "agent_name", "unknown"),
            str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}",
        )


def add_up_all_agent_inputs(agents: List[Agent]) -> int:
    """
    Calculate the total number of tokens across all agent inputs.

    Args:
        agents (List[Agent]): List of Agent objects to count tokens from

    Returns:
        int: Total number of tokens across all agent inputs

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> total_tokens = add_up_all_agent_inputs(agents)
    """
    return sum(count_tokens(agent.short_memory.get_str()) for agent in agents)


def create_swarm(swarm_spec: SwarmSpec, api_key: str):
    """
    Creates and executes a swarm based on the provided specification.

    Args:
        swarm_spec: The swarm specification
        api_key: API key for authentication and billing

    Returns:
        The swarm execution results

    Raises:
        HTTPException: If swarm creation or execution fails
    """
    try:
        # Validate the swarm spec

        task, tasks = validate_swarm_spec(swarm_spec)

        # Create agents in parallel if specified
        agents = []
        if swarm_spec.agents is not None:
            # Use ThreadPoolExecutor for parallel agent creation
            with ThreadPoolExecutor(
                max_workers=min(len(swarm_spec.agents), 10)
            ) as executor:
                # Submit all agent creation tasks
                future_to_agent = {
                    executor.submit(create_single_agent, agent_spec): agent_spec
                    for agent_spec in swarm_spec.agents
                }

                # Collect results as they complete
                for future in as_completed(future_to_agent):
                    agent_spec = future_to_agent[future]
                    try:
                        agent = future.result()
                        agents.append(agent)
                    except HTTPException:
                        # Re-raise HTTP exceptions with original status code
                        raise
                    except Exception as e:
                        logger.error(
                            "Error creating agent {}: {}",
                            getattr(agent_spec, "agent_name", "unknown"),
                            str(e),
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to create agent: {str(e)}",
                        )

        # Create and configure the swarm
        swarm = SwarmRouter(
            name=swarm_spec.name,
            description=swarm_spec.description,
            agents=agents,
            max_loops=swarm_spec.max_loops,
            swarm_type=swarm_spec.swarm_type,
            output_type="dict-all-except-first",
            return_entire_history=False,
            rules=swarm_spec.rules,
            rearrange_flow=swarm_spec.rearrange_flow,
        )

        total_input_tokens = add_up_all_agent_inputs(swarm.agents)
        logger.info("Total input tokens: {}", total_input_tokens)

        # Calculate costs and execute
        start_time = time()

        output = (
            swarm.run(task=task)
            if task is not None
            else (
                swarm.batch_run(tasks=tasks)
                if tasks is not None
                else swarm.run(task=task)
            )
        )

        output_tokens = count_tokens(any_to_str(output))

        usage_data = {
            "input_tokens": total_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_input_tokens + output_tokens,
        }

        # Calculate execution time and costs
        execution_time = time() - start_time

        # Calculate costs
        cost_info = calculate_swarm_cost(
            agents=agents,
            input_text=swarm_spec.task,
            execution_time=execution_time,
            agent_outputs=output,
            service_tier=swarm_spec.service_tier,
        )

        return output, usage_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating swarm: {}", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create swarm: {str(e)}",
        )


async def run_swarm_completion(
    swarm: SwarmSpec, x_api_key: str = None
) -> Dict[str, Any]:
    """
    Run a swarm with the specified task.
    """
    try:
        swarm_name = swarm.name
        agents = swarm.agents

        # Log start of swarm execution
        logger.info(f"Starting swarm {swarm_name} with {len(agents)} agents")

        # Create and run the swarm
        logger.debug(f"Creating swarm object for {swarm_name}")

        # Handle flex processing
        max_retries = 3 if swarm.service_tier == "flex" else 1

        start_time = time()

        for attempt in range(max_retries):
            try:
                result, usage_data = create_swarm(swarm, x_api_key)
                execution_time = time() - start_time
                break
            except HTTPException as e:
                if e.status_code == 429 and swarm.service_tier == "flex":
                    # Resource unavailable error in flex mode
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        backoff_time = (2**attempt) * 5  # 5, 10, 20 seconds
                        logger.info(
                            f"Resource unavailable, retrying in {backoff_time} seconds..."
                        )
                        await asyncio.sleep(backoff_time)
                        continue
                raise
            except Exception as e:
                logger.error(f"Error running swarm: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to run swarm: {e}",
                )

        logger.debug(f"Running swarm task: {swarm.task}")

        if swarm.swarm_type == "MALT":
            length_of_agents = 14
        else:
            length_of_agents = len(agents)

        # Job id
        job_id = generate_key()

        # Format the response
        response = {
            "job_id": job_id,
            "status": "success",
            "swarm_name": swarm_name,
            "description": swarm.description,
            "swarm_type": swarm.swarm_type,
            "output": result,
            "number_of_agents": length_of_agents,
            "service_tier": swarm.service_tier,
            "execution_time": execution_time,
            "usage": usage_data,
        }

        if swarm.tasks is not None:
            response["tasks"] = swarm.tasks

        if swarm.messages is not None:
            response["messages"] = swarm.messages

        return response

    except HTTPException as http_exc:
        logger.error("HTTPException occurred: {}", http_exc.detail)
        raise
    except Exception as e:
        logger.error("Error running swarm {}: {}", swarm_name, str(e))
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run swarm: {e}",
        )


def calculate_mcp_cost(mcp_url: Optional[str] = None) -> float:
    """
    Calculate the cost for using an MCP (Model Control Protocol) URL.

    Args:
        mcp_url (Optional[str]): The MCP URL to calculate cost for. If None, returns 0.

    Returns:
        float: The cost of using the MCP URL. Returns 0.0 if no URL provided,
              otherwise returns 0.1 credits per MCP call.

    Example:
        >>> cost = calculate_mcp_cost("https://example.com/mcp")
        >>> print(cost)
        0.1
    """
    if mcp_url is None:
        return 0.0
    return 0.1


def calculate_swarm_cost(
    agents: List[Any],
    input_text: str,
    execution_time: float,
    agent_outputs: Union[List[Dict[str, str]], str] = None,  # Update agent_outputs type
    service_tier: str = "standard",
) -> Dict[str, Any]:
    """
    Calculate the cost of running a swarm based on agents, tokens, and execution time.
    Includes system prompts, agent memory, and scaled output costs.

    Args:
        agents: List of agents used in the swarm
        input_text: The input task/prompt text
        execution_time: Time taken to execute in seconds
        agent_outputs: List of output texts from each agent or a list of dictionaries
        service_tier: The service tier being used ("standard" or "flex")

    Returns:
        Dict containing cost breakdown and total cost
    """
    # Base costs per unit (these could be moved to environment variables)
    COST_PER_AGENT = 0.01  # Base cost per agent
    COST_PER_1M_INPUT_TOKENS = 2.00  # Cost per 1M input tokens
    COST_PER_1M_OUTPUT_TOKENS = 4.50  # Cost per 1M output tokens

    # Flex processing discounts
    FLEX_INPUT_DISCOUNT = 0.25  # 75% discount for input tokens in flex mode
    FLEX_OUTPUT_DISCOUNT = 0.25  # 75% discount for output tokens in flex mode

    # Get current time in California timezone
    california_tz = pytz.timezone("America/Los_Angeles")
    current_time = datetime.now(california_tz)
    is_night_time = current_time.hour >= 20 or current_time.hour < 6  # 8 PM to 6 AM

    try:
        # Calculate input tokens for task
        task_tokens = count_tokens(input_text)

        # Calculate total input tokens including system prompts and memory for each agent
        total_input_tokens = 0
        total_output_tokens = 0
        per_agent_tokens = {}
        agent_cost = 0

        for i, agent in enumerate(agents):
            agent_input_tokens = task_tokens  # Base task tokens

            # Add system prompt tokens if present
            if agent.system_prompt:
                agent_input_tokens += count_tokens(agent.system_prompt)

            # Add memory tokens if available
            try:
                memory = agent.short_memory.return_history_as_string()
                if memory:
                    memory_tokens = count_tokens(str(memory))
                    agent_input_tokens += memory_tokens
            except Exception as e:
                logger.warning(
                    f"Could not get memory for agent {agent.agent_name}: {str(e)}"
                )

            # Calculate actual output tokens if available, otherwise estimate
            if agent_outputs:
                if isinstance(agent_outputs, list):
                    # Sum tokens for each dictionary's content
                    agent_output_tokens = sum(
                        count_tokens(message["content"]) for message in agent_outputs
                    )
                elif isinstance(agent_outputs, str):
                    agent_output_tokens = count_tokens(agent_outputs)
                elif isinstance(agent_outputs, dict):
                    agent_output_tokens = count_tokens(any_to_str(agent_outputs))
                else:
                    agent_output_tokens = any_to_str(agent_outputs)
            else:
                agent_output_tokens = int(
                    agent_input_tokens * 2.5
                )  # Estimated output tokens

            # Store per-agent token counts
            per_agent_tokens[agent.agent_name] = {
                "input_tokens": agent_input_tokens,
                "output_tokens": agent_output_tokens,
                "total_tokens": (agent_input_tokens + agent_output_tokens),
            }

            # Add to totals
            total_input_tokens += agent_input_tokens
            total_output_tokens += agent_output_tokens

        # Calculate costs (convert to millions of tokens)
        agent_cost = len(agents) * COST_PER_AGENT
        input_token_cost = (
            (total_input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS * len(agents)
        )
        output_token_cost = (
            (total_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS * len(agents)
        )

        # Apply flex processing discounts if applicable
        if service_tier == "flex":
            input_token_cost *= FLEX_INPUT_DISCOUNT
            output_token_cost *= FLEX_OUTPUT_DISCOUNT

        # Apply night time discount
        if is_night_time:
            input_token_cost *= 0.25  # 75% discount
            output_token_cost *= 0.25  # 75% discount

        # Calculate total cost
        total_cost = agent_cost + input_token_cost + output_token_cost

        output = {
            "cost_breakdown": {
                "agent_cost": round(agent_cost, 6),
                "input_token_cost": round(input_token_cost, 6),
                "output_token_cost": round(output_token_cost, 6),
                "token_counts": {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": (total_input_tokens + total_output_tokens),
                    "per_agent": per_agent_tokens,
                },
                "num_agents": len(agents),
                "execution_time_seconds": round(execution_time, 2),
                "service_tier": service_tier,
                "night_time_discount_applied": is_night_time,
            },
            "total_cost": round(total_cost, 6),
        }

        return output

    except Exception as e:
        logger.error(f"Error calculating swarm cost: {str(e)}")
        raise ValueError(f"Failed to calculate swarm cost: {str(e)}")


def calculate_agent_cost(
    agent: Agent,
    input_text: str,
    execution_time: float,
    agent_output: Union[Dict[str, str], str, List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Calculate the cost for a single agent based on its input, output, and execution time.

    Args:
        agent: The agent instance
        input_text: The input task/prompt text
        execution_time: Time taken to execute in seconds
        agent_output: Output from the agent (can be dict, string, or list of dicts)

    Returns:
        Dict containing cost breakdown and total cost for this agent
    """
    # Base costs per unit
    COST_PER_AGENT = 0.01  # Base cost per agent
    COST_PER_1M_INPUT_TOKENS = 2.00  # Cost per 1M input tokens
    COST_PER_1M_OUTPUT_TOKENS = 4.50  # Cost per 1M output tokens

    # Get current time in California timezone
    california_tz = pytz.timezone("America/Los_Angeles")
    current_time = datetime.now(california_tz)
    is_night_time = current_time.hour >= 20 or current_time.hour < 6  # 8 PM to 6 AM

    try:
        # Calculate input tokens
        input_tokens = count_tokens(input_text)  # Base task tokens

        # Add system prompt tokens if present
        if agent.system_prompt:
            input_tokens += count_tokens(agent.system_prompt)

        # Add memory tokens if available
        try:
            memory = agent.short_memory.return_history_as_string()
            if memory:
                memory_tokens = count_tokens(str(memory))
                input_tokens += memory_tokens
        except Exception as e:
            logger.warning(
                f"Could not get memory for agent {agent.agent_name}: {str(e)}"
            )

        # Calculate output tokens
        if agent_output:
            if isinstance(agent_output, list):
                # Sum tokens for each dictionary's content
                output_tokens = sum(
                    count_tokens(message["content"]) for message in agent_output
                )
            elif isinstance(agent_output, str):
                output_tokens = count_tokens(agent_output)
            elif isinstance(agent_output, dict):
                output_tokens = count_tokens(any_to_str(agent_output))
            else:
                output_tokens = count_tokens(any_to_str(agent_output))
        else:
            output_tokens = int(input_tokens * 2.5)  # Estimated output tokens

        # Calculate base costs (convert to millions of tokens)
        agent_base_cost = COST_PER_AGENT
        input_token_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
        output_token_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS

        # Apply discount during California night time hours
        if is_night_time:
            input_token_cost *= 0.25  # 75% discount
            output_token_cost *= 0.25  # 75% discount

        # Calculate total cost
        total_cost = agent_base_cost + input_token_cost + output_token_cost

        return {
            "agent_name": agent.agent_name,
            "cost_breakdown": {
                "agent_base_cost": round(agent_base_cost, 6),
                "input_token_cost": round(input_token_cost, 6),
                "output_token_cost": round(output_token_cost, 6),
                "token_counts": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "execution_time_seconds": round(execution_time, 2),
                "night_time_discount_applied": is_night_time,
            },
            "total_cost": round(total_cost, 6),
        }

    except Exception as e:
        logger.error(f"Error calculating agent cost: {str(e)}")
        raise ValueError(f"Failed to calculate agent cost: {str(e)}")


def agent_completion_validation(
    agent_completion: AgentCompletion, x_api_key: str
) -> str:
    """
    Validates an agent completion request and combines the prompt parts.

    Args:
        agent_completion (AgentCompletion): The agent completion request to validate
        x_api_key (str): The API key for logging requests

    Returns:
        str: The combined prompt string

    Raises:
        HTTPException: If agent name is missing or validation fails
    """
    history = (
        format_dict_to_string(agent_completion.history)
        if agent_completion.history
        else ""
    )

    # Combine all strings first to avoid multiple concatenations
    prompt_parts = [
        agent_completion.agent_config.system_prompt or "",
        agent_completion.task or "",
        agent_completion.agent_config.description or "",
        agent_completion.agent_config.agent_name or "",
        history,
    ]
    combined_prompt = "".join(prompt_parts)
    count_and_validate_prompts(combined_prompt)
    check_model_name(agent_completion.agent_config.model_name)

    return combined_prompt


def agent_usage_calculations(
    agent_completion: AgentCompletion,
    combined_prompt: str,
    result: any,
    x_api_key: str,
):
    """
    Calculates usage metrics for an agent completion request.

    Args:
        agent_completion (AgentCompletion): The agent completion request
        combined_prompt (str): The combined prompt string
        result (any): The result from the agent execution
        x_api_key (str): The API key for deducting credits

    Returns:
        Dict[str, Any]: Dictionary containing usage metrics including:
            - input_tokens (int): Number of input tokens
            - output_tokens (int): Number of output tokens
            - total_tokens (int): Total tokens used
            - mcp_url (float): Cost of MCP URL usage
    """
    if isinstance(result, dict):
        result = any_to_str(result)

    if isinstance(result, list):
        result = any_to_str(result)

    if isinstance(result, str):
        result = result

    input_tokens = count_tokens(
        text=combined_prompt,
        model=agent_completion.agent_config.model_name,
    )

    output_tokens = count_tokens(
        text=any_to_str(result),
        model=agent_completion.agent_config.model_name,
    )

    if agent_completion.agent_config.mcp_url is not None:
        mcp_cost = calculate_mcp_cost(agent_completion.agent_config.mcp_url)

    usage_data = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    # If mcp_url is not none then add it to the usage_data
    if agent_completion.agent_config.mcp_url is not None:
        usage_data["mcp_url"] = mcp_cost

    return usage_data


async def _run_agent_completion(
    agent_completion: AgentCompletion, x_api_key=Header(...)
) -> Dict[str, Any]:
    """
    Run an agent with the specified task.
    """
    try:
        agent_completion_validation(agent_completion, x_api_key)

        if agent_completion.history is not None:
            # Format the dictionary with keys and values on separate lines
            history_prompt = format_dict_to_string(agent_completion.history)
        else:
            history_prompt = ""

        # Create agent from the config
        agent = Agent(
            **agent_completion.agent_config.model_dump(),
            output_type="dict-all-except-first",
        )

        # Run the agent with the provided task
        if agent_completion.history is not None:
            result = agent.run(
                task=f"History: \n\n {history_prompt} \n\n Task: {agent_completion.task}"
            )
        else:
            result = agent.run(task=agent_completion.task)

        usage_data = agent_usage_calculations(
            agent_completion=agent_completion,
            combined_prompt=agent.short_memory.get_str(),
            result=result,
            x_api_key=x_api_key,
        )

        output = {
            "id": generate_key("agent"),
            "success": True,
            "name": agent.name,
            "description": agent_completion.agent_config.description,
            "temperature": agent.temperature,
            "outputs": result,
            "usage": usage_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return output
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        logger.error(f"Unexpected error running agent: {str(e)}")
        logger.exception(e)  # Log full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


async def batched_agent_completion(
    agent_completions: List[AgentCompletion],
    x_api_key: str,
):
    """
    Process multiple agent completions sequentially.

    Args:
        agent_completions (List[AgentCompletion]): List of agent completion tasks to process
        x_api_key (str): API key for authentication

    Returns:
        List[Dict[str, Any]]: List of results from completed agent tasks

    Raises:
        ValueError: If agent_completions is empty
        Exception: For any unexpected errors during batch processing
    """
    LIMIT_REQUESTS = 10

    if len(agent_completions) > LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ERROR: BATCH SIZE EXCEEDED - You can only run up to {LIMIT_REQUESTS} batch agents at a time. Please reduce your batch size and try again. Current batch size: {len(agent_completions)}",
        )

    if not agent_completions:
        raise ValueError("No agent completions provided")

    # Convert the list of AgentCompletion objects to a list of dictionaries
    agent_completions_logs = [
        agent_completion.model_dump() for agent_completion in agent_completions
    ]

    try:
        start_time = time()
        results = await asyncio.gather(
            *[
                asyncio.create_task(_run_agent_completion(agent_completion, x_api_key))
                for agent_completion in agent_completions
            ]
        )

        results = {
            "batch_id": generate_key("agent-batch"),
            "total_requests": len(agent_completions),
            "execution_time": time() - start_time,
            "timestamp": datetime.now(UTC).isoformat(),
            "results": results,
        }

        return results
    except Exception as e:
        logger.error(
            f"Error running agent batch: {str(e)} {traceback.format_exc()} Reconfigure your agent completion input schema "
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


async def get_swarm_types() -> List[str]:
    """Returns a list of available swarm types"""
    return [
        "AgentRearrange",
        "MixtureOfAgents",
        "SpreadSheetSwarm",
        "SequentialWorkflow",
        "ConcurrentWorkflow",
        "GroupChat",
        "MultiAgentRouter",
        "AutoSwarmBuilder",
        "HiearchicalSwarm",
        "auto",
        "MajorityVoting",
        "MALT",
        "DeepResearchSwarm",
    ]


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Swarm Agent API",
    description="API for managing and executing Python agents in the cloud without Docker/Kubernetes.",
    version="1.0.0",
    # debug=True,
)

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": (
            "Welcome to the Swarm API. Check out the docs at https://docs.swarms.world"
        )
    }


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """
    Middleware to capture telemetry for all requests and log to database
    """
    start_time = time()

    # Capture initial telemetry
    telemetry = await capture_telemetry(request)

    # Add request start time
    telemetry["request_timing"] = {
        "start_time": start_time,
        "start_timestamp": datetime.now(UTC).isoformat(),
    }

    # Store telemetry in request state for access in route handlers
    request.state.telemetry = telemetry

    try:
        # Process the request
        response = await call_next(request)

        # Calculate request duration
        duration = time() - start_time

        # Update telemetry with response data
        telemetry.update(
            {
                "response": {
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                }
            }
        )

        # Try to get API key from headers
        api_key = request.headers.get("x-api-key")

        # Log telemetry to database if we have an API key
        if api_key:
            try:
                # Placeholder for the removed Supabase log_api_request
                pass
            except Exception as e:
                logger.error(f"Failed to log telemetry to database: {str(e)}")

        return response

    except Exception as e:
        # Update telemetry with error information
        telemetry.update(
            {
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "duration_seconds": time() - start_time,
                }
            }
        )

        # Try to log error telemetry if we have an API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            try:
                # Placeholder for the removed Supabase log_api_request
                pass
            except Exception as log_error:
                logger.error(f"Failed to log error telemetry: {str(log_error)}")

        raise  # Re-raise the original exception


@app.get("/health", dependencies=[Depends(rate_limiter)])
def health():
    return {"status": "ok"}


@app.get(
    "/v1/swarms/available",
    dependencies=[Depends(verify_api_key), Depends(rate_limiter)],
)
async def check_swarm_types(
    x_api_key: str = Header(...),
) -> Dict[Any, Any]:
    """
    Check the available swarm types.
    """

    # Await
    swarm_types = await get_swarm_types()

    out = {
        "success": True,
        "swarm_types": swarm_types,
    }

    return out


@app.post(
    "/v1/swarm/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
async def run_swarm(swarm: SwarmSpec, x_api_key=Header(...)) -> Dict[str, Any]:
    """
    Run a swarm with the specified task.
    """
    try:
        return await run_swarm_completion(swarm, x_api_key)
    except Exception as e:
        logger.error(f"Error running swarm: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/v1/agent/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
async def run_agent(
    agent_completion: AgentCompletion, x_api_key=Header(...)
) -> Dict[str, Any]:
    """
    Run an agent with the specified task.
    """
    try:
        return await _run_agent_completion(agent_completion, x_api_key)
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/v1/agent/batch/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
async def run_agent_batch(
    agent_completions: List[AgentCompletion], x_api_key=Header(...)
) -> Dict[Any, Any]:
    """
    Run a batch of agents with the specified tasks using a thread pool.

    Args:
        agent_completions: List of agent completion tasks to process
        x_api_key: API key for authentication

    Returns:
        List[Dict[str, Any]]: List of results from completed agent tasks

    Raises:
        HTTPException: If there's an error processing the batch
    """
    try:
        # Process the batch with optimized concurrency
        return await batched_agent_completion(agent_completions, x_api_key)
    except Exception as e:
        logger.error(
            f"Error running agent batch: {str(e)} {traceback.format_exc()} Reconfigure your agent completion input schema "
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post(
    "/v1/swarm/batch/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
def run_batch_completions(
    swarms: List[SwarmSpec], x_api_key=Header(...)
) -> List[Dict[str, Any]]:
    """
    Run a batch of swarms with the specified tasks using a thread pool.
    """
    results = []

    def process_swarm(swarm):
        try:
            # Create and run the swarm directly
            result, usage_data = create_swarm(swarm, x_api_key)
            return {
                "status": "success",
                "swarm_name": swarm.name,
                "result": result,
                "usage": usage_data,
            }
        except HTTPException as http_exc:
            logger.error("HTTPException occurred: {}", http_exc.detail)
            return {
                "status": "error",
                "swarm_name": swarm.name,
                "detail": http_exc.detail,
            }
        except Exception as e:
            logger.error("Error running swarm {}: {}", swarm.name, str(e))
            logger.exception(e)
            return {
                "status": "error",
                "swarm_name": swarm.name,
                "detail": f"Failed to run swarm: {str(e)}",
            }

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=min(len(swarms), 10)) as executor:
        # Submit all swarms to the thread pool
        future_to_swarm = {
            executor.submit(process_swarm, swarm): swarm for swarm in swarms
        }

        # Collect results as they complete
        for future in as_completed(future_to_swarm):
            result = future.result()
            results.append(result)

    return results


# Add this new endpoint
@app.get(
    "/v1/swarm/logs",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
async def get_logs(x_api_key: str = Header(...)) -> Dict[str, Any]:
    """
    Get all API request logs for the user associated with the provided API key,
    excluding any logs that contain a client_ip field in their data.
    """
    try:
        # Placeholder for the removed Supabase get_user_logs
        logs = []
        return {
            "status": "success",
            "count": len(logs),
            "logs": logs,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in get_logs endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in get_logs endpoint: {str(e)}",
        )


@app.get(
    "/v1/models/available",
    dependencies=[
        Depends(verify_api_key),
        Depends(rate_limiter),
    ],
)
async def get_available_models(
    x_api_key: str = Header(...),
) -> Dict[str, Any]:
    """
    Get all available models.
    """
    out = {
        "success": True,
        "models": model_list,
    }
    return out


# --- Main Entrypoint ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        access_log=True,
    )
