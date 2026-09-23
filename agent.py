"""AutoGen agent layer: connects GPT-4o to Notion + Time MCP servers over stdio."""

import os

import jsonref
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.tools import ParametersSchema, ToolSchema
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioMcpToolAdapter, StdioServerParams, create_mcp_server_session
from dotenv import load_dotenv
from mcp import Tool

load_dotenv()

SYSTEM_MESSAGE = (
    "You are a helpful assistant that manages a Notion workspace using the provided MCP tools. "
    "Use the Notion tools to search, read, create and update pages and databases. "
    "Use the time tools when the task involves the current date or time. "
    "When the task is complete, give a short summary of what you did and end with the word TERMINATE."
)


def notion_server_params() -> StdioServerParams:
    token = os.environ["NOTION_API_KEY"]
    return StdioServerParams(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={
            **os.environ,
            # Newer server versions read NOTION_TOKEN; older ones read OPENAPI_MCP_HEADERS.
            "NOTION_TOKEN": token,
            "OPENAPI_MCP_HEADERS": (
                f'{{"Authorization": "Bearer {token}", "Notion-Version": "2022-06-28"}}'
            ),
        },
        read_timeout_seconds=60,
    )


def _inline_schema(schema: dict) -> dict:
    """Resolve $refs and turn oneOf into anyOf so the schema is self-contained for OpenAI."""

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = {("anyOf" if k == "oneOf" else k): walk(v) for k, v in node.items()}
        node.pop("$defs", None)
        return node

    return walk(jsonref.replace_refs(schema, proxies=False, lazy_load=False))


class RawSchemaMcpTool(StdioMcpToolAdapter):
    """MCP tool that sends the server's raw JSON schema to the model and passes arguments through as-is.

    AutoGen's default adapter converts the schema to a pydantic model, which drops nested
    `$ref`/`oneOf` fields (e.g. Notion's `parent.page_id`) and then strips them from the arguments.
    """

    def __init__(self, server_params: StdioServerParams, tool: Tool) -> None:
        # Give the base class a trivial schema so its pydantic conversion can't fail.
        super().__init__(server_params, tool.model_copy(update={"inputSchema": {"type": "object"}}))
        self._tool = tool
        self._raw_schema = _inline_schema(tool.inputSchema)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=ParametersSchema(
                type="object",
                properties=self._raw_schema.get("properties", {}),
                required=self._raw_schema.get("required", []),
            ),
        )

    async def run_json(self, args, cancellation_token: CancellationToken, call_id: str | None = None):
        async with create_mcp_server_session(self._server_params) as session:
            await session.initialize()
            return await self._run(args=dict(args), cancellation_token=cancellation_token, session=session)


async def load_mcp_tools(server_params: StdioServerParams) -> list[RawSchemaMcpTool]:
    async with create_mcp_server_session(server_params) as session:
        await session.initialize()
        result = await session.list_tools()
    return [RawSchemaMcpTool(server_params, tool) for tool in result.tools]


def time_server_params() -> StdioServerParams:
    return StdioServerParams(
        command="uvx",
        args=["mcp-server-time", f"--local-timezone={os.getenv('LOCAL_TIMEZONE', 'UTC')}"],
        read_timeout_seconds=30,
    )


async def setup_team() -> RoundRobinGroupChat:
    tools = await load_mcp_tools(notion_server_params())
    if os.getenv("ENABLE_TIME_MCP", "true").lower() == "true":
        tools += await load_mcp_tools(time_server_params())

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    agent = AssistantAgent(
        name="notion_agent",
        model_client=model_client,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=True,
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(15)
    return RoundRobinGroupChat([agent], termination_condition=termination)


async def run_task(task: str) -> str:
    team = await setup_team()
    result = await team.run(task=task)
    # Return the last text message from the agent, minus the termination keyword.
    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and message.source != "user":
            return content.replace("TERMINATE", "").strip()
    return "No response from agent."


if __name__ == "__main__":
    import asyncio
    import sys

    prompt = " ".join(sys.argv[1:]) or "What pages are in my Notion workspace?"
    print(asyncio.run(run_task(prompt)))
