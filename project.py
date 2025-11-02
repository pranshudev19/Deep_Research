from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import sendgrid
import os
import requests
from sendgrid.helpers.mail import Mail, Email, To, Content
from typing import Dict, List
from IPython.display import display, Markdown
from openai import AsyncOpenAI
import gradio as gr

load_dotenv(override=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GEMINI_API_KEY")
if google_api_key:
    print("Google api key is present!!")
else:
    print("Error!")

gemini_client = AsyncOpenAI(base_url = GEMINI_BASE_URL, api_key = google_api_key)

gemini_model = OpenAIChatCompletionsModel(model = 'gemini-2.5-flash', openai_client = gemini_client)

load_dotenv(override=True)


# ==================== CUSTOM SERPAPI TOOL ====================

@function_tool
def serpapi_search(query: str) -> str:
    """
    Performs a web search using SerpAPI and returns formatted results.
    
    Args:
        query: The search query string
        
    Returns:
        Formatted string containing search results with titles, snippets, and links
    """
    params = {
        "q": query,
        "api_key": os.environ.get('SERPAPI_API_KEY'),
        "engine": "google",
        "num": 10,  # Number of results to return
    }
    
    # Make request to SerpAPI
    response = requests.get("https://serpapi.com/search", params=params)
    
    if response.status_code != 200:
        return f"Error: API request failed with status code {response.status_code}"
    
    results = response.json()
    
    # Format the results for the agent
    formatted_results = []
    
    if "organic_results" in results:
        for i, result in enumerate(results["organic_results"][:10], 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description available")
            link = result.get("link", "")
            
            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   {snippet}\n"
                f"   URL: {link}\n"
            )
    
    # Include answer box if available
    if "answer_box" in results:
        answer = results["answer_box"].get("answer", "")
        if answer:
            formatted_results.insert(0, f"**Direct Answer:** {answer}\n\n")
    
    # Include knowledge graph if available
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        title = kg.get("title", "")
        description = kg.get("description", "")
        if title and description:
            formatted_results.insert(0, f"**Knowledge Graph - {title}:** {description}\n\n")
    
    return "\n".join(formatted_results) if formatted_results else "No results found."


# ==================== SEARCH AGENT ====================

SEARCH_INSTRUCTIONS = """You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 
words. Capture the main points. Write succinctly, no need to have complete sentences or good 
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the 
essence and ignore any fluff. Do not include any additional commentary other than the summary itself."""

search_agent = Agent(
    name="Search agent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[serpapi_search],
    model=gemini_model,
    model_settings=ModelSettings(tool_choice="required"),
)


# ==================== PLANNER AGENT ====================

HOW_MANY_SEARCHES = 3

PLANNER_INSTRUCTIONS = f"""You are a helpful research assistant. Given a query, come up with a set of web searches 
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."""


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


planner_agent = Agent(
    name="PlannerAgent",
    instructions=PLANNER_INSTRUCTIONS,
    model=gemini_model,
    output_type=WebSearchPlan,
)


# ==================== EMAIL AGENT ====================

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send out an email with the given subject and HTML body"""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("your-email@example.com")  # Change this to your verified email
    to_email = To("recipient@example.com")  # Change this to recipient email
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    return {"status": "success", "code": response.status_code}


EMAIL_INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

email_agent = Agent(
    name="Email agent",
    instructions=EMAIL_INSTRUCTIONS,
    tools=[send_email],
    model=gemini_model,
)


# ==================== WRITER AGENT ====================

WRITER_INSTRUCTIONS = """You are a senior researcher tasked with writing a cohesive report for a research query. 
You will be provided with the original query, and some initial research done by a research assistant.

You should first come up with an outline for the report that describes the structure and 
flow of the report. Then, generate the report and return that as your final output.

The final output should be in markdown format, and it should be lengthy and detailed. Aim 
for 5-10 pages of content, at least 1000 words."""


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: List[str] = Field(description="Suggested topics to research further")


writer_agent = Agent(
    name="WriterAgent",
    instructions=WRITER_INSTRUCTIONS,
    model=gemini_model,
    output_type=ReportData,
)


# ==================== ORCHESTRATION FUNCTIONS ====================

async def plan_searches(query: str) -> WebSearchPlan:
    """Use the planner_agent to plan which searches to run for the query"""
    print("📋 Planning searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    print(f"✓ Will perform {len(result.final_output.searches)} searches")
    return result.final_output


async def perform_searches(search_plan: WebSearchPlan) -> List[str]:
    """Call search() for each item in the search plan"""
    print("🔍 Executing searches...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("✓ Finished searching")
    return results


async def search(item: WebSearchItem) -> str:
    """Use the search agent to run a web search for each item in the search plan"""
    input_text = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input_text)
    return result.final_output


async def write_report(query: str, search_results: List[str]) -> ReportData:
    """Use the writer agent to write a report based on the search results"""
    print("✍️  Synthesizing report...")
    input_text = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, input_text)
    print("✓ Finished writing report")
    return result.final_output


async def send_report_email(report: ReportData) -> ReportData:
    """Use the email agent to send an email with the report"""
    print("📧 Sending email...")
    result = await Runner.run(email_agent, report.markdown_report)
    print("✓ Email sent")
    return report


# ==================== MAIN EXECUTION ====================

async def run_research(query: str):
    """Main function to orchestrate the entire research process"""
    print(f"🚀 Starting research for: '{query}'")
    print("=" * 60)
    
    # Step 1: Plan searches
    search_plan = await plan_searches(query)
    
    # Step 2: Execute searches in parallel
    search_results = await perform_searches(search_plan)
    
    # Step 3: Write comprehensive report
    report = await write_report(query, search_results)
    
    # Step 4: Send email with report
    await send_report_email(report)
    
    print("=" * 60)
    print("🎉 Research complete!")
    
    return report


# ==================== EXAMPLE USAGE ====================

async def main():
    """Main entry point for running examples"""
    
    # Example 1: Run the full research pipeline
    query = "Latest AI Agent frameworks in 2025"
    report = await run_research(query)

    # Display the results
    print("\n" + "=" * 60)
    print("SHORT SUMMARY:")
    print("=" * 60)
    print(report.short_summary)

    print("\n" + "=" * 60)
    print("FULL REPORT:")
    print("=" * 60)
    display(Markdown(report.markdown_report))

    print("\n" + "=" * 60)
    print("FOLLOW-UP QUESTIONS:")
    print("=" * 60)
    for i, question in enumerate(report.follow_up_questions, 1):
        print(f"{i}. {question}")


async def test_individual_search():
    """Test the search agent directly"""
    print("\n\n" + "=" * 60)
    print("TESTING INDIVIDUAL SEARCH:")
    print("=" * 60)

    message = "Latest AI Agent frameworks in 2025"
    result = await Runner.run(search_agent, message)
    display(Markdown(result.final_output))


async def test_planner():
    """Test the planner agent"""
    print("\n\n" + "=" * 60)
    print("TESTING SEARCH PLANNER:")
    print("=" * 60)

    message = "Latest AI Agent frameworks in 2025"
    result = await Runner.run(planner_agent, message)
    for i, search_item in enumerate(result.final_output.searches, 1):
        print(f"\n{i}. Query: {search_item.query}")
        print(f"   Reason: {search_item.reason}")

# ==================== DEPLOYMENT ====================
async def run_research_pipeline(query):
    search_plan = await plan_searches(query)
    search_results = await perform_searches(search_plan)
    report = await write_report(query, search_results)
    await send_email(report)
    return report.short_summary, report.markdown_report

def sync_wrapper(query):
    return asyncio.run(run_research_pipeline(query))

gr.Interface(
    fn=sync_wrapper,
    inputs=gr.Textbox(label="Enter your Research Query"),
    outputs=[
        gr.Textbox(label="Summary", lines=3),
        gr.Markdown(label="Detailed Report")
    ],
    title="Agentic AI Research Assistant",
    description="An agentic AI that plans, searches, and writes deep research reports using SerpAPI and Gemini/OpenAI models."
).launch()

# Run the main function
if __name__ == "__main__":
    # Choose which example to run:
    
    # Option 1: Run full research pipeline
    asyncio.run(main())
    
    # Option 2: Test individual search
    # asyncio.run(test_individual_search())
    
    # Option 3: Test planner
    # asyncio.run(test_planner())