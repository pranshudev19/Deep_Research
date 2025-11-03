from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncOpenAI
import asyncio
import os
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from sendgrid import SendGridAPIClient
load_dotenv(override = True)

GEMINi_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GEMINI_API_KEY")
if google_api_key:
    print("Google api key is present!!")
else:
    print("Error!")

gemini_client = AsyncOpenAI(base_url = GEMINi_BASE_URL, api_key = google_api_key)

gemini_model = OpenAIChatCompletionsModel(model = 'gemini-2.5-flash', openai_client = gemini_client)


@function_tool
def send_job_email(body:str):
    """Sendout an email with the given body to all employers prospects"""
    sg = sendgrid.SendGridAPIClient(api_key = os.environ.get('SENDGRID_API_KEY'))
    from_email = Email('pranshuwork19@gmail.com')
    to_email = To('devpramod@yahoo.com')
    content = Content('text/plain', body)
    mail = Mail(from_email, to_email, "Application email", content).get()
    sg.client.mail.send.post(request_body = mail)
    return {"status":"success"}

instructions_01 = """You are a professional email writing assistant skilled in crafting consise, personalized and formal job application, internship oppurtunity or networking emails.
    Write a short, polished email that I can send to a potential employer or hiring manager. The email should:
    - Have a clear, respectful greeting and subject line suggestion.
    - Introduce me briefly (e.g. name, professional background, area of expertise).
    - Express genuine interest in the company or role.
    - Highlight one or two key strengths or achievements relevant to the position.
    - Politely indicate my openness to discuss oppurtunities or share my resume.
    - End with a professional closing and signature format. 
    
    Tone: Formal, confident, consise( 120 - 150 words).
    Avoid unnecessary jargon and repetition

    Assume the following details unless I provide new ones:

    - Name: **Pranshu Devhade**
    - Field: **Artificial Intelligence and Data Science, Machine Learning, Software Development**
    - Experience: **Internship Experince in Machine Learning, AI agents, Next.js, Tailwind, Data-driven systems**
    - Goal: **To connect with potential empolyers or hiring managers for AI or software development roles**
    """

job_agent = Agent(
    name = "Job Applicant",
    instructions = instructions_01,
    model = gemini_model
)
job_agent_tool = job_agent.as_tool(tool_name="job_agent", tool_description="Write a application email to hiring manager.")

tools = [job_agent_tool,send_job_email]

job_manager_instructions = """
You are Job Manager Agent. Your goal is to compose and send a professional email to the potential employer or hiring manager using job_agent and 
send_job_email tools.
Follow these steps carefully:
1. Generate Drafts: Use the job_agent to generate the email drafts. Do not proceed until the draft is ready.

2. Evaluate: Review the drafts and decide the whether is professional and best to impress using your judgement. Do not proceed until it is ready.

3. Sending: Send the email using the send_job_email tools

Crucial Rules:
- You must use the job_agent to generate the drafts - do not write them yourself.
- You must send exactly ONE email using the send_job_email tool- never more than one.
"""
job_manager = Agent(
    name = "Job Manager",
    instructions = job_manager_instructions,
    tools = tools,
    model = gemini_model
)
message = "Send out a job application email to Hiring Manager at tech company."

async def main():
    result = await Runner.run(job_manager, message)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

