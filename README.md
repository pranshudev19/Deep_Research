🧠 Deep Research

An AI-powered multi-agent research assistant that autonomously plans, searches, summarizes, and generates comprehensive research reports.
Built using Gemini 2.5, SerpAPI, and Gradio, it demonstrates an end-to-end intelligent workflow — from query planning to report generation and email delivery.

🚀 Overview

Deep Research is designed to automate the entire research lifecycle:

Planner Agent — Decomposes the main query into focused sub-topics.

Search Agent — Gathers real-time information from the web using SerpAPI.

Writer Agent — Synthesizes a detailed report (in Markdown format).

Email Agent — Sends the final report directly via SendGrid API.

All agents communicate seamlessly under an agentic workflow using Gemini 2.5 LLM for reasoning and decision orchestration.

🧩 Features

🔍 Web Intelligence: Uses SerpAPI for live web search and context extraction.

🧠 Agentic Automation: Multi-agent collaboration for planning, research, and synthesis.

📝 Report Generation: Automatically generates structured, markdown-formatted reports.

📧 Email Integration: Sends summarized and detailed reports using SendGrid.

🌐 Interactive UI: Built with Gradio, and deployable on Hugging Face Spaces.

🛠️ Tech Stack

LLM Framework: Gemini 2.5 (via OpenAI-compatible API)

Search Engine: SerpAPI

Deployment/UI: Gradio, Hugging Face Spaces

Email Service: SendGrid API

Environment Management: dotenv

Language: Python 3.10+
