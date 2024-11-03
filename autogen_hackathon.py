import os
api_key = os.getenv("OPENAI_API_KEY")
llm_config={"model": "gpt-4-turbo",
            "api_key": f"{api_key}",
            }


COMPANY = "Google"
task = f"Based on a {COMPANY}, create a report on the company's future performance."

import openai
import autogen
import GoogleNewsScraper

openai.api_key = api_key


news_titles = GoogleNewsScraper.Company_News_Scraper(COMPANY)

user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message=f"Your task is to initiate a company performance analysis for {COMPANY}. "
                   "Direct each specialized agent to consider relevant news articles when performing their analysis. "
                   "Once each agent completes its task, gather all results and compile them into a cohesive report.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

planner_agent = autogen.ConversableAgent(
    name="PlannerAgent",
    system_message="Keep track of risk assessment, resilience, growth potential, financial health, and competitive positioning analyses. "
                     "Ensure that each agent considers the provided news titles" + str(news_titles) + "in their evaluation, remind them to numerically assess each metric. "
                    "If an agent fails to complete their task, provide guidance or reminders as needed. "
                    "When all agents have been called, call the FinalWriterAgent to compile the final report.",
    description="PlannerAgent. Oversees task flow, keeps track of each agent's progress, and provides reminders or guidance as necessary.",
    llm_config=llm_config,
)

# Risk Assessment Agent
risk_assessment_agent = autogen.ConversableAgent(
    name="RiskAssessmentAgent",
    system_message=f"Analyze {COMPANY}'s risk factors using the provided real-time news titles: " + str(news_titles) + ". "
                   "Focus on market volatility, financial leverage, and operational risks. Assign a score from 0 to 5 for each sub-metric, "
                   "and provide a final overall risk score.",
    description="RiskAssessmentAgent. Calculates a risk score based on key risk factors, "
                "including market volatility, financial leverage, operational risks, and relevant news.",
    llm_config=llm_config,
)

# Resilience Agent
resilience_agent = autogen.ConversableAgent(
    name="ResilienceAgent",
    system_message=f"Analyze {COMPANY}’s resilience using the provided real-time news titles: " + str(news_titles) + ". "
                   "Focus on cash reserves, adaptability to change, crisis management, and employee retention. Assign a score from 0 to 5 for each sub-metric, "
                   "and provide a final resilience score.",
    description="ResilienceAgent. Calculates a resilience score based on factors such as cash reserves, adaptability, crisis management, "
                "employee retention, and relevant news.",
    llm_config=llm_config,
)

# Growth Potential Agent
growth_potential_agent = autogen.ConversableAgent(
    name="GrowthPotentialAgent",
    system_message=f"Analyze {COMPANY}’s growth potential using the provided real-time news titles: " + str(news_titles) + ". "
                   "Focus on product innovation, market expansion, partnerships, and R&D investments. Assign a score from 0 to 5 for each sub-metric, "
                   "and provide a final growth potential score.",
    description="GrowthPotentialAgent. Calculates a growth potential score based on product innovation, market expansion, partnerships, R&D investments, and relevant news.",
    llm_config=llm_config,
)

# Financial Health Agent
financial_health_agent = autogen.ConversableAgent(
    name="FinancialHealthAgent",
    system_message=f"Analyze {COMPANY}’s financial health using the provided real-time news titles: " + str(news_titles) + ". "
                   "Focus on profitability ratios, debt levels, liquidity, and overall stability. Assign a score from 0 to 5 for each sub-metric, "
                   "and provide a final financial health score.",
    description="FinancialHealthAgent. Calculates a financial health score based on profitability ratios, debt levels, liquidity, financial stability, and relevant news.",
    llm_config=llm_config,
)

# Competitive Positioning Agent
competitive_positioning_agent = autogen.ConversableAgent(
    name="CompetitivePositioningAgent",
    system_message=f"Analyze {COMPANY}’s competitive positioning using the provided real-time news titles: " + str(news_titles) + ". "
                   "Focus on market share, technological advantages, brand reputation, and positioning against competitors. Assign a score from 0 to 5 for each sub-metric, "
                   "and provide a final competitive positioning score.",
    description="CompetitivePositioningAgent. Calculates a competitive positioning score based on market share, technological advantages, brand reputation, "
                "competitive positioning, and relevant news.",
    llm_config=llm_config,
)

final_writer_agent = autogen.ConversableAgent(
    name="FinalWriterAgent",
    system_message="You are responsible for compiling the final report on the company's performance. "
                   "Collect the outputs from each metric-focused agent (Risk, Resilience, Growth, Financial Health, "
                   "Competitive Positioning) and assign a rating for each metric on a scale from 0 to 5. "
                   "Present the final report in markdown format, with a clear title, individual sections for each metric, "
                   "and an overall assessment summary. If there is no score available for a metric, indicate it as 3",
    description="FinalWriterAgent. Collects, synthesizes, and rates each metric, providing a comprehensive "
                "summary report with ratings from 0 to 5 for each category.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, risk_assessment_agent, resilience_agent, growth_potential_agent,
            financial_health_agent, competitive_positioning_agent, final_writer_agent],
    messages=[],
    max_round=7,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [],  # User Proxy can only transition to the Planner
        # planner_agent: [risk_assessment_agent, resilience_agent, growth_potential_agent, financial_health_agent, competitive_positioning_agent, final_writer_agent, user_proxy],  # Planner manages the flow and communicates with each agent
        risk_assessment_agent: [resilience_agent],  # Agents return to the Planner after task completion
        resilience_agent: [growth_potential_agent],
        growth_potential_agent: [financial_health_agent],
        financial_health_agent: [competitive_positioning_agent],
        competitive_positioning_agent: [final_writer_agent],
        final_writer_agent: []  # FinalWriterAgent is the last step with no further transitions
    },
    speaker_transitions_type="allowed",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)

# Convert the final output to a string (if not already) to pass into the prompt
final_output_text = str(groupchat_result)

summary_prompt = f"""
Based on the detailed analysis provided by the FinalWriterAgent, please:
1. Extract the five performance metrics (Risk, Resilience, Growth Potential, Financial Health, Competitive Positioning) And put then in a Python Dictionary .
2. Provide a concise summary bullet points that include:
   - Key strengths and opportunities
   - Main risks or challenges
   - Overall direction of the company's future performance

   {final_output_text}
"""


# Pass this prompt to the model for summarization using the updated API
def get_chat_completion(prompt, model="gpt-4-turbo"):
    # Creating a message as required by the API
    messages = [{"role": "system", "content": "You are an expert summarizer and evaluator."},
                {"role": "user", "content": prompt}]
    
    # Calling the ChatCompletion API
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    
    # Returning the extracted response
    return response.choices[0].message.content

# Assuming `summary_prompt` contains the prompt text for summarization
summarized_report = get_chat_completion(summary_prompt)

# Output the summary
# print("Summary and Extracted Metrics:", summarized_report)

import ast
import re

def extract_dictionary_from_string(text):
    # Regular expression to match a dictionary pattern in the text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    
    if match:
        try:
            # Convert matched dictionary-like string to a dictionary
            dictionary = ast.literal_eval(match.group())
            if isinstance(dictionary, dict):
                return dictionary
            else:
                raise ValueError("Matched content is not a dictionary")
        except (SyntaxError, ValueError):
            print("Error: Unable to parse the dictionary.")
            return None
    else:
        print("No dictionary found in the text.")
        return None

t = extract_dictionary_from_string(summarized_report)

def extract_bullet_points(text):
    # Updated regex to match bullet points that begin with •, -, or *
    bullet_points = re.findall(r"(?m)^(?:\s*[•\-\*])\s+(.+)", text)
    return bullet_points

result = extract_bullet_points(summarized_report)

print("Extracted Metrics:", t)
print("bullet points:", result)