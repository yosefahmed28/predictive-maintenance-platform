import os
import sys
import json

# Root path configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from langchain_ollama import ChatOllama
from src.rag.rag_tools import query_technical_manuals, get_machine_telemetry_status

def run_maintenance_assistant(user_query: str):
    llm = ChatOllama(model="llama3.2", temperature=0.1)
    tools = [query_technical_manuals, get_machine_telemetry_status]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "You are the Industrial Maintenance & Diagnostics AI Agent.\n"
        "1. Check machine telemetry when asked about specific machines/UDIs.\n"
        "2. If high failure risk or abnormal metrics are reported, call query_technical_manuals to retrieve fix protocols.\n"
        "3. Provide direct, actionable maintenance steps based on telemetry and manual findings."
    )

    messages = [
        ("system", system_prompt),
        ("human", user_query)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Tool Execution Loop
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_outputs = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "get_machine_telemetry_status":
                udi_val = tool_args.get("udi", 10)
                out = get_machine_telemetry_status.invoke({"udi": udi_val})
                tool_outputs.append(f"Telemetry Data (UDI {udi_val}): {out}")
                
            elif tool_name == "query_technical_manuals":
                query_str = tool_args.get("query", "maintenance protocol")
                out = query_technical_manuals.invoke({"query": query_str})
                tool_outputs.append(f"Manual Context: {out}")

        synthesis_messages = messages + [
            ("assistant", f"Tool Outputs: {json.dumps(tool_outputs)}"),
            ("human", "Summarize findings and output recommended maintenance actions.")
        ]
        final_answer = llm.invoke(synthesis_messages)
        return final_answer.content
    else:
        return response.content

if __name__ == "__main__":
    print("="*60)
    print("      INDUSTRIAL MAINTENANCE AI ASSISTANT READY")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nUser Query > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                break
            
            print("\nProcessing request locally via Ollama...")
            result = run_maintenance_assistant(user_input)
            print("\n--- DIAGNOSTIC REPORT ---")
            print(result)
            print("="*50)
            
        except KeyboardInterrupt:
            break