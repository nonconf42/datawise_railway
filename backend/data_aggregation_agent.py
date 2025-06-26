import pandas as pd
import numpy as np
import os
import importlib.util
import sys
import random
import string
import shutil
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import re
import uuid

# --- 1. Core Data Structures ---

@dataclass
class Section:
    """A single, self-contained section of a strategic recommendation."""
    text: str
    table: Optional[pd.DataFrame] = None
    code: str = ""

@dataclass
class StrategicRecommendation:
    """Stores a complete, multi-part strategic recommendation with all its components."""
    title: str
    finding: Section
    action_logic: Section
    feasibility: Section
    effect: Section
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_from_feedback: Optional[str] = None

# --- 2. Utility Functions ---

def save_code_to_file(code_string: str, filename: str) -> str:
    if not filename.endswith('.py'):
        filename += '.py'
    commented_code = f"# {filename}\n{code_string}"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(commented_code)
    return os.path.abspath(filename)

def extract_python_code(text: str) -> Optional[str]:
    code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(code_pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()

# --- 3. The Strategic Recommendation Agent ---

class DataAggregationAgent:
    """
    An agent that generates a series of deep, multi-part strategic recommendations.
    """
    
    def __init__(self, llm_model, dataset, num_recommendations: int = 3):
        self.llm_model = llm_model
        self.dataset = dataset
        self.num_recommendations = num_recommendations
        self.recommendations_pool: List[StrategicRecommendation] = []

    def run(self) -> List[StrategicRecommendation]:
        """Main entry point for autonomous generation of recommendations."""
        print(f"[INFO] Starting Strategic Recommendation Agent to generate {self.num_recommendations} reports.")
        for i in range(self.num_recommendations):
            print(f"\n--- Generating Autonomous Recommendation #{i + 1} of {self.num_recommendations} ---")
            try:
                new_recommendation = self._generate_one_recommendation(
                    previous_recommendations=self.recommendations_pool
                )
                if new_recommendation:
                    self.recommendations_pool.append(new_recommendation)
                    print(f"[INFO] Successfully generated Recommendation: '{new_recommendation.title}'")
                else:
                    print(f"[WARN] Failed to generate recommendation #{i + 1}. Moving to next.")
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred while generating recommendation #{i + 1}: {e}")
        self.save_results("strategic_recommendations")
        return self.recommendations_pool

    def generate_recommendation_from_feedback(self, feedback_text: str, source_recommendation_id: str) -> Optional[StrategicRecommendation]:
        """Generates a single, new strategic recommendation heavily guided by user feedback."""
        print(f"[INFO] Generating new recommendation based on feedback: '{feedback_text}'")
        source_rec = next((r for r in self.recommendations_pool if r.id == source_recommendation_id), None)
        if not source_rec:
            print(f"[ERROR] Source recommendation with ID {source_recommendation_id} not found.")
            return None
        try:
            new_recommendation = self._generate_one_recommendation(
                previous_recommendations=self.recommendations_pool,
                feedback_on_source={"text": feedback_text, "source_rec": source_rec}
            )
            if new_recommendation:
                new_recommendation.generated_from_feedback = f"Feedback on Rec ID: {source_recommendation_id}"
                self.recommendations_pool.append(new_recommendation)
                print(f"[INFO] Successfully generated new recommendation from feedback: '{new_recommendation.title}'")
                self.save_results("strategic_recommendations")
                return new_recommendation
        except Exception as e:
            print(f"[ERROR] Failed to generate recommendation from feedback: {e}")
        return None

    def _generate_one_recommendation(self, previous_recommendations: List[StrategicRecommendation], feedback_on_source: Optional[Dict] = None) -> Optional[StrategicRecommendation]:
        """Orchestrates the 4-step 'assembly line' to generate a single, complete recommendation."""
        context = {"previous_findings_full_context": self._get_full_previous_findings_context(previous_recommendations)}
        if feedback_on_source:
            context["feedback_on_source"] = feedback_on_source
        
        for task in ["finding", "action_logic", "feasibility", "effect"]:
            print(f"[DEBUG] Step: Generating '{task}'...")
            section = self._execute_sub_agent_task(task=task, context=context)
            if not section:
                print(f"[ERROR] Aborting recommendation generation because step '{task}' failed.")
                return None
            context[task] = section

        print("[DEBUG] Step: Generating Title...")
        title_prompt = self._get_prompt(task="title", step="generate_text", context=context)
        title = self.llm_model.llm_call(prompt=title_prompt).strip().strip('"')
        
        return StrategicRecommendation(
            title=title,
            finding=context["finding"],
            action_logic=context["action_logic"],
            feasibility=context["feasibility"],
            effect=context["effect"]
        )

    def _execute_sub_agent_task(self, task: str, context: Dict) -> Optional[Section]:
        """Encapsulates the two-step 'Code -> Analyze' process for a single task."""
        try:
            print(f"[DEBUG]   Sub-step: Generating code for '{task}'...")
            code_prompt = self._get_prompt(task=task, step="generate_code", context=context)
            llm_code_response = self.llm_model.llm_call(prompt=code_prompt)
            agg_code = extract_python_code(llm_code_response)
            if not agg_code:
                print(f"[ERROR] No Python code found in the LLM response for task '{task}'.")
                return None

            print(f"[DEBUG]   Sub-step: Executing code for '{task}'...")
            table_data = self._run_dynamic_code_with_correction(agg_code, context)
            if table_data is None:
                print(f"[WARN] Code for task '{task}' failed after correction attempt. Continuing without a table.")
                table_data = pd.DataFrame() 

            print(f"[DEBUG]   Sub-step: Generating analysis for '{task}'...")
            text_prompt = self._get_prompt(task=task, step="generate_text", context=context, table_data=table_data)
            analysis_text = self.llm_model.llm_call(prompt=text_prompt)

            return Section(text=analysis_text.strip(), table=table_data, code=agg_code)
        
        except Exception as e:
            print(f"[ERROR] Sub-agent task '{task}' failed: {e}")
            return None

    def _run_dynamic_code_with_correction(self, code_string: str, context: Dict) -> Optional[pd.DataFrame]:
        """
        Attempts to run generated code. If it fails, it asks the LLM to debug it up to 3 times.
        """
        max_attempts = 7
        current_code = code_string
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[DEBUG] Attempt #{attempt} to execute code...")
                # The helper function _run_dynamic_code will raise an exception on failure
                return self._run_dynamic_code(current_code)
            
            except Exception as e:
                last_error = str(e)
                print(f"[WARN] Attempt #{attempt} failed with error: {last_error}")
                
                # If this was the last attempt, don't try to correct, just exit.
                if attempt == max_attempts:
                    print(f"[ERROR] All {max_attempts} execution attempts failed. Final error: {last_error}")
                    break
                
                # --- Self-Correction Step ---
                print(f"[INFO] Attempting self-correction (Attempt {attempt+1}/{max_attempts})...")
                try:
                    # Get a debug prompt with the latest failed code and error
                    debug_prompt = self._get_debug_prompt(current_code, last_error, context)
                    llm_fixed_code_response = self.llm_model.llm_call(prompt=debug_prompt)
                    
                    # Update the code for the next loop iteration
                    corrected_code = extract_python_code(llm_fixed_code_response)
                    
                    if not corrected_code or corrected_code == current_code:
                        print("[ERROR] Self-correction failed: LLM did not return new or different code.")
                        return None # Stop if the LLM isn't helping
                    
                    current_code = corrected_code
                    
                except Exception as llm_e:
                    print(f"[ERROR] An error occurred during the self-correction LLM call: {llm_e}")
                    # Abort if the correction process itself fails
                    return None

        # If the loop completes without a successful return, it means all attempts failed.
        return None

    def _run_dynamic_code(self, code_string: str) -> pd.DataFrame:
        """Helper to execute code. Throws an exception on failure."""
        file_name = ''.join(random.choices(string.ascii_lowercase, k=12))
        scripts_dir = "temp_scripts"
        os.makedirs(scripts_dir, exist_ok=True)
        file_path = os.path.join(scripts_dir, f"{file_name}.py")
        
        print(f"[DEBUG] Attempting to execute code:\n---\n{code_string}\n---")
        
        try:
            save_code_to_file(code_string, file_path)
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            
            module = importlib.import_module(file_name)
            
            if hasattr(module, 'aggregate'):
                aggregate_func = getattr(module, 'aggregate')
                result = aggregate_func(self.dataset.train_base.copy())
                
                if isinstance(result, pd.DataFrame):
                    return result
                else:
                    raise TypeError(f"Generated code did not return a pandas DataFrame. Returned type: {type(result)}")
            else:
                raise AttributeError("No 'aggregate' function found in the generated code.")
        finally:
            if file_name in sys.modules:
                del sys.modules[file_name]
            if os.path.exists(file_path):
                os.remove(file_path)
            pycache_path = os.path.join(scripts_dir, '__pycache__')
            if os.path.exists(pycache_path):
                 shutil.rmtree(pycache_path, ignore_errors=True)
    
    def _get_full_previous_findings_context(self, recommendations: List[StrategicRecommendation]) -> str:
        if not recommendations:
            return "None."
        
        full_context_str = ""
        for i, rec in enumerate(recommendations):
            full_context_str += f"### Previous Finding {i+1} ###\n"
            full_context_str += f"**Title:** {rec.title}\n"
            full_context_str += f"**Text:**\n{rec.finding.text}\n\n"
            table = rec.finding.table
            if table is not None and not table.empty:
                full_context_str += f"**Data Table:**\n{table.to_string()}\n\n"
        return full_context_str

    def _prepare_prompt_context(self, base_context: Dict, table_data: Optional[pd.DataFrame] = None) -> Dict:
        """Creates a single, flat dictionary for prompt formatting."""
        flat_context = {
            "dataset_description": self.dataset.description,
            "columns_description_str": "\n".join([f"- `{k}`: {v}" for k, v in self.dataset.features_description.items()]),
            "valid_columns": list(self.dataset.train_base.columns),
            "previous_findings_full_context": base_context.get("previous_findings_full_context", "None.")
        }

        if "finding" in base_context:
            section = base_context["finding"]
            flat_context["finding_text"] = section.text
            table = section.table
            flat_context["finding_table_string"] = table.to_string() if table is not None and not table.empty else "N/A"

        if "action_logic" in base_context:
            flat_context["action_logic_text"] = base_context["action_logic"].text

        if "feasibility" in base_context:
            flat_context["feasibility_text"] = base_context["feasibility"].text
        
        if table_data is not None and not table_data.empty:
            flat_context['table_string'] = table_data.to_string()
        else:
            flat_context['table_string'] = "No data table was generated for this step."
        
        if "feedback_on_source" in base_context:
            source_rec = base_context["feedback_on_source"]["source_rec"]
            feedback_text = base_context["feedback_on_source"]["text"]
            flat_context['feedback_block'] = f'**CRITICAL INSTRUCTION: ADDRESS USER FEEDBACK**\nA user provided feedback on "{source_rec.title}": "{feedback_text}"\nYour primary goal is to generate a **new Finding** that directly responds to this feedback.'
            flat_context['task_instructions'] = "1. Address the user's feedback directly.\n2. Write a Python `aggregate(dataset)` function."
        else:
            flat_context['feedback_block'] = "Your task is to find a **new and unique** opportunity not covered before."
            flat_context['task_instructions'] = "1. Analyze previous findings to avoid repetition.\n2. Write a Python `aggregate(dataset)` function."

        return flat_context

    def _get_debug_prompt(self, faulty_code: str, error_message: str, context: Dict) -> str:
        """Creates a prompt for the LLM to debug a piece of code."""
        full_context = self._prepare_prompt_context(context)
        grounding_block = self._get_prompt_grounding_block(for_debug=True).format(**full_context)
        
        return f"""
You are a Python debugging expert. The following Python code failed to execute.
Your task is to analyze the code, the context, and the error message, and then return a corrected version of the code.

**CONTEXT AND SCHEMA:**
{grounding_block}

**FAULTY CODE:**
```python
{faulty_code}
```

**ERROR MESSAGE:**
```
{error_message}
```

**INSTRUCTIONS:**
1.  Read the error message carefully to identify the mistake.
2.  Rewrite the entire `aggregate` function with the necessary corrections, ensuring it adheres to all critical code generation rules.
3.  Return ONLY the corrected Python code inside a markdown block. Do not add any explanations.

**Your turn. Provide the corrected code.**
"""
        
    def _get_prompt_grounding_block(self, for_debug=False) -> str:
        """Helper to create the consistent schema and rules block."""
        code_gen_rules = """
**CRITICAL CODE GENERATION RULES:**
1.  **Return a Flat pandas DataFrame:** The `aggregate` function MUST return a single pandas DataFrame.
2.  **No Nested Objects:** The cells of the returned DataFrame must only contain primitive types (numbers, strings, booleans, timestamps).
3.  **No Mixed-Type Columns for Math:** Do not create a column or Series that contains both numbers and strings if you intend to perform mathematical operations on it.
4.  **Use ONLY Provided Column Names:** The available columns are `{valid_columns}`. Do NOT use any other column names (e.g., 'Order Value', 'Order ID').
5.  **Handle Data Types:** Convert 'Order Time' to datetime using `pd.to_datetime(dataset['Order Time'])`.
6.  **Include All Imports:** You MUST include imports like `import pandas as pd` and `import numpy as np` *inside* the `aggregate` function.
7.  **Handle Merges Carefully:** Proactively rename columns on one DataFrame *before* merging to avoid name collisions.
8.  **Enclose in Markdown:** Your response MUST contain ONLY the Python code, enclosed in a single markdown block (```python ... ```).
"""
        # The debug prompt does not need the final markdown rule
        if for_debug:
             code_gen_rules = "\n".join(code_gen_rules.split('\n')[:-1])


        return f"""
**DATASET CONTEXT AND SCHEMA:**
- Dataset Description: {{dataset_description}}
- Available Columns and Descriptions:
{{columns_description_str}}
{code_gen_rules}
"""

    def _get_prompt(self, task: str, step: str, context: Dict, table_data: Optional[pd.DataFrame] = None) -> str:
        """Acts as a prompt router, returning the correct prompt from the master library."""
        grounding_block_template = self._get_prompt_grounding_block()
        
        # --- FIX: Refined text prompts to forbid code and guide formatting ---
        prompts = {
            "finding": {
                "generate_code": """You are a data analyst... {feedback_block}\n{grounding_block}\n**CONTEXT - PREVIOUSLY GENERATED FINDINGS:**\n{previous_findings_full_context}\n**YOUR TASK:**\n{task_instructions}\n**Your turn. Write the Python code.**""",
                "generate_text": """
You are a senior analyst. Your task is to write the text for the **Finding** section.
Your response must be professional prose only. Do NOT include any code blocks.
Use the provided data table as your core evidence to concisely state the problem or opportunity. Use bold markdown for any headers.

{grounding_block}
**DATA TABLE (EVIDENCE):**
{table_string}

**EXAMPLE OF GOOD FINDING TEXT:**
**Key Insight:** Our production data reveals a critical mismatch between products and manufacturing equipment.

**Your turn. Write the text for the Finding section.**
"""
            },
            "action_logic": {
                "generate_code": """You are a solutions architect... {grounding_block}\n**CONTEXT - FINDING TO ADDRESS:**\n- Finding Text: {finding_text}\n- Finding Table: {finding_table_string}\n**Your turn. Write the Python code.**""",
                "generate_text": """
You are a senior analyst. Your task is to write the **Action Logic** section.
Your response must be professional prose only. Do NOT include any code blocks.
Your response MUST be structured into exactly four sections, using markdown bolding for the titles: **Problem**, **Solution**, **Implementation**, and **Evidence**.

{grounding_block}
**CONTEXT - FINDING TO ADDRESS:**
- Finding Text: {finding_text}
**DATA TABLE (SUPPORTING EVIDENCE):**
{table_string}

**Your turn. Write the text for the Action Logic section following the 4-part structure precisely.**
"""
            },
            "feasibility": {
                "generate_code": """You are a project manager... {grounding_block}\n**CONTEXT - PLAN TO ASSESS:**\n- Finding Text: {finding_text}\n- Action Logic Text: {action_logic_text}\n**Your turn. Write the Python code.**""",
                "generate_text": """
You are a senior analyst. Your task is to write the **Implementation Feasibility** section.
Your response must be professional prose only. Do NOT include any code blocks.
Based on the provided data, analyze risks and confirm feasibility. Use bold markdown for any headers.

{grounding_block}
**CONTEXT - ACTION TO BE IMPLEMENTED:**
- Action Logic Text: {action_logic_text}
**DATA TABLE (FEASIBILITY DATA):**
{table_string}

**Your turn. Write the text for the Implementation Feasibility section.**
"""
            },
            "effect": {
                "generate_code": """You are a financial analyst... {grounding_block}\n**CONTEXT - FULL PROPOSAL:**\n- Finding Text: {finding_text}\n- Action Logic Text: {action_logic_text}\n- Feasibility Text: {feasibility_text}\n**Your turn. Write the Python code.**""",
                "generate_text": """
You are a senior analyst. Your task is to write the **Expected Effect** section.
Your response must be professional prose only. Do NOT include any code blocks.
Based on the context and data table, summarize the projected improvements. Use bold markdown for any headers.

{grounding_block}
**CONTEXT - ACTION BEING IMPLEMENTED:**
- Action Logic Text: {action_logic_text}
**DATA TABLE (PROJECTED IMPACT):**
{table_string}

**Your turn. Write the text for the Expected Effect section.**
"""
            },
            "title": {
                "generate_text": """You are a senior business analyst. Based on the following summary, write a single, concise, and compelling title for a strategic recommendation report. The title should be a single sentence and not contain any markdown formatting or quotation marks.\n\n**Finding:** {finding_text}\n**Proposed Action:** {action_logic_text}\n\n**Your turn. Write the title.**"""
            }
        }
        
        prompt_template = prompts.get(task, {}).get(step)
        if not prompt_template:
            raise ValueError(f"No prompt found for task '{task}' and step '{step}'")

        final_context = self._prepare_prompt_context(context, table_data)
        final_context["grounding_block"] = grounding_block_template.format(**final_context)
        
        return prompt_template.format(**final_context)

    def save_results(self, output_dir: str):
        """Saves all generated strategic recommendations to a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        summary_path = os.path.join(output_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Strategic Recommendations Summary\n\n")
            for i, rec in enumerate(self.recommendations_pool, 1):
                f.write(f"## {i}. {rec.title}\n\n")
                if rec.generated_from_feedback:
                    f.write(f"_(Generated from: {rec.generated_from_feedback})_\n\n")
                f.write(f"{rec.finding.text}\n\n")
                f.write(f"[View Full Recommendation](./recommendation_{rec.id}.md)\n\n")

        for rec in self.recommendations_pool:
            rec_path = os.path.join(output_dir, f"recommendation_{rec.id}.md")
            with open(rec_path, "w", encoding="utf-8") as f:
                f.write(f"# {rec.title}\n\n")
                if rec.generated_from_feedback:
                    f.write(f"_(Generated from: {rec.generated_from_feedback})_\n\n")

                def write_section(name: str, section: Section):
                    f.write(f"## {name}\n\n")
                    f.write(f"{section.text}\n\n")
                    if section.table is not None and not section.table.empty:
                        f.write("#### Supporting Data\n")
                        f.write(f"```\n{section.table.to_string()}\n```\n\n")

                write_section("Finding", rec.finding)
                write_section("Action Logic", rec.action_logic)
                write_section("Implementation Feasibility", rec.feasibility)
                write_section("Expected Effect", rec.effect)

        print(f"[INFO] All {len(self.recommendations_pool)} recommendations saved to '{output_dir}/'")