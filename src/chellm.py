import os
from rdkit import Chem
from rdkit.Chem import Draw
from openai import OpenAI
from IPython.display import display
import logging
import pubchempy as pcp
import json
import gradio as gr

# -------------------
# Logging
# -------------------
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

def display_smiles(smiles: str, filename: str = "molecule.png") -> str:
    """
    Generate and save an image of a molecule from a SMILES string.

    Args:
        smiles (str): SMILES representation of the molecule.
        filename (str): Output image filename.

    Returns:
        str: Path to the saved image.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        img = Draw.MolToImage(mol)
        img.save(filename)

        return os.path.abspath(filename)

    except Exception as e:
        raise ValueError(f"Error drawing SMILES {smiles}: {e}")
    
def get_smiles_from_pubchem(name: str) -> str:
    """
    Get the SMILES representation of a molecule from PubChem based on its name.
    
    Args:
        name (str): The name of the molecule.
    
    Returns:
        str: The SMILES representation of the molecule.
    """
    try:
        compounds = pcp.get_compounds(name, "name")
        if not compounds:
            return "No compounds found."
        compound = compounds[0]
        return compound.smiles
    except Exception as e:
        return f"An error occured: {e}"
    
# -------------------
# Tool schema (OpenAI style)
# -------------------
get_smiles_from_pubchem_json = {
    "type": "function",
    "function": {
        "name": "get_smiles_from_pubchem",
        "description": "Get a SMILES string from PubChem given a molecule name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The common name of the molecule (e.g., 'benzoic acid')",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

display_smiles_json = {
    "type": "function",
    "function": {
        "name": "display_smiles",
        "description": "Draw a molecule from a SMILES string and save it as an image file.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "The SMILES representation of the molecule"
                },
                "filename": {
                    "type": "string",
                    "description": "Name of the output image file",
                    "default": "molecule.png"
                }
            },
            "required": ["smiles"],
            "additionalProperties": False
        }
    }
}

tools = [get_smiles_from_pubchem_json, display_smiles_json]

class CheLMM:
    def __init__(self):
        self.ollama = "llama3.3:latest"
        self.openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        return (
            "You are a domain expert in cheminformatics and molecular chemistry.\n\n"

            "When the user requests a SMILES string for a molecule by name, you MUST call "
            "the get_smiles_from_pubchem tool to obtain it. You must not invent or guess SMILES.\n\n"

            "When the user asks to draw, visualize, show, display, or render a molecule, you MUST:\n"
            "1. Obtain the SMILES string. If the user provides a molecule name, call "
            "get_smiles_from_pubchem to retrieve the SMILES.\n"
            "2. Call the display_smiles tool with the SMILES string to generate and save an image "
            "of the molecule in the current working directory.\n\n"

            "If the user provides a SMILES string directly and asks to draw it, call display_smiles "
            "using the provided SMILES without calling get_smiles_from_pubchem.\n\n"

            "If an error occurs while retrieving or drawing the molecule, explain the error and ask "
            "the user for clarification.\n\n"

            "If the user does not request drawing or visualization, return only the SMILES string "
            "as plain text and nothing else. Do not include explanations or additional text."
        )
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model=self.ollama, messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

# ------------------
# MAIN
# ------------------
if __name__ == "__main__":
    chellm = CheLMM()
    gr.ChatInterface(chellm.chat).launch()

