import numpy as np
import pandas as pd
import argparse
import requests
from rdkit import Chem
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import pubchempy
import mwclient 

# -------------------
# LLM
# -------------------
OLLAMA = "gpt-oss:20b"
OLLAMA_API = "http://localhost:11434/api/chat"

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

system_prompt = """
You will receive raw text extracted from the PubChem “Melting Point” section for a chemical compound. Your task is to identify the compound’s melting point from this text and return it as a single numerical value in Kelvin.

Requirements:

1. Extract the melting point value stated in the text.
   - If multiple values are provided, use the primary or most typical value.
   - If a range is given, return the midpoint.
   - Ignore qualitative descriptions (e.g., “solidifies at”, “softening point”).

2. Convert the melting point to Kelvin if it is provided in Celsius or Fahrenheit.

3. Output formatting:
   - Return only a single number.
   - Do not include units, explanations, text, or formatting.
   - No extra spaces, no labels.

If the text does not contain a clear melting point, output “NaN”.
"""

def get_cid_from_smiles(smiles: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    r = requests.get(url)
    if r.status_code != 200:
        # Query PubChem using SMILES → Compound
        compounds = pubchempy.get_compounds(smiles, namespace='smiles')
        cid = compounds[0].cid
        return cid
    data = r.json()
    cid = data.get("IdentifierList", {}).get("CID", [None])[0]
    return cid         


def section_to_text(section):
    """
    Converts a PUG-View section (Melting Point) into a full text representation.
    """
    lines = []

    # Section heading
    if "TOCHeading" in section:
        lines.append(section["TOCHeading"])
    elif "Name" in section:
        lines.append(section["Name"])

    # Extract textual info
    for info in section.get("Information", []):
        # StringValue
        if "StringValue" in info:
            lines.append(info["StringValue"])

        # Value with markup
        if "Value" in info:
            val = info["Value"]
            if "StringWithMarkup" in val:
                lines.append(val["StringWithMarkup"][0]["String"])
            if "Number" in val:
                lines.append(str(val["Number"]))

        # Description text fields
        if "Description" in info:
            lines.append(info["Description"])

    # Recurse into subsections
    for subsection in section.get("Section", []):
        subtext = section_to_text(subsection)
        if subtext.strip():
            lines.append(subtext)

    return "\n".join(lines)


def find_melting_point_sections(record):
    """
    Recursively search the entire PUG-View record for Melting Point sections.
    Returns a list of matching sections.
    """
    matches = []

    def recurse(sec):
        heading = sec.get("TOCHeading", "").lower()
        name = sec.get("Name", "").lower()

        if heading == "melting point" or name == "melting point":
            matches.append(sec)

        for s in sec.get("Section", []):
            recurse(s)

    for top in record.get("Section", []):
        recurse(top)

    return matches


def get_melting_point_text(cid: int):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    r = requests.get(url)
    if r.status_code != 200:
        return None

    data = r.json()
    record = data.get("Record", {})
    sections = find_melting_point_sections(record)

    if not sections:
        return None

    # Convert each section to text and join
    return "\n\n".join(section_to_text(sec) for sec in sections)


def get_melting_point_from_smiles(smiles: str):
    cid = get_cid_from_smiles(smiles)
    
    if cid is None:
        print(f"Unable to resolve CID for SMILES: {smiles}")
        return None

    text = get_melting_point_text(cid)
    if not text:
        print(f"No melting point section found for CID {cid}")
        return None

    return text

# -------------------
# RDKit
# -------------------
def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)

# -------------------
# Wikipedia
# -------------------
def name_from_smiles(smiles: str) -> str:
    """
    Given a SMILES string, return the preferred name of the compound
    using PubChem (via PubChemPy).
    """
    # Query PubChem using SMILES → Compound
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')

    if not compounds:
        raise KeyError(f"No PubChem compound found for SMILES: {smiles}")

    compound = compounds[0]

    # PubChemPy automatically resolves several name fields
    if compound.iupac_name:
        return compound.iupac_name
    if compound.synonyms:
        return compound.synonyms[0]
    if compound.title:
        return compound.title

    raise KeyError(f"Compound found but no name available for SMILES: {smiles}")

def wikiscrape(smiles: str) -> str:
    name = name_from_smiles(smiles)
    print(f'Molecule: {name}')    
    
    site   = mwclient.Site('en.wikipedia.org')
    page   = site.pages[name]
    targets = ['MeltingPtC', 'MeltingPtK']
    
    if not page.exists:
        return np.nan
    
    # Follow redirect if present
    if page.redirect:
        page = page.redirects_to()
    
    # print(page.name)
    wikitext = page.text()
    
    result = None
    for line in wikitext.splitlines():
        for t in targets:
            if t in line:
                result = line.strip()
                break
        if result:
            break

    return result

# -------------------
# Main functions (single SMILES and file)
# -------------------
def single_smile(smiles: str, verbose: bool = True):
    """
    Given a SMILES string, extract melting point info and run the LLM.

    - If `smiles` is a non-empty string, executes the workflow.
    - Tries PubChem first; if unavailable, falls back to Wikipedia scraping.
    - Prints intermediate output and returns the LLM response text.
    """
    smiles = smiles.strip()
    smiles = canonicalize_smiles(smiles) # Canonicalize
    
    result = get_melting_point_from_smiles(smiles)
    
    if result is not None:
        source = "pubchem"
        
        if verbose:
            print("=== Melting Point Section ===")
            print("--------------------------")
            print(result)
            print("--------------------------")
            print()
        
        response = ollama.chat.completions.create(
            model=OLLAMA,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": result}
            ]
        )
        out = response.choices[0].message.content
        if verbose:
            print(f'Source: {source}')
            print(f'Melting Poing is: {out} K')
    else:
        source = "wiki"
        if verbose:
            print("\nMelting point is not available! Will now search on Wikipedia!\n")

        result = wikiscrape(smiles)

        # If Wikipedia didn't yield usable text, skip LLM call
        if (result is None) or (isinstance(result, float) and np.isnan(result)) or (isinstance(result, str) and not result.strip()):
            if verbose:
                print('No Wikipedia melting point found.')
            out = "NaN"
        else:
            try:
                response = ollama.chat.completions.create(
                    model=OLLAMA,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": result}
                    ]
                )
                out = response.choices[0].message.content
                if verbose:
                    print(f'Source: {source}')
                    print(f'Melting Poing is: {out} K')
            except Exception as e:
                if verbose:
                    print(f'Not found or LLM error: {e}')
                out = "NaN"
    cleaned = out if isinstance(out, str) and out.strip() else "NaN"
    return cleaned, source
        
def file_smiles(file: str):
    data = pd.read_csv(file)
    data = data.sample(50)

    ids, smiles_list, melt_points, sources = [], [], [], []
    for i in tqdm(range(len(data))):
        smi = str(data['SMILES'].iloc[i]).strip()
        print(f'Input: {smi}')
        
        tm, source = single_smile(smi, verbose=False)
        
        print(f'Melting Poing is: {tm} K')
        print(f'Source: {source}')
        
        ids.append(data['id'].iloc[i])
        smiles_list.append(smi)
        melt_points.append(tm)
        sources.append(source)

    result_df = pd.DataFrame({'id': ids, 'SMILES': smiles_list, 'Tm': melt_points, 'source': sources})
    result_df.to_csv('../data/res.csv', index=False)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract melting point from PubChem and process with LLM.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", type=str, help="The SMILES string of the molecule")
    group.add_argument("--file",   type=str, help="Path to .csv with smiles")
    args = parser.parse_args()

    if args.smiles:
        single_smile(args.smiles)
    else:
        file_smiles(args.file)
