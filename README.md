### The amazing `uv`

Follow the instructions here to install uv - I recommend using the Standalone Installer approach at the very top:

https://docs.astral.sh/uv/getting-started/installation/

Start by running `uv self update` to make sure you're on the latest version of uv.

And now simply run:  
`uv sync`  
And marvel at the speed and reliability! If necessary, uv should install python 3.12, and then it should install all the packages.  

Checking that everything is set up nicely:  
1. Confirm that you now have a folder called '.venv' in your project root directory (agents)
2. If you run `uv python list` you should see a Python 3.12 version in your list (there might be several)
3. If you run `uv tool list` you should see crewai as a tool

Just FYI on using uv:  
With uv, you do a few things differently:  
- Instead of `pip install xxx` you do `uv add xxx` - it gets included in your `pyproject.toml` file and will be automatically installed next time you need it  
- Instead of `python my_script.py` you do `uv run my_script.py` which updates and activates the environment and calls your script  
- You don't actually need to run `uv sync` because uv does this for you whenever you call `uv run`  
- It's better not to edit pyproject.toml yourself, and definitely don't edit uv.lock. If you want to upgrade all your packages, run `uv lock --upgrade`

### Introduction to APIs in Python

Read: `https://chatgpt.com/share/68062432-43c8-8012-ad91-6311d4ad5858`