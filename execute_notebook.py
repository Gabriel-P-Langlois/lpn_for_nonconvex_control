#!/usr/bin/env python3
"""
Robust notebook executor for SLURM batch jobs with Python 3.13
"""
import sys
import os
import asyncio
from pathlib import Path

# Fix for Python 3.13 event loop issues in batch mode
import nest_asyncio
nest_asyncio.apply()

from nbclient import NotebookClient
from nbformat import read, write

def setup_event_loop():
    """Ensure we have a working event loop for batch execution"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def execute_notebook_sync(nb_path):
    """Execute a notebook synchronously - works better in batch mode"""
    
    if not os.path.exists(nb_path):
        print(f"ERROR: Notebook {nb_path} not found", file=sys.stderr)
        return False
    
    out_path = nb_path.replace(".ipynb", "_executed.ipynb")
    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    
    print(f"Executing: {nb_path}")
    print(f"Notebook dir: {nb_dir}")
    
    try:
        # Read the notebook
        with open(nb_path) as f:
            nb = read(f, as_version=4)
        
        # Create the client with proper working directory
        client = NotebookClient(
            nb,
            timeout=None,  # No timeout for long-running cells
            kernel_name="python3",
            resources={'metadata': {'path': nb_dir}},
            # Important for batch execution
            force_raise_on_iopub_timeout=False,
            store_widget_state=False
        )
        
        # Setup event loop for batch mode
        loop = setup_event_loop()
        
        # Execute using the event loop directly
        async def execute_async():
            await client.async_execute()
        
        # Run the execution
        loop.run_until_complete(execute_async())
        
        # Save the executed notebook
        with open(out_path, "w") as f:
            write(nb, f)
        
        print(f"SUCCESS: Completed {nb_path}")
        print(f"Output saved to: {out_path}")
        return True
        
    except Exception as e:
        print(f"ERROR executing {nb_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execute_notebook.py <notebook_path>")
        sys.exit(1)
    
    # Execute and exit with appropriate code
    success = execute_notebook_sync(sys.argv[1])
    sys.exit(0 if success else 1)
