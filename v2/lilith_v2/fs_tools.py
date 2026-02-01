import os
from typing import List, Dict, Union, Any

class FileSystemTools:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def _validate_path(self, path: str) -> str:
        """Ensure path is within root_path."""
        abs_path = os.path.abspath(os.path.join(self.root_path, path))
        if not abs_path.startswith(self.root_path):
            raise ValueError(f"Path '{path}' is outside workspace root.")
        return abs_path

    def read_file(self, path: str) -> str:
        """Read content of a file."""
        target = self._validate_path(path)
        try:
            with open(target, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{path}' not found."
        except Exception as e:
            return f"Error reading '{path}': {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        target = self._validate_path(path)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to '{path}'."
        except Exception as e:
            return f"Error writing '{path}': {str(e)}"

    def list_dir(self, path: str = ".") -> Union[List[str], str]:
        """List contents of a directory."""
        target = self._validate_path(path)
        try:
            return os.listdir(target)
        except Exception as e:
            return f"Error listing '{path}': {str(e)}"
