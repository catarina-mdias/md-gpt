"""
Lightweight MCP server that ONLY lists and reads files inside ./exams.
No OCR, no PDF parsing. 100% pip-installable.
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import base64

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXAMS_DIR = Path(os.getenv("EXAMS_DIR", "exams")).resolve()
EXAMS_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
   name="ExamsMCPServer",
   json_response=True,
   stateless_http=False,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ExamInfo(BaseModel):
   id: int
   filename: str
   path: str


class FileContent(BaseModel):
   filename: str
   content_base64: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_exam_files() -> List[Path]:
   """Return all files in EXAMS_DIR (PDF or otherwise)."""
   return sorted(EXAMS_DIR.glob("*"))


# ---------------------------------------------------------------------------
# MCP RESOURCES & TOOLS
# ---------------------------------------------------------------------------
@mcp.resource("exam://list")
def list_exams() -> List[ExamInfo]:
   """
   List all files inside the EXAMS_DIR.
   """
   files = _iter_exam_files()
   return [
       ExamInfo(
           id=i,
           filename=f.name,
           path=str(f.relative_to(EXAMS_DIR)),
       )
       for i, f in enumerate(files)
   ]


@mcp.tool()
def get_exam_file(id: Optional[int] = None, filename: Optional[str] = None) -> FileContent:
   """
   Returns the raw file content (base64-encoded).
   Your frontend or backend decides how to convert it later.
   """
   files = _iter_exam_files()


   if id is None and filename is None:
       raise ValueError("Provide either id or filename")


   if id is not None:
       if id < 0 or id >= len(files):
           raise ValueError(f"Invalid exam id {id}")
       target = files[id]
   else:
       target = EXAMS_DIR / filename
       if not target.exists():
           raise FileNotFoundError(f"No such exam file: {filename}")


   content = target.read_bytes()
   encoded = base64.b64encode(content).decode()


   return FileContent(
       filename=target.name,
       content_base64=encoded,
   )

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
   mcp.run(transport="stdio")

