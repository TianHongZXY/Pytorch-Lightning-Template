import json
from typing import List, Dict
from tqdm import tqdm


class DataProcessor(object):
  """Base class for loading data from files in common formats"""

  @classmethod
  def _read_json(cls, input_file) -> Dict:
    """Reads a JSON file."""
    with open(input_file, "r") as f:
      return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file) -> List[Dict]:
    """Reads a JSON Lines file."""
    with open(input_file, "r") as f:
      return [json.loads(ln) for ln in tqdm(f.readlines()) if ln]

  @classmethod
  def _read_txt(cls, input_file) -> List[str]:
    """Reads a txt file."""
    with open(input_file, "r") as f:
      return f.readlines()

