# General Agentic Memory (GAM)

[![arXiv](https://img.shields.io/badge/arXiv-2511.18423-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.18423)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## About

**General Agentic Memory (GAM)** is a next-generation memory framework for AI agents that uses a Just-in-Time (JIT) approach. It features a dual-agent architecture with a Memorizer that builds structured memory and a Researcher that performs deep research at runtime to generate context-aware responses. GAM achieves state-of-the-art performance on benchmarks like LoCoMo, HotpotQA, RULER, and NarrativeQA.

## Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/raj2237/General-Agentic-Memory.git
cd General-Agentic-Memory

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Configuration

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"
```

### Basic Usage

```python
from gam import MemoryAgent, ResearchAgent, OpenAIGenerator, OpenAIGeneratorConfig
from gam import InMemoryMemoryStore, InMemoryPageStore

# Configure generator
config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key="your_api_key"
)
generator = OpenAIGenerator.from_config(config)

# Initialize stores
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

# Create memory agent
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# Memorize documents
memory_agent.memorize("Your document text here")

# Create research agent and query
research_agent = ResearchAgent(
    page_store=page_store,
    memory_store=memory_store,
    generator=generator
)

result = research_agent.research("Your question here")
print(result.integrated_memory)
```

Check `examples/quickstart/` for more examples.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{yan2025general,
  title={General Agentic Memory Via Deep Research},
  author={Yan, BY and Li, Chaofan and Qian, Hongjin and Lu, Shuqi and Liu, Zheng},
  journal={arXiv preprint arXiv:2511.18423},
  year={2025}
}
```
