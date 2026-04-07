# Contributing Guidelines

This is a **Mini Project** for the B.Tech Computer Science and Engineering program at MBCET.

---

## 👥 Team Members

| Name | Roll Number |
|------|-------------|
| Julia Mariam John | B23CS2137 |
| Nirmel B Joseph | B23CS2148 |
| Rohith NS | B23CS2156 |

**Project Guide:** Mr. Praveen J.S, Assistant Professor, Department of Computer Science and Engineering

---

## 📋 Project Overview

**Title:** Hybrid Knowledge-Graph-Enhanced Retrieval-Augmented Generation for Academic Information Systems

**SDG Alignment:**
- SDG 4 (Quality Education) - Improving structured access to academic information
- SDG 9 (Industry, Innovation and Infrastructure) - Hybrid intelligent information retrieval systems

---

## 🚀 For Team Members

### Development Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd mbcet-chunking-pipeline
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Tests**
   ```bash
   pytest -v
   ```

---

## 📝 Workflow Guidelines

### Branch Naming

- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `docs/<name>` - Documentation updates

### Commit Messages

Use clear, descriptive commit messages:
```
feat: add knowledge graph builder
fix: resolve PDF table extraction issue
docs: update README with installation steps
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and test locally
3. Push and create a Pull Request
4. Request review from team members
5. Merge after approval

---

## 🧪 Testing

Before pushing changes, ensure:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

---

## 📁 Module Responsibilities

| Module | Description |
|--------|-------------|
| `scraper/` | Web scraping and PDF processing |
| `chunker/` | Semantic chunking and entity extraction |
| `config.py` | Configuration management |
| `main.py` | CLI entry point |

---

## 🕸️ Knowledge Graph Data Maintenance

- The authoritative teaching relationship source is `data/entities/teaching_assignments.json`.
- Use valid entity IDs only:
  - Faculty IDs must exist in `data/entities/faculty.json`
  - Course IDs must exist in `data/entities/courses.json`
- Keep mappings deterministic and deduplicated (same faculty-course pair should appear once).
- Regenerate the graph artifact after changes:
  ```bash
  python main.py --stage kg
  ```

---

## 📚 References

1. Linders, J. and Tomczak, J.M., "Knowledge graph-extended retrieval augmented generation for question answering," *Applied Intelligence*, vol. 55, 2025.
2. Li, Z. et al., "Retrieval-augmented generation for educational application: A systematic survey," *Computers and Education: Artificial Intelligence*, vol. 8, 2025.

---

## 📄 License

This project is licensed under the MIT License.
