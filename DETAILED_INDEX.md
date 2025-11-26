# 📚 AI Tutorial Repository - Comprehensive Detailed Index

> **Repository**: AI-Tutorial-Codes-Included  
> **Last Updated**: November 2025  
> **Total Files**: 185+ Python and Jupyter Notebook files  
> **Categories**: 24+ distinct topic directories

This document provides a comprehensive index of all tutorials, implementations, and code resources contained within this repository. Use this index to quickly locate specific topics, frameworks, or implementations.

---

## 📋 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Agentic AI and Agents](#agentic-ai-and-agents)
   - [AI Agents Codes](#ai-agents-codes)
   - [Agentic AI Codes](#agentic-ai-codes)
   - [Agentic AI Memory](#agentic-ai-memory)
4. [Multi-Agent Frameworks](#multi-agent-frameworks)
5. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
6. [Agent Communication Protocol (ACP)](#agent-communication-protocol-acp)
7. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
8. [LLM Projects and Evaluation](#llm-projects-and-evaluation)
9. [Machine Learning Project Codes](#machine-learning-project-codes)
10. [Data Science](#data-science)
11. [Computer Vision](#computer-vision)
12. [Voice AI](#voice-ai)
13. [Security](#security)
14. [Robotics](#robotics)
15. [Adversarial Attacks](#adversarial-attacks)
16. [SHAP-IQ Explainability](#shap-iq-explainability)
17. [Root-Level Tutorials](#root-level-tutorials)
18. [Frameworks and Tools Index](#frameworks-and-tools-index)

---

## Repository Overview

This repository is a comprehensive collection of AI tutorials, implementations, and practical guides covering:

- **Agentic AI Systems**: Multi-agent frameworks, autonomous agents, and agent orchestration
- **LLM Integration**: OpenAI, Gemini, Mistral, Claude, and open-source models
- **RAG Systems**: Retrieval-augmented generation with various vector stores
- **MCP/ACP Protocols**: Model Context Protocol and Agent Communication Protocol implementations
- **Voice AI**: Speech recognition, synthesis, and voice-enabled assistants
- **Computer Vision**: Object tracking, image analysis, and CNN architectures
- **ML Pipelines**: End-to-end machine learning workflows and optimization

---

## Directory Structure

```
AI-Tutorial-Codes-Included/
├── A2A_Simple_Agent/                    # Agent-to-Agent protocol implementation
├── AI Agents Codes/                     # Core AI agent implementations
│   ├── Human Handoff - Parlant/         # Human handoff patterns
│   └── [40+ agent notebooks/scripts]
├── AI Agents and Agentic AI/            # Conceptual resources
├── Adversarial Attacks/                 # LLM security testing
├── Agent Communication Protocol/        # ACP implementations
│   └── Getting Started/                 # ACP beginner tutorials
├── Agentic AI Codes/                    # Advanced agentic implementations
├── Agentic AI Memory/                   # Memory systems for agents
├── Computer Vision/                     # CV tutorials
├── Data Science/                        # Data analysis and visualization
├── GPT-5/                              # GPT-5 and RouteLLM guides
├── LLM Evaluation/                      # LLM evaluation frameworks
├── LLM Projects/                        # LLM application projects
├── MCP Codes/                           # Model Context Protocol
├── ML Project Codes/                    # Machine learning pipelines
├── MLFlow for LLM Evaluation/           # MLFlow integration
│   └── OpenAI Tracing/                  # OpenAI observability
├── Mirascope/                           # Mirascope framework tutorials
├── OAuth 2.1 for MCP Servers/           # Authentication for MCP
├── RAG/                                 # RAG implementations
├── Robotics/                            # Robotics and LeRobot
├── SHAP-IQ/                             # Model explainability
├── Security/                            # AI security frameworks
├── Voice AI/                            # Speech and voice processing
└── [77 root-level notebooks/scripts]    # Standalone tutorials
```

---

## Agentic AI and Agents

### AI Agents Codes

Located in: `AI Agents Codes/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `neuro_symbolic_hybrid_agent_Marktechpost.ipynb` | Neuro-symbolic hybrid agent with logical planning | Neural networks, Logic programming |
| `agentic_multi_agent_rl_gridworld_Marktechpost.ipynb` | Reinforcement learning environment-acting agent | RL, Multi-agent coordination |
| `agentic_benchmarking_empirical_study_Marktechpost.ipynb` | Benchmarking reasoning strategies in agentic systems | Evaluation frameworks |
| `agentic_deep_rl_curriculum_ucb_meta_control_Marktechpost.ipynb` | Deep RL with curriculum progression and UCB planning | Deep RL, Meta-learning |
| `Computer_Use_Agent_Local_AI_Marktechpost.ipynb` | Computer-use agent with local AI models | Desktop automation, Local models |
| `enterprise_agentic_benchmarking_framework_Marktechpost.ipynb` | Enterprise AI benchmarking framework | Enterprise evaluation |
| `Value_Alignment_and_Ethics_in_Agentic_Systems_Marktechpost.ipynb` | Ethically aligned autonomous agents | Ethics, Value alignment |
| `advanced_stable_baselines3_trading_agent_marktechpost.py` | RL trading agents with Stable-Baselines3 | Trading, RL |
| `Agentic_AI_LangChain_AutoGen_HuggingFace_Marktechpost.ipynb` | Multi-agent systems with AutoGen, LangChain | AutoGen, LangChain, HuggingFace |
| `uagents_multi_agent_marketplace_Marktechpost.ipynb` | Multi-agent marketplace with uAgent | uAgent, Marketplace |
| `secure_ai_agent_with_guardrails_marktechpost.py` | Secure AI agent with guardrails and PII redaction | Security, Guardrails |
| `Langchain_Deepagents.ipynb` | LangChain's DeepAgents library | LangChain, DeepAgents |
| `LangChain_XGBoost_Agentic_Pipeline_Tutorial_Marktechpost.ipynb` | Conversational ML pipeline with LangChain + XGBoost | LangChain, XGBoost |
| `AI_Crypto_Agent_Secure_Comms_Marktechpost.ipynb` | Cryptographic agent with hybrid encryption | Cryptography, Security |
| `agentic_rag_tutorial_marktechpost.py` | Advanced agentic RAG system | RAG, Agentic retrieval |
| `supervisor_framework_crewai_gemini_marktechpost.py` | Hierarchical supervisor agent with CrewAI | CrewAI, Gemini |
| `ai_desktop_automation_agent_tutorial_Marktechpost.ipynb` | AI desktop automation agent | Desktop automation |
| `how_to_build_an_advanced_end_to_end_voice_ai_agent_using_hugging_face_pipelines.py` | Voice AI agent with HuggingFace | Voice AI, HuggingFace |
| `parlant.py` | Conversational AI with Parlant | Parlant |
| `advanced_ocr_ai_agent_Marktechpost.ipynb` | Multilingual OCR agent with EasyOCR | OCR, EasyOCR, OpenCV |
| `advanced_neural_agent_Marktechpost.ipynb` | Neural AI agent with adaptive learning | Neural networks |
| `Building Advanced MCP Agents with Multi-Agent Coordination.ipynb` | MCP agents with multi-agent coordination | MCP, Gemini |
| `Build a Complete Multi-Domain AI Web Agent Using Notte and Gemini` | Web agent with Notte and Gemini | Notte, Gemini |
| `Bioinformatics AI Agent with Biopython` | Bioinformatics agent for DNA/protein analysis | Biopython |
| `agent_lightning_prompt_optimization_Marktechpost.ipynb` | Agent development with Microsoft Agent-Lightning | Agent-Lightning |
| `Advanced AI Agent with Summarized Short Term and Vector-Based LongTerm Memory` | Memory-enhanced AI agent | Memory systems |
| `langgraph_time_travel_research_agent_Marktechpost.ipynb` | Research agent with LangGraph time-travel | LangGraph |
| `deep_research_agent_Marktechpost.ipynb` | Deep research agent with DuckDuckGo API | Research, Gemini |
| `hrm_braininspired_ai_agent_huggingface_marktechpost.py` | Brain-inspired hierarchical reasoning agent | HuggingFace |
| `graphagent_gemini_advanced_tutorial_Marktechpost.ipynb` | Graph-structured AI agent with Gemini | Graph agents, Gemini |
| `mle_agent_ollama_local_pipeline_Marktechpost.ipynb` | ML pipeline with MLE-Agent and Ollama | MLE-Agent, Ollama |
| `self_verifying_dataops_agent_local_hf_marktechpost.py` | Self-verifying data operations agent | HuggingFace, DataOps |
| `multi_agent_omics_integration_pipeline_Marktechpost.ipynb` | Multi-agent system for omics data | Bioinformatics |
| `wetlab_protocol_planner_codegen_Marktechpost.ipynb` | Wet-lab protocol planner with CodeGen | CodeGen, Lab automation |
| `agentic_data_infrastructure_strategy_qwen_marktechpost.py` | Data infrastructure with Qwen models | Qwen |
| `context_folding_llm_agent_long_horizon_Marktechpost.ipynb` | Context folding for long-horizon tasks | LLM agents |
| `Exploration_Agents_Problem_Solving_Marktechpost.ipynb` | Exploration agents for problem solving | Exploration |
| `agentic_ai_time_series_forecasting_darts_hf_Marktechpost.ipynb` | Time series forecasting agent | Darts, HuggingFace |
| `semantic_kernel_gemini_agent_Marktechpost.ipynb` | Semantic Kernel with Gemini | Semantic Kernel, Gemini |

#### Human Handoff - Parlant Subdirectory
| File | Description |
|------|-------------|
| `agent.py` | Agent implementation for human handoff |
| `handoff.py` | Handoff mechanism implementation |

### Agentic AI Codes

Located in: `Agentic AI Codes/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `advanced_multitool_agentic_ai_marktechpost.py` | Offline multi-tool reasoning agent | Planning, Error recovery |

### Agentic AI Memory

Located in: `Agentic AI Memory/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `memory_driven_agentic_ai_Marktechpost.ipynb` | Memory-powered AI with episodic experiences | Episodic memory, Semantic patterns |
| `spacy_agentic_ai_system_Marktechpost.ipynb` | Multi-agent reasoning with spaCy | spaCy, Knowledge graphs |
| `neural_memory_agents_continual_learning_Marktechpost.ipynb` | Neural memory agents with meta-learning | Meta-learning, Experience replay |
| `Model_Native_Agentic_AI_End_to_End_RL_Marktechpost.ipynb` | Model-native agent with end-to-end RL | Reinforcement learning |
| `Persistent_Memory_Personalised_Agentic_AI_Marktechpost.ipynb` | Persistent memory with decay and self-evaluation | Personalization, Memory decay |

---

## Multi-Agent Frameworks

### A2A Simple Agent

Located in: `A2A_Simple_Agent/`

| File | Description |
|------|-------------|
| `main.py` | Main entry point for A2A agent |
| `agent_executor.py` | Agent execution logic |
| `client.py` | Client implementation |
| `pyproject.toml` | Project configuration |
| `README.md` | Documentation |

---

## Model Context Protocol (MCP)

### MCP Codes

Located in: `MCP Codes/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `Model_Context_Protocol_Integration_Marktechpost.ipynb` | MCP for real-time resource and tool integration | MCP |

### OAuth 2.1 for MCP Servers

Located in: `OAuth 2.1 for MCP Servers/`

| File | Description |
|------|-------------|
| `server.py` | MCP server with OAuth 2.1 |
| `auth.py` | Authentication implementation |
| `config.py` | Configuration settings |
| `finance.py` | Finance-related MCP tools |
| `README.md` | Documentation |

---

## Agent Communication Protocol (ACP)

Located in: `Agent Communication Protocol/`

### Getting Started

| File | Description |
|------|-------------|
| `agent.py` | Weather agent implementation |
| `client.py` | ACP client |

---

## RAG (Retrieval-Augmented Generation)

Located in: `RAG/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `Semantic_Caching.ipynb` | Semantic LLM caching for RAG | Caching, Cost optimization |
| `agentic_rag_with_routing_and_self_check_marktechpost.py` | Agentic RAG with query routing | Query routing, Self-checking |
| `enterprise_ai_rag_guardrails_Marktechpost.ipynb` | Enterprise AI assistant with guardrails | Guardrails, Policy |
| `rag_evaluation.py` | RAG pipeline evaluation with synthetic data | Evaluation |

---

## LLM Projects and Evaluation

### LLM Projects

Located in: `LLM Projects/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `opik_local_llm_tracing_and_evaluation_marktechpost.py` | LLM tracing with Opik | Opik, Observability |

### LLM Evaluation

Located in: `LLM Evaluation/`

| File | Description |
|------|-------------|
| `LLM_Arena_as_a_Judge.ipynb` | LLM arena evaluation |
| `llm_arena_as_a_judge.py` | Python implementation |
| `README.md` | Documentation |

### MLFlow for LLM Evaluation

Located in: `MLFlow for LLM Evaluation/`

| File | Description |
|------|-------------|
| `MLFlow_Intro.ipynb` | MLFlow introduction |
| `GitHub_Notebook_Render_Error.md` | Troubleshooting guide |

#### OpenAI Tracing Subdirectory
| File | Description |
|------|-------------|
| `guardrails.py` | Guardrails implementation |
| `multi_agent_demo.py` | Multi-agent demonstration |

### GPT-5

Located in: `GPT-5/`

| File | Description |
|------|-------------|
| `GPT_5.ipynb` | GPT-5 model capabilities guide |
| `RouteLLM.ipynb` | RouteLLM for LLM usage optimization |

---

## Machine Learning Project Codes

Located in: `ML Project Codes/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `regression_language_model_transformer_rlm_tutorial_Marktechpost.ipynb` | Transformer-based regression language model | Transformers |
| `advanced_litserve_multi_endpoint_api_tutorial_marktechpost.py` | Multi-endpoint ML APIs with LitServe | LitServe, Batching |
| `Advanced_Ivy_Framework_Agnostic_ML_Tutorial_Marktechpost.ipynb` | Framework-agnostic ML with Ivy | Ivy |
| `lightly_ai_self_supervised_active_learning_Marktechpost.ipynb` | Self-supervised learning with Lightly AI | Lightly AI |
| `tpot_advanced_pipeline_optimization_marktechpost.py` | AutoML pipeline optimization with TPOT | TPOT |
| `advanced_jax_flax_optax_training_pipeline_Marktechpost.ipynb` | Advanced architectures with JAX/Flax/Optax | JAX, Flax, Optax |
| `meta_hydra_advanced_implementation_Marktechpost.ipynb` | ML experiment pipelines with Hydra | Hydra |
| `advanced_reflex_reactive_webapp_Marktechpost.ipynb` | Reactive web apps with Reflex | Reflex |
| `Advanced_PySpark_End_to_End_Tutorial_Marktechpost.ipynb` | End-to-end ML with PySpark | Apache Spark, PySpark |
| `A Coding Implementation to End-to-End Transformer Model Optimization with Hugging Face Optimum, ONNX Runtime, and Quantization.ipynb` | Model optimization with ONNX | ONNX, Quantization |
| `Build a Complete End-to-End NLP Pipeline with Gensim` | NLP pipeline with Gensim | Gensim |
| `Building an Advanced Convolutional Neural Network with Attention for DNA Sequence Classification and Interpretability.ipynb` | CNN for DNA sequence classification | CNN, Attention |
| `Code Implementation to Master DeepSpeed` | DeepSpeed mastery | DeepSpeed |
| `Getting Started with Asyncio.ipynb` | Async Python programming | Asyncio |
| `LLM_Parameters.ipynb` | LLM parameters exploration | LLM |
| `Quantum_State_Evolution_and_Entanglement_QuTiP_Tutorial_Marktechpost.ipynb` | Quantum computing with QuTiP | QuTiP |
| `custom_gpt_local_colab_chat_marktechpost.py` | Custom GPT local chat | Local LLM |
| `gluonTS_advanced_multimodel_tutorial_Marktechpost.ipynb` | Time series with GluonTS | GluonTS |
| `huggingface_trackio_advanced_tutorial_Marktechpost.ipynb` | HuggingFace Trackio | HuggingFace |
| `zarr_implementation_tutorial.ipynb` | Zarr data storage | Zarr |

---

## Data Science

Located in: `Data Science/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `advanced_pygwalker_visual_analysis_marktechpost.py` | Interactive analytics with PyGWalker | PyGWalker |
| `advanced_textual_data_dashboard_Marktechpost.ipynb` | Terminal-based dashboard with Textual | Textual |
| `Active_Learning.ipynb` | Supervised AI without annotated data | Active learning |
| `Advanced_Bokeh_Interactive_Dashboard_Marktechpost.ipynb` | Interactive dashboards with Bokeh | Bokeh |
| `Advanced_PyTest_Custom_Plugins_Fixtures_Tutorial_Marktechpost.ipynb` | PyTest plugins and fixtures | PyTest |
| `Building an End-to-End Data Science Workflow with Machine Learning, Interpretability, and Gemini AI Assistance.ipynb` | Data science workflow with Gemini | Gemini |
| `Imbalanced_Datasets.ipynb` | Handling imbalanced datasets | ML |
| `Unified_Tool_Orchestration_Framework_Marktechpost.ipynb` | Tool orchestration framework | Orchestration |
| `dash_plotly_local_online_dashboard_Marktechpost.ipynb` | Dashboards with Dash/Plotly | Dash, Plotly |
| `matlab_python_oct2py_colab_tutorial_Marktechpost.ipynb` | MATLAB-Python integration | Oct2Py |

---

## Computer Vision

Located in: `Computer Vision/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `optuna_advanced_hpo_marktechpost.py` | Hyperparameter optimization with Optuna | Optuna |
| `How to Master Advanced TorchVision v2 Transforms, MixUp, CutMix, and Modern CNN Training for State-of-the-Art Computer Vision.ipynb` | TorchVision transforms and CNN training | TorchVision, MixUp, CutMix |

---

## Voice AI

Located in: `Voice AI/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `agentic_voice_ai_autonomous_assistant_Marktechpost.ipynb` | Agentic voice AI assistant | Voice AI |
| `voice_ai_whisperx_advanced_tutorial_Marktechpost.ipynb` | Voice AI pipeline with WhisperX | WhisperX |
| `guide_to_building_an_end_to_end_speech_enhancement_and_recognition_pipeline_with_speechbrain.py` | Speech enhancement with SpeechBrain | SpeechBrain |
| `how_to_build_an_advanced_end_to_end_voice_ai_agent_using_hugging_face_pipelines.py` | Voice AI with HuggingFace | HuggingFace |

---

## Security

Located in: `Security/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `building_a_hybrid_rule_based_and_machine_learning_framework_to_detect_and_defend_against_jailbreak_prompts_in_llm_systems.py` | Jailbreak defense framework | Security, ML |

---

## Robotics

Located in: `Robotics/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `lerobot_pusht_bc_tutorial_marktechpost.py` | LeRobot behavioral cloning tutorial | LeRobot |

---

## Adversarial Attacks

Located in: `Adversarial Attacks/`

| File | Description | Key Technologies |
|------|-------------|------------------|
| `Single-Turn Attacks.ipynb` | Single-turn adversarial attacks with deepteam | deepteam |

---

## SHAP-IQ Explainability

Located in: `SHAP-IQ/`

| File | Description |
|------|-------------|
| `Intro_to_SHAP_IQ.ipynb` | SHAP-IQ introduction |
| `SHAP_IQ_Visuals.ipynb` | SHAP-IQ visualization tutorials |

---

## Mirascope Framework

Located in: `Mirascope/`

| File | Description |
|------|-------------|
| `Chain_of_Thought.ipynb` | Chain of thought prompting |
| `Knowledge_Graph.ipynb` | Knowledge graph creation with LLM |
| `Removing_Semantic_Duplicates.ipynb` | Semantic deduplication |
| `Self_Refine.ipynb` | Self-refinement techniques |

---

## Root-Level Tutorials

These notebooks and scripts are located in the repository root and cover diverse AI topics:

### Multi-Agent and Orchestration

| File | Description |
|------|-------------|
| `A_Coding_Guide_to_ACP_Systems_Marktechpost.ipynb` | Agent Communication Protocol systems |
| `Advanced_PEER_MultiAgent_Tutorial_Marktechpost.ipynb` | PEER multi-agent framework |
| `AutoGen_SemanticKernel_Gemini_Flash_MultiAgent_Tutorial_Marktechpost.ipynb` | AutoGen + Semantic Kernel + Gemini |
| `AutoGen_TeamTool_RoundRobin_Marktechpost.ipynb` | AutoGen round-robin workflows |
| `LangGraph_Gemini_MultiAgent_Research_Team_Marktechpost.ipynb` | Multi-agent research with LangGraph |
| `agent2agent_collaboration_Marktechpost.ipynb` | Agent2Agent collaborative problem solving |
| `beeai_multi_agent_workflow_Marktechpost.ipynb` | BeeAI multi-agent workflows |
| `gemini_autogen_multiagent_framework_Marktechpost.ipynb` | AutoGen + Gemini multi-agent framework |
| `nomic_gemini_multi_agent_ai_Marktechpost.ipynb` | Nomic + Gemini multi-agent |
| `primisai_nexus_multi_agent_workflow_Marktechpost.ipynb` | PrimisAI Nexus multi-agent |
| `openai_agents_multiagent_research_Marktechpost.ipynb` | OpenAI agents for research |
| `production_ready_custom_ai_agents_workflows_Marktechpost.ipynb` | Production-ready agent workflows |
| `advanced_langgraph_multi_agent_pipeline_Marktechpost.ipynb` | LangGraph multi-agent pipelines |
| `advanced_google_adk_multi_agent_tutorial_Marktechpost.ipynb` | Google ADK multi-agent systems |

### AI Agents and Assistants

| File | Description |
|------|-------------|
| `Self_Improving_AI_Agent_with_Gemini_Marktechpost.ipynb` | Self-improving agent with Gemini |
| `gemini_agent_network_Marktechpost.ipynb` | Asynchronous AI agent network |
| `sage_ai_agent_gemini_implementation_Marktechpost.ipynb` | SAGE framework agent |
| `graph_agent_framework_with_gemini_Marktechpost.ipynb` | Graph agent framework |
| `nebius_llama3_multitool_agent_Marktechpost.ipynb` | Nebius + Llama3 multi-tool agent |
| `cipher_memory_agent_Marktechpost.ipynb` | Cipher workflow for AI agents |
| `Live_Python_Execution_and_Validation_Agent_Marktechpost.ipynb` | Live Python execution agent |
| `Custom_Tool_For_AI_Agent_Marktechpost.ipynb` | Custom tools for AI agents |
| `Customizable_MultiTool_AI_Agent_with_Claude_Marktechpost (1).ipynb` | Multi-tool agent with Claude |
| `advanced_ai_agent_hugging_face_marktechpost.py` | AI agent with HuggingFace |
| `tinydev_gemini_implementation_Marktechpost.ipynb` | TinyDev AI-powered applications |
| `streamlit_ai_agent_multitool_interface_Marktechpost.ipynb` | Streamlit AI agent interface |

### Mistral AI

| File | Description |
|------|-------------|
| `Getting_Started_with_Mistral_Agents_API.ipynb` | Mistral Agents API introduction |
| `Mistral_Devstral_Compact_Loading_Marktechpost.ipynb` | Mistral Devstral loading |
| `Mistral_Guardrails.ipynb` | Mistral guardrails |
| `agent_orchestration_with_mistral_agents_api.py` | Agent orchestration with Mistral |
| `how to enable function calling in Mistral Agents.py` | Function calling in Mistral |
| `mistral_devstral_compact_loading_marktechpost.py` | Devstral compact loading |

### LangGraph and LangChain

| File | Description |
|------|-------------|
| `GraphAIAgent_LangGraph_Gemini_Workflow_Marktechpost.ipynb` | Iterative AI workflow with LangGraph |
| `prolog_gemini_langgraph_react_agent_Marktechpost.ipynb` | Prolog + LangGraph ReAct agent |
| `Jina_LangChain_Gemini_AI_Assistant_Marktechpost.ipynb` | Jina + LangChain + Gemini assistant |
| `ollama_langchain_tutorial_marktechpost.py` | Ollama + LangChain tutorial |

### MCP (Model Context Protocol)

| File | Description |
|------|-------------|
| `mcp_gemini_agent_tutorial_Marktechpost.ipynb` | MCP-powered agent with Gemini |
| `Context_Aware_Assistant_MCP_Gemini_LangChain_LangGraph_Marktechpost.ipynb` | Context-aware MCP assistant |
| `custom_mcp_tools_integration_with_fastmcp_marktechpost.py` | Custom MCP tools with FastMCP |

### Data Processing and Pipelines

| File | Description |
|------|-------------|
| `Synthetic_Data_Creation.ipynb` | Synthetic data creation with SDV |
| `polars_sql_analytics_pipeline_Marktechpost.ipynb` | Polars SQL analytics |
| `lilac_functional_data_pipeline_Marktechpost.ipynb` | Lilac data pipelines |
| `dagster_advanced_pipeline_Marktechpost.ipynb` | Dagster pipeline orchestration |
| `Modin_Powered_DataFrames_Marktechpost.ipynb` | Modin DataFrames |
| `async_config_tutorial_Marktechpost.ipynb` | Async configuration patterns |
| `advanced_async_python_sdk_tutorial_Marktechpost.ipynb` | Async Python SDK development |

### Web Scraping and Search

| File | Description |
|------|-------------|
| `Enhanced_BrightData_Gemini_Scraper_Tutorial_Marktechpost.ipynb` | BrightData web scraper |
| `Competitive_Analysis_with_ScrapeGraph_Gemini_Marktechpost.ipynb` | ScrapeGraph competitive analysis |
| `advanced_serpapi_tutorial_Marktechpost.ipynb` | SerpAPI integration |
| `smartwebagent_tavily_gemini_webintelligence_marktechpost2.py` | Web intelligence with Tavily |

### Bioinformatics and Research

| File | Description |
|------|-------------|
| `BioCypher_Agent_Tutorial_Marktechpost.ipynb` | BioCypher agent |
| `PyBEL_BioKG_Interactive_Tutorial_Marktechpost.ipynb` | PyBEL knowledge graphs |
| `paperqa2_gemini_research_agent_Marktechpost.ipynb` | PaperQA2 research agent |
| `advanced_pubmed_research_assistant_tutorial_Marktechpost.ipynb` | PubMed research assistant |

### AI Frameworks and Tools

| File | Description |
|------|-------------|
| `Cognee_Agent_Tutorial_with_HuggingFace_Integration_Marktechpost.ipynb` | Cognee agent with HuggingFace |
| `CrewAI_Gemini_Workflow_Marktechpost.ipynb` | CrewAI + Gemini workflow |
| `UAgents_Gemini_Event_Driven_Tutorial_Marktechpost.ipynb` | uAgents event-driven tutorial |
| `Gemini_Pandas_Agent_Marktechpost.ipynb` | Gemini DataFrame agent |
| `Advanced_AI_Evaluator_Enterprise_Grade_Framework_Marktechpost.ipynb` | Enterprise AI evaluator |
| `pipecat_huggingface_implementation_Marktechpost.ipynb` | Pipecat + HuggingFace |
| `self_hosted_llm_ollama_Marktechpost.ipynb` | Self-hosted LLM with Ollama |
| `griffe_ai_code_analyzer_Marktechpost.ipynb` | Griffe code analyzer |
| `advanced_dspy_qa_Marktechpost.ipynb` | DSPy QA system |
| `Lyzr_Chatbot_Framework_Implementation_Marktechpost.ipynb` | Lyzr chatbot framework |

### Code Execution and Development

| File | Description |
|------|-------------|
| `daytona_secure_ai_code_execution_tutorial_Marktechpost.ipynb` | Secure code execution with Daytona |
| `Smart_Python_to_R_Converter_with_Gemini_Validation_Marktechpost (2).ipynb` | Python to R converter |
| `parsl_ai_agent_pipeline_marktechpost.py` | Parsl AI agent pipeline |

### Visualization and Tracking

| File | Description |
|------|-------------|
| `roboflow_supervision_advanced_tracking_analytics_pipeline_Marktechpost.ipynb` | Object tracking with Roboflow |

### Financial and Market Analysis

| File | Description |
|------|-------------|
| `openbb_advanced_portfolio_market_intelligence_Marktechpost.ipynb` | Portfolio intelligence with OpenBB |
| `network.ipynb` | Financial agent networking |
| `inflation_agent.py` | Inflation tracking agent |
| `emi_agent.py` | EMI calculation agent |

### Other Tutorials

| File | Description |
|------|-------------|
| `Presidio.ipynb` | PII detection with Microsoft Presidio |
| `Upstage_Groundedness_Check_Tutorial_Marktechpost.ipynb` | Groundedness verification |
| `Pyversity.ipynb` | Pyversity tutorial |
| `JSON_Prompting.ipynb` | JSON prompting techniques |
| `guide_to_building_an_end_to_end_speech_enhancement_and_recognition_pipeline_with_speechbrain.py` | SpeechBrain ASR pipeline |

---

## Frameworks and Tools Index

Quick reference for finding tutorials by framework or tool:

### LLM Providers

| Provider | Tutorials |
|----------|-----------|
| **OpenAI/GPT** | `GPT-5/`, `openai_agents_multiagent_research_Marktechpost.ipynb`, LLM Evaluation |
| **Google Gemini** | 30+ tutorials including agent networks, DataFrame agents, research assistants |
| **Mistral** | `Getting_Started_with_Mistral_Agents_API.ipynb`, function calling, guardrails |
| **Claude/Anthropic** | `Customizable_MultiTool_AI_Agent_with_Claude_Marktechpost.ipynb` |
| **Llama** | `nebius_llama3_multitool_agent_Marktechpost.ipynb` |
| **Ollama** | `self_hosted_llm_ollama_Marktechpost.ipynb`, `ollama_langchain_tutorial_marktechpost.py` |

### Agent Frameworks

| Framework | Tutorials |
|-----------|-----------|
| **LangChain** | Multiple tutorials for agents, RAG, and pipelines |
| **LangGraph** | Research agents, multi-agent pipelines, time-travel |
| **AutoGen** | Multi-agent frameworks, round-robin workflows |
| **CrewAI** | Supervisor frameworks, Gemini integration |
| **uAgents** | Event-driven tutorials, marketplace |
| **BeeAI** | Multi-agent workflows |
| **PrimisAI Nexus** | Multi-agent workflows with OpenAI |

### ML/DL Frameworks

| Framework | Tutorials |
|-----------|-----------|
| **PyTorch** | TorchVision transforms, CNN training |
| **JAX/Flax** | Advanced training pipelines |
| **HuggingFace** | Voice AI, agent integration, optimizations |
| **Stable-Baselines3** | RL trading agents |
| **TPOT** | AutoML pipelines |
| **Ivy** | Framework-agnostic ML |

### Data Tools

| Tool | Tutorials |
|------|-----------|
| **Polars** | SQL analytics pipelines |
| **Pandas** | Gemini DataFrame agent |
| **Modin** | Accelerated DataFrames |
| **PySpark** | End-to-end ML pipelines |

### Visualization

| Tool | Tutorials |
|------|-----------|
| **Streamlit** | AI agent interfaces |
| **Dash/Plotly** | Interactive dashboards |
| **Bokeh** | Interactive visualizations |
| **PyGWalker** | Visual analysis |
| **Textual** | Terminal dashboards |

### Protocols

| Protocol | Tutorials |
|----------|-----------|
| **MCP** | `MCP Codes/`, OAuth 2.1, FastMCP integration |
| **ACP** | `Agent Communication Protocol/`, weather agent |
| **A2A** | `A2A_Simple_Agent/`, financial agents |

---

## Statistics Summary

| Category | Count |
|----------|-------|
| Total Directories | 24+ |
| Total Files | 185+ |
| Jupyter Notebooks | 150+ |
| Python Scripts | 35+ |
| Agent-related Tutorials | 60+ |
| Multi-agent Tutorials | 25+ |
| RAG Tutorials | 4 |
| Voice AI Tutorials | 4 |
| Computer Vision Tutorials | 2 |
| Data Science Tutorials | 10+ |
| ML Pipeline Tutorials | 20+ |

---

## Quick Start Guide

1. **For Beginners**: Start with `Getting_Started_with_Mistral_Agents_API.ipynb` or `Agent Communication Protocol/Getting Started/`

2. **For Multi-Agent Systems**: Explore `AI Agents Codes/` and multi-agent tutorials in the root

3. **For RAG Applications**: Check `RAG/` folder and `agentic_rag_tutorial_marktechpost.py`

4. **For Voice AI**: See `Voice AI/` folder

5. **For MCP/ACP Protocols**: Visit `MCP Codes/`, `OAuth 2.1 for MCP Servers/`, and `Agent Communication Protocol/`

6. **For ML Pipelines**: Explore `ML Project Codes/` and `Data Science/`

---

## Contributing

Feel free to contribute by:
- Adding new tutorials
- Updating existing implementations
- Improving documentation
- Fixing bugs in code samples

---

## License

Please refer to the repository's LICENSE file for licensing information.

---

*This index was auto-generated and provides a comprehensive overview of the repository contents for easy navigation and reference.*
