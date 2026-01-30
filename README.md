# Tropical Disease Diagnosis with RAG and Fine-Tuned models
This repository contains code and notebooks for building two systems: a Retrieval-Augmented Generation (RAG) system and a fine-tuned language model for diagnosing tropical diseases. The RAG system leverages a vector database to retrieve relevant information, while the fine-tuned model is trained to provide accurate diagnoses based on the retrieved data.
## RAG System
The system use vector database Qdrant to store and retrieve embeddings using three type of vectors, with the help of Colpali model to generate embeddings. The RAG system is implemented in the `rag-tropical-disease-medgemma.ipynb` notebook.
## Fine-Tuned Language Model
The fine-tuned language model is based on the LLaMA architecture and is trained using supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) techniques. The training process is documented in the `sft_finetune.ipynb` and `dpo-training.ipynb` notebooks.
## Notebooks
- `extract_clean_text.ipynb`: Notebook for extracting and cleaning text data for training
- `create_dataset.ipynb`: Notebook for creating datasets using prompt that call Gemini API to generate QAs
- `sft_finetune.ipynb`: Contains code for supervised fine-tuning of the LLaMA model
- `dpo-training.ipynb`: Details the DPO training process for the language model
- `rag-tropical-disease-medgemma.ipynb`: demo pipeline of RAG system for tropical disease diagnosis
- `sft-tropical-disease-llama.ipynb`: Demo pipeline of fine-tuned LLaMA model for tropical disease diagnosis using SFT
- `dpo-tropical-disease-llama.ipynb`: demo pipeline of fine-tuned LLaMA model for tropical disease diagnosis using DPO
- `base-tropical-disease-medgemma.ipynb`: Base model setup for tropical disease diagnosis
## Requirements
- Python 3.8+
- Jupyter Notebook
- Required libraries: `transformers`, `qdrant-client`, `wandb`, `huggingface_hub`, `google-api-python-client`
- Access to Qdrant vector database and Hugging Face Hub.
- Access to some tropical books like Hunter Tropical Medicine, CDC Yellow Book, etc.
## Contributor
- Viet Tin Le (10422078@student.vgu.edu.vn)
- Khoi Nguyen Nguyen (10422058@student.vgu.edu.vn)
- Tri An Yamashita (10422004@student.vgu.edu.vn)
- Tran Quoc Dat Nguyen (10422017@student.vgu.edu.vn)
- Thao Vy Nguyen (10421067@student.vgu.edu.vn)
