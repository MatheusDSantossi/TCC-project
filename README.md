# TCC - Chatbot com Gemini API e Embeddings para Supermercado/E-commerce

Este repositório contém todos os arquivos relacionados ao meu projeto de Trabalho de Conclusão de Curso (TCC) na Universidade Federal Rural de Pernambuco (UFRPE).

O objetivo deste projeto é desenvolver um chatbot inteligente utilizando a poderosa Gemini API e técnicas de embeddings para oferecer uma experiência de compra otimizada além de poder ter uma melhor expirência em um ambiente de supermercado/e-commerce.

## Funcionalidades do Chatbot (teste):

- **Atendimento personalizado**: O chatbot será capaz de interagir com os clientes de forma personalizada, entendendo suas necessidades e respondendo a perguntas específicas sobre produtos.
- **Recomendação de produtos**: Através da análise do histórico de compras e preferências do cliente, o chatbot poderá realizar recomendações de produtos relevantes.
- **Auxílio na navegação**: O chatbot guiará o cliente pelo catálogo de produtos, facilitando a busca e localização de itens específicos.
- **Finalização da compra**: O chatbot integrará com o sistema de e-commerce para permitir que os clientes finalizem suas compras diretamente na interface do chat.
- **Suporte ao cliente**: O chatbot estará disponível para responder a perguntas frequentes, solucionar problemas e oferecer suporte aos clientes durante todo o processo de compra.

## Tecnologias Utilizadas:

- **Gemini API**: A API de última geração da Google para processamento de linguagem natural e geração de texto.
- **Embeddings**: Técnicas de aprendizado de máquina para representar produtos e informações em um espaço vetorial, permitindo a realização de buscas e recomendações personalizadas.
- **Python**: Linguagem de programação utilizada para desenvolver o chatbot.
- **Bibliotecas relevantes**:  - `google.generativeai`: Biblioteca para inteligência artificial da Google.
    - `numpy`: Biblioteca para computação numérica em Python.
    - `sklearn.metrics.pairwise`: Submódulo do scikit-learn para cálculo de similaridade de cosseno.
    - `google.colab`: Biblioteca para acessar dados do Google Colab.
    - `pandas`: Biblioteca para manipulação e análise de dados em Python.

## Estrutura do Repositório (teste):

- `/data`: Contém os dados utilizados para treinar o modelo de embeddings e alimentar o chatbot (ex: base de dados de produtos).
- `/notebooks`: Notebooks Jupyter com experimentos, análises e desenvolvimento do modelo de embeddings.
- `/src`: Código fonte do chatbot, incluindo a implementação da API Gemini e lógica de interação com o usuário.
- `/tests`: Arquivos de teste para garantir a qualidade do código e o bom funcionamento do chatbot.

## Como Executar o Projeto:

1. Clone este repositório:
   ```bash
   git clone https://github.com/MatheusDSantossi/TCC-project.git
