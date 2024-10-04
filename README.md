# 🧠 Interface de IA com Flask - Especialista em Farmácia

Este projeto implementa uma interface de IA desenvolvida com Flask, que integra módulos para transcrição de áudio, descrição de imagens, geração de textos e processamento de vídeos. Além disso, inclui uma IA altamente especializada na área de farmácia, capaz de responder a perguntas e fornecer recomendações com base em informações detalhadas e precisas.

## 🔧 Funcionalidades

### 1. **Transcrição de Áudio**
Permite o envio de arquivos de áudio para transcrição utilizando o modelo `Whisper` da OpenAI.

- **Entrada:** Arquivo de áudio (mp3, wav, etc.)
- **Saída:** Transcrição do áudio em texto.

### 2. **Descrição de Imagens**
Utiliza o modelo `CLIP` para descrever o conteúdo de imagens enviadas pelo usuário.

- **Entrada:** Arquivo de imagem.
- **Saída:** Descrição da imagem e grau de similaridade com instruções pré-definidas.

### 3. **Geração de Texto com Ollama**
Gera respostas e textos com base em um prompt fornecido pelo usuário, utilizando IA avançada.

- **Entrada:** Texto do prompt.
- **Saída:** Texto gerado com base no prompt fornecido.

### 4. **Processamento de Vídeo**
Permite o envio de vídeos para análise de frames com base em instruções específicas. O modelo `CLIP` processa o vídeo e calcula a similaridade entre o conteúdo do vídeo e a instrução dada.

- **Entrada:** Arquivo de vídeo e uma instrução de detecção.
- **Saída:** Similaridade média entre o vídeo e a descrição fornecida.

### 5. **Especialista em Farmácia**
A IA foi treinada para responder como um farmacêutico profissional, capaz de fornecer informações sobre medicamentos, interações medicamentosas, efeitos colaterais, tratamentos crônicos, vacinas, e muito mais.

- **Entrada:** Pergunta relacionada a farmácia (medicamentos, tratamentos, interações, etc.)
- **Saída:** Resposta detalhada e profissional em português, baseada em conhecimento especializado.

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.12+
- Pipenv ou Virtualenv para gerenciamento de dependências
- Requisitos do sistema para MoviePy e FFmpeg para processamento de vídeo
