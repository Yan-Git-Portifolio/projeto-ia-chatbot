
# Projeto Chatbot com Neutralização de Gênero

Este é um projeto de chatbot baseado em BERTimbau, modificado para lidar com variações de gênero nas entradas dos usuários. O objetivo é tornar o chatbot mais robusto e capaz de entender diferentes formas de expressão, garantindo uma resposta mais precisa e inclusiva.

## Alterações Realizadas

### 1. **Neutralização de Gênero**:
   - **Motivação**: O chatbot tinha dificuldades em reconhecer variações de gênero em palavras, como "obrigado" e "obrigada".
   - **Solução**: Foi criada a função `neutralize_gender`, que remove a terminação de gênero das palavras (por exemplo, removendo o "a" ou "o" final), garantindo que o chatbot trate termos como "obrigado" e "obrigada" de forma neutra. Isso melhora a precisão das respostas, sem a necessidade de diferenciar gêneros.

### 2. **Alterações na Função `clear_writing`**:
   - A função `clear_writing` foi alterada para incorporar a solução de neutralização de gênero. Agora, as palavras são neutralizadas antes da análise. Além disso, palavras como artigos definidos (por exemplo, "a", "o", "as", "os") são removidas do texto de entrada.

### 3. **Integração com o Modelo BERTimbau**:
   - O modelo BERTimbau foi mantido no código para tratar variações de gênero e contexto, usando o tokenizador e a arquitetura do modelo para garantir que as palavras sejam analisadas corretamente dentro do contexto.

### 4. **Acentuação**:
   - **Nota**: A normalização de acentuação não foi aplicada no projeto, pois, no meu sistema operacional (Linux), não houve problemas com a acentuação das palavras. Isso garante que palavras com acentos, como "Vênus", sejam corretamente reconhecidas sem a necessidade de remoção dos acentos.

## Tema do Chatbot

O tema do nosso chatbot é **astronomia**, escolhido porque achamos o assunto fascinante e acreditamos que ele oferece um vasto leque de tópicos interessantes para interações. O chatbot foi projetado para responder a perguntas sobre planetas, estrelas, constelações, fenômenos astronômicos e outros tópicos relacionados à astronomia de maneira acessível e inclusiva.

## Integrantes do Projeto

Este projeto foi desenvolvido por:

- **Yan Siqueira**
- **Igor Daltio**
- **João Dias**
