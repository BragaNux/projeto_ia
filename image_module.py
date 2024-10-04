import os
import torch
import clip
from PIL import Image, UnidentifiedImageError

def describe_image_with_clip(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    try:
        # Carregar a imagem e pré-processá-la
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except UnidentifiedImageError:
        print(f"Erro: Não foi possível identificar o arquivo de imagem '{image_path}'.")
        return None

    # Lista de descrições automáticas para comparar com a imagem
    possible_descriptions = [
    "uma paisagem de montanha",
    "um gato",
    "um cachorro",
    "uma árvore",
    "um carro",
    "um prédio",
    "uma pessoa sorrindo",
    "uma praia",
    "uma cidade",
    "uma rua",
    "jair bolsonaro",
    "prontuário médico",
    "uma bicicleta",
    "um lago",
    "um campo de flores",
    "um pôr do sol",
    "uma ponte",
    "um avião no céu",
    "um rio",
    "um estádio de futebol",
    "uma estação de trem",
    "uma criança brincando",
    "um casal abraçado",
    "um parque",
    "uma floresta",
    "um deserto",
    "um barco navegando",
    "um trem",
    "um ônibus",
    "uma motocicleta",
    "uma estrada",
    "um mercado",
    "um shopping center",
    "um hospital",
    "um mapa",
    "um robô",
    "um computador",
    "um celular",
    "uma cadeira",
    "uma mesa",
    "um relógio",
    "um livro",
    "um jornal",
    "uma televisão",
    "um quadro",
    "um estádio lotado",
    "uma manifestação política",
    "um acidente de carro",
    "um festival de música",
    "uma fogueira",
    "um evento esportivo",
    "uma conferência",
    "uma apresentação no palco",
    "um restaurante",
    "um zoológico",
    "um museu",
    "uma biblioteca",
    "um avião pousando",
    "um trator",
    "uma fazenda",
    "uma pessoa trabalhando no escritório",
    "um laboratório de ciência",
    "uma cirurgia",
    "um policial",
    "um bombeiro",
    "um astronauta",
    "um estádio vazio",
    "um estacionamento",
    "uma academia de ginástica",
    "um médico examinando um paciente",
    "um cemitério",
    "um casamento",
    "uma festa de aniversário",
    "uma loja de roupas",
    "um mercado de frutas",
    "uma praça pública",
    "uma estátua famosa",
    "um monumento histórico",
    "uma sala de aula",
    "um professor ensinando",
    "um ônibus escolar",
    "uma sala de tribunal",
    "um juiz",
    "um advogado",
    "um réu",
    "um réptil",
    "um pássaro voando",
    "um arco-íris",
    "uma nevasca",
    "uma tempestade",
    "uma planta de energia",
    "um porto com navios",
    "um parque de diversões",
    "um circo",
    "uma escultura moderna",
    "um helicóptero",
    "uma pessoa jogando video game",
    "uma pessoa azul jogando darksouls"
]


    # Tokenizar as descrições
    text = clip.tokenize(possible_descriptions).to(device)

    with torch.no_grad():
        # Comparar a imagem com as descrições
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Encontrar a descrição mais provável
    most_similar_index = probs.argmax()
    most_similar_description = possible_descriptions[most_similar_index]
    similarity_score = probs[0][most_similar_index]

    return most_similar_description, similarity_score

def process_image():
    image_path = input("Digite o caminho completo para a imagem: ")

    if not os.path.isfile(image_path):
        print("Erro: O caminho fornecido não é um arquivo válido.")
        return

    # Chama a função para descrever a imagem automaticamente
    description, similarity = describe_image_with_clip(image_path)

    if description is not None:
        print(f"A descrição mais provável da imagem é: '{description}' com uma similaridade de {similarity:.2f}")

# Testar o código
if __name__ == "__main__":
    process_image()
