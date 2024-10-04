from moviepy.editor import VideoFileClip
import os
import torch
import clip
from PIL import Image

def describe_frame(frame, model, preprocess, device):
    """Processa um frame do vídeo e descreve o que aparece usando o CLIP."""
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    text_descriptions = [
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
    text_tokens = clip.tokenize(text_descriptions).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    most_probable_idx = probs.argmax()
    description = text_descriptions[most_probable_idx]
    return description, probs[0][most_probable_idx]

def process_video_file(video_path):
    """Processa o vídeo e descreve o que aparece."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Carregar o vídeo
    clip_video = VideoFileClip(video_path)
    frame_count = int(clip_video.duration * clip_video.fps)
    descriptions = []
    
    print("Analisando o vídeo frame a frame...")

    # Processar cada frame do vídeo
    for i, frame in enumerate(clip_video.iter_frames()):
        description, similarity = describe_frame(frame, model, preprocess, device)
        descriptions.append(description)

        # Logging opcional para acompanhar o progresso
        print(f"Frame {i+1}/{frame_count}: {description} (similaridade: {similarity:.2f})")
    
    # Obter a descrição mais frequente do vídeo
    most_common_description = max(set(descriptions), key=descriptions.count)
    return most_common_description
