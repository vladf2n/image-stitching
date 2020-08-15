# image-stitching

Algoritmo para realizar um image-stitching com OpenCV em Python

Para o funcionamento dos código apenas o OpenCV deve ser instalado utilizando o seguinte comando: ``pip3 install opencv-python``

#### Instruções para execução

1. Para executar o Image Stitching: ``python3 image_stitching.py``

#### Alguns Pontos Sobre o Código
 
 O algoritmo de Image Stitching utiliza o descritor de imagem ORB para idenficar descritores similares entre as imagens e conecta-las. Para conectar os pontos foi utilizado RANSAC para gerar a matrix homográfica.

#### Saída

 Resultado do Image Stitching:
 
 ![picture](titchedImage.png)