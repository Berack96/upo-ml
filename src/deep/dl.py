# ***********************************************************
# Esercizio 1.A
# L'esercizio prevede la costruzione di un  generatore di caption di immagini.
# Utilizzando il tool preferito, costruire una CNN in grado di costruire
# un suitable embedding di un immagine, il quale verrà passato ad una RNN
# per la generazione del testo della caption.
# Per il training utilizzare il Flickr Dataset.
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

# # Esercizio 1.B
# L'esercizio consiste nell'applicare una delle tecniche viste a lezione
# per l'object detection tramite bounding box per costruire un riconoscitore
# di oggetti all'interno di una immagine presa con una fotocamera.
# Il dataset da utilizzare è Object365, un dataset di HQ (high-quality)
# immagini con bounding boxes di oggetti.
# Contiene 365 oggetti, 600k immagini, e 10 milioni di bounding boxes.
# Se necessario eseguire un subsampling per ridurre le dimensioni del dataset.
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model.train(data='Objects365.yaml', epochs=100, imgsz=640)


# ***********************************************************
# Esercizio 2.A
# Realizzare un sistema di pulizia immagini (image denoising),
# basato su una opportuna architettura ad Autoencoder,
# che prendendo in input un dataset di immagini soggette a "rumore"
# ne restituisca una versione "pulita".
# Utilizzare il dataset Fashion-MNIST (inserire un opportuno filtro di noise per ottenere l'input)
# https://www.kaggle.com/datasets/zalando-research/fashionmnist

# Esercizio 2.B
# Realizzare un sistema di text classification per frasi (sequenze testuali)
# che possono essere considerate "positive" o "negative"(sentiment analysis).
# Il sistema deve basarsi sull'uso di architetture transformer.
# Suggerimento: vedere modelli offerti da HuggingFace.
# Alcuni dataset suggeriti su kaggle: https://www.kaggle.com/datasets?search=sentiment+analysis
