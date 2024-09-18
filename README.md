# UPO Machine Learning & Deep Learning
Il progetto è stato fatto usando [Python 3.10.12](https://www.python.org/downloads/release/python-31012/)\
Per poter eseguire i vari componenti è prima necessario installare i package elencati nel file [requirements](https://github.com/Berack96/upo-ml/blob/main/requirements) tramite il seguente comando:
```cmd
pip install -r requirements
```

## Machine Learning
Tutti gli algoritmi implementati si possono utilizzare nel main del programma che si trova nel file [app.py](https://github.com/Berack96/upo-ml/blob/main/src/app.py#L110)\
Modificando la funzione alla riga [123](https://github.com/Berack96/upo-ml/blob/main/src/app.py#L123) si può scegliere il dataset con l'algoritmo implementato e far partire il processo di addestramento; alla fine verrà mostrato un riepilogo dei risultati\
I tre algoritmi implementati sono i seguenti:
- [Linear Regression](https://github.com/Berack96/upo-ml/blob/main/src/learning/supervised.py#L43)
- [MultiLayerPerceptron](https://github.com/Berack96/upo-ml/blob/main/src/learning/supervised.py#L57)
- [K-Means](https://github.com/Berack96/upo-ml/blob/main/src/learning/unsupervised.py#L7)

## Deep Learning
Entrambi gli esercizi sono stati svolti tramite file Jupyter Notebook in modo da poter eseguire solo pezzi di codice.\
In entrambi i casi si possono vedere i risultati dell'ultima elaborazione in modo da evitare di riaddestrare il modello.\
I due esercizi sono:
- [Image Captioning](https://github.com/Berack96/upo-ml/blob/main/src/deep/caption.ipynb): L'esercizio prevede la costruzione di un  generatore di caption di immagini. Utilizzando il tool preferito, costruire una CNN in grado di costruire un suitable embedding di un immagine, il quale verrà passato ad una RNN per la generazione del testo della caption. Per il training utilizzare il Flickr Dataset. (da scaricare manualmente)
- [Image Denoising](https://github.com/Berack96/upo-ml/blob/main/src/deep/denoise.ipynb): Realizzare un sistema di pulizia immagini (image denoising), basato su una opportuna architettura ad Autoencoder, che prendendo in input un dataset di immagini soggette a "rumore" ne restituisca una versione "pulita". Utilizzare il dataset Fashion-MNIST (inserire un opportuno filtro di noise per ottenere l'input)
