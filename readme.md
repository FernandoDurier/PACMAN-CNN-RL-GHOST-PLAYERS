#Modo Todo Manual do Jogo 
### Para tornar os ghosts manuais basta incluir em qualquer outro comando os seguintes args 
####-g ghostManual -G KeyboardAgent2,KeyboardAgent3,KeyboardAgent4,KeyboardAgent5
python pacman.py -l mediumClassic4Ghosts -g ghostManual -G KeyboardAgent2,KeyboardAgent3,KeyboardAgent4,KeyboardAgent5

# Q-learning
# Aprendendo mapa pequeno

# Aprendendo mapa medio
python pacman.py -p PacmanQAgent -n 3 -l mediumGrid -a numTraining=3

# Aprendendo mapa classico
python pacman.py -p PacmanQAgent -n 3 -l mediumClassic -a numTraining=3

# Apos aprendizado mapa pequeno
python pacman.py -p PacmanQAgent -x 2000 -n 2005 -l smallGrid

# Apos aprendizado mapa medio
python pacman.py -p PacmanQAgent -x 2000 -n 2003 -l mediumGrid

# Apos aprendizado mapa classico
python pacman.py -p PacmanQAgent -x 2000 -n 2003 -l mediumClassic


# Aproximate Q-Learning
# Aprendendo mapa pequeno
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n 5 -l smallGrid -a numTraining=10

# Aprendendo mapa medio
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n 3 -l mediumGrid -a numTraining=3 

# Aprendendo mapa classico
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -n 3 -l mediumClassic -a numTraining=3 


# Apos aprendizado mapa pequeno
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 55 -l smallGrid 

# Apos aprendizado mapa medio
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l mediumGrid 

# Apos aprendizado mapa classico
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l mediumClassic 

#Com 4 fantasmas
#Treinamento
python pacman.py -p PacmanQAgent -n 3 -l originalClassic -a numTraining=3

#Mapa Original Clássico com Aproximação
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l originalClassic