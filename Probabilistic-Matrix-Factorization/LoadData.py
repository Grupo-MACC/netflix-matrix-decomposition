from numpy import *
import random


def load_rating_data(file_path='ml-100k/u.data'):
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = [] # lista vacía que irá guardando tripletas [user, item, rating]
    for line in open(file_path, 'r'):  # Abre el archivo u.data en modo lectura
        (userid, movieid, rating, ts) = line.split('\t')  # cada línea tiene 4 columnas separadas por tab
        uid = int(userid)  # id del usuario 
        mid = int(movieid) # id de la película
        rat = float(rating) # rating (1–5)
        prefer.append([uid, mid, rat]) # guarda tripleta [usuario, item, rating]. Ignora timestamp =ts
    data = array(prefer) # convierte la lista a numpy array
    return data


def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random() # número aleatorio [0,1)
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = array(train_data)
    test_data = array(test_data)
    return train_data, test_data
