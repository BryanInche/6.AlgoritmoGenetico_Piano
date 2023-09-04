#!/usr/bin/env python
# coding: utf-8

# El powerpoint esta en este link: https://www.canva.com/design/DAFm-qwHE9k/rcQNjbHOU1J1pPc2NxugXg/edit?utm_content=DAFm-qwHE9k&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# ####Librerias a instalar

# In[1]:


get_ipython().system('sudo apt install -y fluidsynth')


# In[2]:


get_ipython().system('pip install --upgrade pyfluidsynth')


# In[3]:


get_ipython().system('pip install pretty_midi')


# ####Pasamos de el archivo musical real a un MIDI

# In[27]:


import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import random

from IPython import display
from matplotlib import pyplot as plt
from typing import Optional


# In[28]:


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate para audio playback (ese numero es la frecuencia para que suene bien la cancion)
_SAMPLING_RATE = 44100


# Aquí vamos a hacerlo para ambas canciones:
# 
# 1- Wii = Canción más larga
# 
# 
# 2- Passionfruit = El riff que vimos en la presentación

# In[31]:


sample_file_wii = '/content/Wii Channels - Mii Channel.mid'
sample_file_passionfruit = '/content/passionfruit_112bpm_g-major.mid.midi'
print(sample_file_wii)
print(sample_file_passionfruit)


# In[43]:


pm_wii = pretty_midi.PrettyMIDI(sample_file_wii)
pm_passionfruit = pretty_midi.PrettyMIDI(sample_file_passionfruit)
songs = ["wii", "passionfruit"]


# In[44]:


# Tocar audios de archivos MIDI en Python
def display_audio(pm: pretty_midi.PrettyMIDI, seconds=300):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)


# In[34]:


display_audio(pm_wii)


# In[35]:


display_audio(pm_passionfruit)


# In[45]:


# Cantidad de instrumentos en el MIDI

#Wii
print('Number of instruments in Wii:', len(pm_wii.instruments))
instrument_wii = pm_wii.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument_wii.program)
print('Instrument name:', instrument_name)

#Passionfruit
print('Number of instruments in Passionfruit:', len(pm_passionfruit.instruments))
instrument_passionfruit = pm_passionfruit.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument_passionfruit.program)
print('Instrument name:', instrument_name)


# In[53]:


# Lectura del archivo MIDI
counter=0
for song in [instrument_wii, instrument_passionfruit]:
  print("Canción:", songs[counter])
  counter+=1
  for i, note in enumerate(song.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name},'
          f' duration={duration:.4f}')


# In[54]:


# Pasar archivos MIDI a notas (en formato MIDI)
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    prev_start = start

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


# In[56]:


raw_notes_wii = midi_to_notes(sample_file_wii)
raw_notes_passionfruit = midi_to_notes(sample_file_passionfruit)
print(list(raw_notes_wii['pitch']))
print(list(raw_notes_passionfruit['pitch']))


# In[57]:


# Grafico de notas musicales en un diagrama de piano
def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
  if count:
    title = f'First {count} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
  plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
  plt.plot(
      plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)


# In[58]:


plot_piano_roll(raw_notes_wii, count=100)


# In[59]:


plot_piano_roll(raw_notes_passionfruit, count=100)


# In[60]:


# Transformar notas guardadas en un DataFrame a un archivo MIDI (para poder escuchar las canciones)
def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str,
  instrument_name: str,
  velocity: int = 100,
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm


# In[61]:


example_file = 'example.midi'
example_pm = notes_to_midi(
    raw_notes_wii, out_file=example_file, instrument_name=instrument_name)


# In[62]:


display_audio(example_pm)


# #### Usamos nuestro algoritmo genético

# ##### Cancion de Wii

# In[63]:


POPULATION_SIZE = 400
MutationRate = 0.03
possibleNotes = list(range(128))
target = list(raw_notes_wii['pitch'])
notes = len(target)

def generateGenome(notes):
    # Función para crear genomas
    return [random.choice(possibleNotes) for _ in range(notes)]

def generatePopulation(n, notes):
    # Función para crear poblaciones
    population = [generateGenome(notes) for _ in range(n)]
    return population

def fitnessFunction(genome):
    #Calculo la suma de distancias entre cada nota del genoma y el target
    diff_sum = sum(abs(genome[i] - target[i]) for i in range(notes))
    # La fitness será 1 - la diferencia de sumas/diferencia de sumas posibles
    fitness = 1 - (diff_sum / (len(target) * (max(possibleNotes) - min(possibleNotes))))
    return fitness

def selectParents(population, fitnessValues):
    #Selección de padres usando Accept-Reject
    parentA, parentB = None, None

    while parentA is None or parentB is None:
        indexA = random.randint(0, len(population) - 1)
        indexB = random.randint(0, len(population) - 1)

        if random.uniform(0, 1) < fitnessValues[indexA]:
            parentA = population[indexA]
        if random.uniform(0, 1) < fitnessValues[indexB]:
            parentB = population[indexB]

    return parentA, parentB

def crossoverFunction(parentA, parentB):
    # Usamos el single-point crossover

    # Elegimos el single point entre 1 y la longitud del genoma
    singlePoint = random.randint(1, len(parentA) - 1)

    # Hacemos el crossover
    childA = parentA[:singlePoint] + parentB[singlePoint:]
    childB = parentB[:singlePoint] + parentA[singlePoint:]

    return childA, childB

def mutateGenome(genome, mutationRate, generation):
    # Hacemos la mutación, teniendo en cuenta, la mutationRate y por qué generación vamos
    # Si la generación es menor o igual a 300 --> Todo normal
    for i in range(len(genome)):
        if random.uniform(0, 1) <= mutationRate:
            genome[i] = random.choice(possibleNotes)

    return genome

def runEvolution(mutationRate, POPULATION_SIZE):
    # Corremos el algoritmo genético hasta llegar al máx fitness = 1

    population = generatePopulation(POPULATION_SIZE, notes)

    fitness = 0
    nextGeneration = []
    generation = 0

    #Máximo fitness = 1
    while fitness < 1:
        population = sorted(population, key=lambda genome: fitnessFunction(genome), reverse=True)
        # Debido al largo de la canción, para acelerar el proceso al principio, decidimos que la
        # nueva generación tome la mejor mitad de la generación anterior
        nextGeneration = population[:int(len(population) / 2)]

        while len(nextGeneration) < POPULATION_SIZE:
            parentA, parentB = selectParents(population, [fitnessFunction(genome) for genome in population])
            childA, childB = crossoverFunction(parentA, parentB)

            childA = mutateGenome(childA, mutationRate, generation)
            childB = mutateGenome(childB, mutationRate, generation)

            nextGeneration += [childA, childB]

        population = nextGeneration
        fitness = max(fitnessFunction(genome) for genome in population)
        bestGenome = population[0]

        print("Las notas de la mejor canción de la generación", generation, "son:", bestGenome, "con una fitness de:", fitness)
        generation += 1

    population = sorted(population, key=lambda genome: fitnessFunction(genome), reverse=True)

    return print(population[0])


# In[ ]:


#Corremos el algoritmo
runEvolution(MutationRate, POPULATION_SIZE)


# ##### Riff de passionfruit

# In[68]:


import random

POPULATION_SIZE = 800
MutationRate = 0.03
possibleNotes = list(range(128))
target = list(raw_notes_passionfruit['pitch'])
notes = len(target)

def generateGenome(notes):
    #Función para crear genomas
    return [random.choice(possibleNotes) for _ in range(notes)]

def generatePopulation(n, notes):
    #Función para crear poblaciones
    population = [generateGenome(notes) for _ in range(n)]
    return population

def fitnessFunction(genome):
    score = sum(1 for i in range(notes) if genome[i] == target[i])
    fitness = score / len(target)
    return fitness

def selectParents(population, fitnessValues):
    #Usamos el accept-reject method para seleccionar los padres
    parentA, parentB = None, None

    while parentA is None or parentB is None:
        #Elijo un fenoma random de la población
        indexA = random.randint(0, len(population) - 1)
        indexB = random.randint(0, len(population) - 1)

        #Comparo su fitness con mi número random entre 0 y 1
        # y aplico el accept-reject method
        if random.uniform(0, 1) < fitnessValues[indexA]:
            parentA = population[indexA]
        if random.uniform(0, 1) < fitnessValues[indexB]:
            parentB = population[indexB]

    return parentA, parentB

def crossoverFunction(parentA, parentB):
    # Usamos el single-point crossover

    # Elegimos un número random entre 1 y la longitud del genoma
    singlePoint = random.randint(1, len(parentA) - 1)
    # Hacemos el crossover
    childA = parentA[:singlePoint] + parentB[singlePoint:]
    childB = parentB[:singlePoint] + parentA[singlePoint:]

    return childA, childB

def mutateGenome(genome, mutationRate):
    # Hacemos la mutación, teniendo en cuenta, la mutationRate
    for i in range(len(genome)):
        if random.uniform(0, 1) <= mutationRate:
            genome[i] = random.choice(possibleNotes)
    return genome

def runEvolution(mutationRate, POPULATION_SIZE):
    # Corremos el algoritmo genético hasta llegar al máx fitness

    population = generatePopulation(POPULATION_SIZE, notes)

    fitness = 0
    nextGeneration = []
    generation = 0

    while fitness < 1:
        population = sorted(population, key=lambda genome: fitnessFunction(genome), reverse=True)
        # Agregamos el 20% de la generación anterior a la nueva generación para acelerar el proceso
        nextGeneration = population[:int(len(population)*0.1)]

        while len(nextGeneration) < POPULATION_SIZE:
            parentA, parentB = selectParents(population, [fitnessFunction(genome) for genome in population])
            childA, childB = crossoverFunction(parentA, parentB)

            childA = mutateGenome(childA, mutationRate)
            childB = mutateGenome(childB, mutationRate)

            nextGeneration += [childA, childB]

        population = nextGeneration
        fitness = max(fitnessFunction(genome) for genome in population)
        bestGenome = population[0]

        print("The notes for the best song of generation", generation, "are:", bestGenome, "with a fitness of", fitness)
        generation += 1

    population = sorted(population, key=lambda genome: fitnessFunction(genome), reverse=True)

    return print(population[0])


# In[ ]:


runEvolution(MutationRate, POPULATION_SIZE)


# #### Probamos el resultado que nos dió el algoritmo genético

# ######Canción Wii

# Acá hay que tener en cuenta que lo cortamos antes de que llegue al fitness 1. Por lo que no va a sonar igual, pero sí, por las forma en que definimos la fitness function va a sonar bastante parecido. Si hubiesemos dejado el algoritmo corriendo por más tiempo (1 hora o un poco más) sí hubiesemos llegado al fitness 1

# Acá pegamos los valores del ejemplo que habíamos mostrado en clase de lo de Wii. Pero si quieren correrlo por su cuenta, habría que cambiar el return de runEvolution por population[0]. Y escribir:
# gen_7708_list = runEvolution(MutationRate)

# In[71]:


# Este es el mejor genoma de la última generación que dejamos pasar antes de cortarlo
# Generacion numero 7708. fitness: 0.9495506535947712
gen_7708_list = [63, 67, 47, 66, 69, 68, 72, 61, 67, 59, 62, 59, 60, 70, 54, 64, 56, 55, 52, 57, 56, 46, 59, 65, 63, 68, 62, 75, 67, 69, 59, 61, 66, 66, 73, 56, 64, 65, 53, 80, 62, 70, 73, 54, 63, 46, 61, 72, 62, 69, 75, 51, 68, 63, 70, 57, 68, 65, 61, 58, 56, 61, 61, 51, 62, 41, 56, 57, 40, 64, 54, 31, 56, 55, 40, 56, 63, 38, 59, 34, 56, 60, 61, 65, 51, 61, 65, 62, 70, 50, 58, 60, 64, 75, 70, 74, 64, 71, 66, 58, 62, 60, 58, 56, 68, 59, 59, 57, 70, 57, 67, 75, 72, 73, 80, 71, 44, 60, 69, 66, 79, 68, 75, 66, 67, 62, 66, 63, 70, 74, 59, 67, 63, 81, 44, 76, 56, 70, 59, 63, 60, 38, 56, 66, 71, 69, 62, 70, 54, 65, 53, 56, 66, 47, 55, 58, 60, 62, 60, 51, 56, 50, 52, 52, 61, 50, 51, 60, 46, 50, 54, 70, 54, 71, 45, 66, 59, 84, 54, 78, 64, 76, 64, 56, 60, 50, 60, 51, 52, 57, 46, 59, 49, 52, 46, 55, 51, 60, 61, 49, 60, 64, 59, 58, 59, 63, 49, 59, 59, 59, 62, 51, 54, 56, 59, 49, 47, 53, 57, 45, 57, 50, 59, 57, 42, 64, 49, 59, 68, 48, 56, 48, 57, 46, 41, 45, 58, 60, 61, 35, 60, 64, 47, 53, 63, 39, 59, 53, 56, 50, 51, 60, 54, 54, 62, 61, 36, 66, 72, 67, 67, 71, 49, 57, 71, 70, 54, 65, 65, 75, 52, 55, 61, 66, 55, 68, 63, 68, 73, 66, 74, 63, 64, 56, 57, 50, 59, 62, 49, 55, 54, 58, 53, 43, 63, 63, 59, 70, 59, 70, 68, 67, 64, 65, 58, 75, 72, 56, 61, 64, 51, 78, 64, 68, 79, 52, 70, 44, 63, 80, 60, 61, 82, 48, 64, 55, 76, 67, 63, 67, 53, 60, 57, 66, 64, 54, 60, 37, 60, 66, 39, 59, 49, 37, 57, 59, 38, 58, 62, 38, 60, 37, 65, 57, 60, 64, 57, 60, 67, 53, 64, 54, 51, 63, 66, 73, 69, 71, 65, 63, 65, 69, 66, 67, 53, 62, 65, 60, 60, 66, 64, 55, 71, 70, 74, 79, 79, 67, 49, 61, 60, 63, 70, 74, 77, 65, 70, 62, 65, 64, 72, 68, 58, 56, 66, 74, 47, 71, 46, 73, 61, 61, 60, 41, 60, 68, 69, 69, 62, 67, 52, 67, 56, 57, 67, 47, 56, 62, 75, 71, 56, 52, 60, 41, 53, 54, 60, 49, 52, 59, 55, 41, 52, 69, 42, 70, 49, 71, 58, 78, 54, 76, 59, 81, 70, 48, 56, 52, 56, 58, 56, 58, 47, 57, 51, 59, 47, 45, 59, 57, 55, 44, 57, 65, 48, 41, 62, 62, 51, 55, 57, 52, 58, 48, 58, 52, 56, 50, 50, 62, 59, 46, 58, 56, 58, 60, 39, 59, 48, 59, 62, 48, 65, 50, 58, 37, 42, 46, 57, 61, 63, 40, 56, 68, 47, 59, 61, 37, 59, 56, 54, 65, 43, 55, 57, 52, 60, 64, 41, 70, 68, 67, 65, 65, 52, 65, 69, 66, 51, 55, 64, 68, 55, 61]
print("El largo de la canción es de :", len(gen_7708_list), "notas")


# In[72]:


gen_7708_df = raw_notes_wii.copy()


# In[73]:


gen_7708_df['pitch'] = gen_7708_list


# In[74]:


example_file_7708 = 'example7708.midi'
example_7708 = notes_to_midi(gen_7708_df, out_file=example_file_7708, instrument_name=instrument_name)


# In[75]:


print(plot_piano_roll(gen_7708_df, count=100))
display_audio(example_7708)

# Nota sobre la cancion: a pesar de pegarle a gran parte de la canción,
# pero al errarle a una nota de un acorde (conjunto de 3 notas)
# o armonizacion (conjunto de 2 notas) auditivamente puede sonar feo. Pero, visualmente se puede ver como le pega a casi todas las notas


# In[ ]:


# Funcion para exportar los MIDI como archivo .mid
from google.colab import files
#files.download(example_file_7708)


# ###### Riff Passionfruit

# In[79]:


# Listas con secuencia de notas generadas por el AG
gen_0_list = [100, 109, 98, 70, 56, 119, 109, 118, 15, 2, 18, 6, 15, 75, 36, 0, 52, 82, 68, 30, 44, 102, 43, 113, 84, 36, 48, 79, 102, 28, 92, 58, 126, 51, 66, 71, 117, 96, 58, 75, 22, 64, 110, 64, 32, 20, 26, 74]
gen_10_list = [67, 81, 61, 122, 32, 61, 16, 69, 41, 52, 108, 40, 14, 11, 30, 49, 107, 61, 95, 47, 120, 62, 56, 51, 67, 114, 59, 103, 34, 46, 121, 24, 2, 58, 91, 41, 119, 54, 75, 3, 21, 119, 86, 94, 62, 52, 59, 41]
gen_50_list = [52, 59, 61, 124, 59, 88, 52, 17, 52, 52, 108, 61, 49, 11, 61, 49, 107, 61, 49, 68, 86, 49, 62, 58, 55, 119, 59, 103, 34, 59, 77, 17, 59, 51, 28, 34, 44, 54, 56, 103, 54, 121, 52, 108, 61, 52, 59, 41]
gen_100_list = [52, 59, 61, 52, 59, 61, 52, 17, 120, 52, 59, 61, 49, 11, 61, 49, 59, 61, 49, 59, 60, 49, 78, 58, 96, 58, 59, 51, 58, 59, 26, 58, 59, 51, 58, 41, 44, 54, 56, 44, 54, 121, 52, 59, 61, 52, 59, 41]
gen_200_list = [52, 59, 61, 52, 59, 61, 52, 59, 61, 52, 59, 61, 49, 59, 61, 49, 59, 61, 49, 59, 61, 49, 59, 58, 51, 58, 59, 51, 58, 59, 51, 58, 59, 51, 58, 36, 44, 54, 56, 44, 54, 108, 52, 59, 61, 52, 59, 61]
gen_353_list = [52, 59, 61, 52, 59, 61, 52, 59, 61, 52, 59, 61, 49, 59, 61, 49, 59, 61, 49, 59, 61, 49, 59, 58, 51, 58, 59, 51, 58, 59, 51, 58, 59, 51, 58, 59, 44, 54, 56, 44, 54, 49, 52, 59, 61, 52, 59, 61]


# In[80]:


gen_0_df = raw_notes_passionfruit.copy()
gen_10_df = raw_notes_passionfruit.copy()
gen_50_df = raw_notes_passionfruit.copy()
gen_100_df = raw_notes_passionfruit.copy()
gen_200_df = raw_notes_passionfruit.copy()
gen_353_df = raw_notes_passionfruit.copy()


# In[81]:


gen_0_df['pitch'] = gen_0_list
gen_10_df['pitch'] = gen_10_list
gen_50_df['pitch'] = gen_50_list
gen_100_df['pitch'] = gen_100_list
gen_200_df['pitch'] = gen_200_list
gen_353_df['pitch'] = gen_353_list


# In[96]:


example_file_0 = 'example0.midi'
example_file_10 = 'example10.midi'
example_file_50 = 'example50.midi'
example_file_100 = 'example100.midi'
example_file_200 = 'example200.midi'
example_file_353 = 'example356.midi'

example_0 = notes_to_midi(gen_0_df, out_file=example_file_0, instrument_name=instrument_name)
example_10 = notes_to_midi(gen_10_df, out_file=example_file_10, instrument_name=instrument_name)
example_50 = notes_to_midi(gen_50_df, out_file=example_file_50, instrument_name=instrument_name)
example_100 = notes_to_midi(gen_100_df, out_file=example_file_100, instrument_name=instrument_name)
example_200 = notes_to_midi(gen_200_df, out_file=example_file_200, instrument_name=instrument_name)
example_353 = notes_to_midi(gen_353_df, out_file=example_file_353, instrument_name=instrument_name)


# In[90]:


# Gen 0
print(plot_piano_roll(gen_0_df, count=100))
display_audio(example_0)


# In[91]:


# Gen 10
print(plot_piano_roll(gen_10_df, count=100))
display_audio(example_10)


# In[92]:


# Gen 50
print(plot_piano_roll(gen_50_df, count=100))
display_audio(example_50)


# In[93]:


# Gen 100
print(plot_piano_roll(gen_100_df, count=100))
display_audio(example_100)


# In[94]:


# Gen 200
print(plot_piano_roll(gen_200_df, count=100))
display_audio(example_200)


# In[95]:


# Gen 353 - Final one
print(plot_piano_roll(gen_353_df, count=100))
display_audio(example_353)


# In[ ]:


# Funcion para exportar los MIDI como archivo .mid
from google.colab import files
files.download(example_file_0)
files.download(example_file_10)
files.download(example_file_50)
files.download(example_file_100)
files.download(example_file_200)
files.download(example_file_353)

