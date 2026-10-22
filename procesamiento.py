import numpy as np
import matplotlib.pyplot as plt

def invertir_colores(imagen):
    return 1 - imagen

def extraer_capa_roja(imagen):
    capa_roja = np.copy(imagen)
    capa_roja[:, :, 1] = capa_roja[:, :, 2] = 0
    return capa_roja

def extraer_capa_verde(imagen):
    capa_verde = np.copy(imagen)
    capa_verde[:, :, 0] = capa_verde[:, :, 2] = 0
    return capa_verde

def extraer_capa_azul(imagen):
    capa_azul = np.copy(imagen)
    capa_azul[:, :, 0] = capa_azul[:, :, 1] = 0
    return capa_azul

def filtrar_magenta(imagen):
    canal_magenta = np.copy(imagen)
    canal_magenta[:, :, 0] = canal_magenta[:, :, 2] = 1
    return canal_magenta

def filtrar_cyan(imagen):
    canal_cian = np.copy(imagen)
    canal_cian[:, :, 1] = canal_cian[:, :, 2] = 1
    return canal_cian

def filtrar_amarillo(imagen):
    canal_amarillo = np.copy(imagen)
    canal_amarillo[:, :, 0] = canal_amarillo[:, :, 1] = 1
    return canal_amarillo

def reconstruir_rgb(capa_roja, capa_verde, capa_azul):
    return capa_roja + capa_verde + capa_azul

def fusion_sin_ecualizar(imagen_uno, imagen_dos):
    imagen_fusion = imagen_uno + imagen_dos
    imagen_fusion[imagen_fusion > 1] = 1
    return imagen_fusion

def fusion_ecualizada(imagen_uno, imagen_dos):
    return (imagen_uno + imagen_dos) / 2

def ecualizar_por_factor(imagen, factor=0.5):
    return imagen * factor

def tecnica_promedio_rgb(imagen):
    imagen_gris = np.copy(imagen)
    promedio = (imagen[:, :, 0] + imagen[:, :, 1] + imagen[:, :, 2]) / 3
    imagen_gris[:, :, 0] = imagen_gris[:, :, 1] = imagen_gris[:, :, 2] = promedio
    return imagen_gris

def escala_grises_promedio_average(imagen):
    return (imagen[:, :, 0] + imagen[:, :, 1] + imagen[:, :, 2]) / 3

def escala_grises_luminosidad(imagen):
    return 0.299 * imagen[:, :, 0] + 0.587 * imagen[:, :, 1] + 0.114 * imagen[:, :, 2]

def escala_grises_tonalidad_midgray(imagen):

    rojo = imagen[:, :, 0]
    verde = imagen[:, :, 1]
    azul = imagen[:, :, 2]
    maximo = np.maximum(np.maximum(rojo, verde), azul)
    minimo = np.minimum(np.minimum(rojo, verde), azul)
    return (maximo + minimo) / 2
