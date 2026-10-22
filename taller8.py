import matplotlib.pyplot as plt
import numpy as np
import procesamiento as proce

imagen_messi = np.array(plt.imread("messi.jpg")) / 255

def punto1():

    matriz_colores = np.zeros([3, 3, 3], dtype=np.uint8)
    matriz_colores[0, 0] = [0, 255, 255]
    matriz_colores[0, 1] = [255, 255, 255]
    matriz_colores[0, 2] = [255, 0, 0]
    matriz_colores[1, 0] = [255, 0, 255]
    matriz_colores[1, 1] = [128, 128, 128]
    matriz_colores[1, 2] = [0, 255, 0]
    matriz_colores[2, 0] = [255, 255, 0]
    matriz_colores[2, 1] = [0, 0, 0]
    matriz_colores[2, 2] = [0, 0, 255]
    plt.imshow(matriz_colores, interpolation="nearest")
    plt.title("Punto 1: Matriz 3x3")
    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.show()

def punto2():

    matriz_barras_tv = np.ones([7, 11, 3])
    matriz_barras_tv[0:5, 0, 2] = 0
    matriz_barras_tv[0:5, 1:3, 0] = 0
    matriz_barras_tv[0:5, 3:5, 0] = matriz_barras_tv[0:5, 3:5, 2] = 0
    matriz_barras_tv[0:5, 5:7, 1] = 0
    matriz_barras_tv[0:5, 7:9, 1] = matriz_barras_tv[0:5, 7:9, 2] = 0
    matriz_barras_tv[0:5, 9:11, 0] = matriz_barras_tv[0:5, 9:11, 1] = 0
    matriz_barras_tv[5:7, 0] = 0.6
    matriz_barras_tv[5:7, 1] = 0.54
    matriz_barras_tv[5:7, 2] = 0.46
    matriz_barras_tv[5:7, 3] = 0.38
    matriz_barras_tv[5:7, 4] = 0.3
    matriz_barras_tv[5:7, 5] = 0.18
    matriz_barras_tv[5:7, 6] = 0.11
    matriz_barras_tv[5:7, 7:11] = 0
    plt.imshow(matriz_barras_tv)
    plt.title("Punto 2")
    plt.axis("off")
    plt.show()

def punto3():

    imagen = np.copy(imagen_messi)
    imagen_invertida = proce.invertir_colores(imagen)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_invertida)
    plt.title("Invertida")
    plt.axis("off")
    plt.show()

def punto4():

    imagen = np.copy(imagen_messi)
    capa_roja = proce.extraer_capa_roja(imagen)
    capa_verde = proce.extraer_capa_verde(imagen)
    capa_azul = proce.extraer_capa_azul(imagen)
    
    plt.subplot(2, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(capa_roja)
    plt.title("Punto 4: Capa Roja")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(capa_verde)
    plt.title("Punto 5: Capa Verde")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(capa_azul)
    plt.title("Punto 6: Capa Azul")
    plt.axis("off")
    plt.show()

def punto7():

    imagen = np.copy(imagen_messi)
    imagen_magenta = proce.filtrar_magenta(imagen)
    imagen_cyan = proce.filtrar_cyan(imagen)
    imagen_amarillo = proce.filtrar_amarillo(imagen)
    
    plt.subplot(2, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(imagen_magenta)
    plt.title("Punto 7: Magenta")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(imagen_cyan)
    plt.title("Punto 8: Cyan")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(imagen_amarillo)
    plt.title("Punto 9: Amarillo")
    plt.axis("off")
    plt.show()

def punto10():

    imagen = np.copy(imagen_messi)
    capa_roja = proce.extraer_capa_roja(imagen)
    capa_verde = proce.extraer_capa_verde(imagen)
    capa_azul = proce.extraer_capa_azul(imagen)
    imagen_reconstruida = proce.reconstruir_rgb(capa_roja, capa_verde, capa_azul)
   
    plt.subplot(2, 2, 1)
    plt.imshow(capa_roja)
    plt.title("Capa R")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(capa_verde)
    plt.title("Capa G")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(capa_azul)
    plt.title("Capa B")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(imagen_reconstruida)
    plt.title("Reconstruida")
    plt.axis("off")
    plt.show()

def punto11():

    imagen = np.copy(imagen_messi)
    imagen_espejo = imagen[:, ::-1]
    imagen_fusion_sin_ecualizar = proce.fusion_sin_ecualizar(imagen, imagen_espejo)
    
    plt.subplot(2, 2, 1)
    plt.imshow(imagen)
    plt.title("Imagen 1")
    plt.axis("off")
   
    plt.subplot(2, 2, 2)
    plt.imshow(imagen_espejo)
    plt.title("Imagen 2")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(imagen_fusion_sin_ecualizar)
    plt.title("Punto 11: Sin Ecualizar")
    plt.axis("off")
    plt.show()

def punto12():

    imagen = np.copy(imagen_messi)
    imagen_espejo = imagen[:, ::-1]
    imagen_fusion_ecualizada = proce.fusion_ecualizada(imagen, imagen_espejo)
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagen)
    plt.title("Imagen 1")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(imagen_espejo)
    plt.title("Imagen 2")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(imagen_fusion_ecualizada)
    plt.title("Punto 12: Ecualizada")
    plt.axis("off")
    plt.show()

def punto13():

    imagen = np.copy(imagen_messi)
    imagen_ecualizada_factor = proce.ecualizar_por_factor(imagen, 0.5)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original (1.0)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_ecualizada_factor)
    plt.title("Punto 13: Factor 0.5")
    plt.axis("off")
    plt.show()

def punto14():

    imagen = np.copy(imagen_messi)
    imagen_promedio_rgb = proce.tecnica_promedio_rgb(imagen)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_promedio_rgb)
    plt.title("Punto 14: Promedio (RGB)")
    plt.axis("off")
    plt.show()

def punto15():

    imagen = np.copy(imagen_messi)
    imagen_grises_promedio = proce.escala_grises_promedio_average(imagen)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_grises_promedio, cmap="gray")
    plt.title("Punto 15: Promedio")
    plt.axis("off")
    plt.show()

def punto16():

    imagen = np.copy(imagen_messi)
    imagen_grises_luminosidad = proce.escala_grises_luminosidad(imagen)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_grises_luminosidad, cmap="gray")
    plt.title("Punto 16: Luminosidad")
    plt.axis("off")
    plt.show()

def punto17():

    imagen = np.copy(imagen_messi)
    imagen_grises_tonalidad = proce.escala_grises_tonalidad_midgray(imagen)
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen_grises_tonalidad, cmap="gray")
    plt.title("Punto 17: Tonalidad")
    plt.axis("off")
    plt.show()

def menu():
    
    while True:
        print("\n--- MENU TALLER 8 ---")
        print("1. Punto 1")
        print("2. Punto 2")
        print("3. Punto 3")
        print("4. Puntos 4, 5 y 6 (introduzca 4)")
        print("7. Puntos 7, 8 y 9 (introduzca 7)")
        print("10. Punto 10")
        print("11. Punto 11")
        print("12. Punto 12")
        print("13. Punto 13")
        print("14. Punto 14")
        print("15. Punto 15")
        print("16. Punto 16")
        print("17. Punto 17")
        print("0. Salir")
        
        opcion = input("Seleccione una opcion: ").strip()
        if opcion == "1":
            punto1()
        elif opcion == "2":
            punto2()
        elif opcion == "3":
            punto3()
        elif opcion == "4":
            punto4()
        elif opcion == "7":
            punto7()
        elif opcion == "10":
            punto10()
        elif opcion == "11":
            punto11()
        elif opcion == "12":
            punto12()
        elif opcion == "13":
            punto13()
        elif opcion == "14":
            punto14()
        elif opcion == "15":
            punto15()
        elif opcion == "16":
            punto16()
        elif opcion == "17":
            punto17()
        elif opcion == "0" or opcion == "salir":
            break
        else:
            print("Opcion no valida.")

if __name__ == "__main__":
    menu()
