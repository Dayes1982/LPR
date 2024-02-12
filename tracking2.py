import cv2
from ultralytics import YOLO

# Cargamos video o streaming
cap = cv2.VideoCapture("videos/traffic.mp4")
# Cargamos el modelo de YoloV8
model = YOLO("yolov8n.pt")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        #results = model.track(frame, persist=True)

        """
        Inferencia
        conf	float	0.25	Establece el umbral mínimo de confianza para las detecciones. Los objetos detectados con una confianza inferior a este umbral serán descartados. Ajustar este valor puede ayudar a reducir los falsos positivos.
        iou	float	0.7	Umbral de Intersección sobre Unión (IoU) para la Supresión No Máxima (NMS). Los valores más altos dan lugar a menos detecciones al eliminar las cajas superpuestas, lo que resulta útil para reducir los duplicados.
        imgsz	int or tuple	640	Define el tamaño de la imagen para la inferencia. Puede ser un único número entero 640 para un redimensionamiento cuadrado o una tupla (alto, ancho). Un tamaño adecuado puede mejorar la precisión de la detección y la velocidad de procesamiento.
        half	bool	False	Permite la inferencia de media precisión (FP16), que puede acelerar la inferencia del modelo en las GPU compatibles con un impacto mínimo en la precisión.
        device	str	None	Especifica el dispositivo para la inferencia (por ejemplo, cpu, cuda:0 o 0). Permite a los usuarios seleccionar entre la CPU, una GPU específica u otros dispositivos de cálculo para la ejecución del modelo.
        max_det	int	300	Número máximo de detecciones permitidas por imagen. Limita el número total de objetos que el modelo puede detectar en una sola inferencia, evitando salidas excesivas en escenas densas.
        vid_stride	int	1	Salto de fotogramas para entradas de vídeo. Permite saltar fotogramas en los vídeos para acelerar el procesamiento a costa de la resolución temporal. Un valor de 1 procesa cada fotograma, valores superiores omiten fotogramas.
        stream_buffer	bool	False	Determina si todos los fotogramas deben almacenarse en la memoria intermedia al procesar secuencias de vídeo (True), o si el modelo debe devolver el fotograma más reciente (False). Útil para aplicaciones en tiempo real.
        visualize	bool	False	Activa la visualización de las características del modelo durante la inferencia, proporcionando información sobre lo que el modelo está "viendo". Útil para la depuración y la interpretación del modelo.
        augment	bool	False	Permite el aumento del tiempo de prueba (TTA) para las predicciones, mejorando potencialmente la robustez de la detección a costa de la velocidad de inferencia.
        agnostic_nms	bool	False	Activa la Supresión No Máxima (NMS) agnóstica de clases, que fusiona las cajas superpuestas de clases diferentes. Útil en escenarios de detección multiclase en los que el solapamiento de clases es habitual.
        classes	list[int]	None	Filtra las predicciones a un conjunto de ID de clase. Sólo se devolverán las detecciones que pertenezcan a las clases especificadas. Útil para centrarse en objetos relevantes en tareas de detección multiclase.
        retina_masks	bool	False	Utiliza máscaras de segmentación de alta resolución si están disponibles en el modelo. Esto puede mejorar la calidad de la máscara para las tareas de segmentación, proporcionando detalles más finos.
        embed	list[int]	None	Especifica las capas de las que extraer vectores de características o incrustaciones. Útil para tareas posteriores como la agrupación o la búsqueda de similitudes.
        Visualización
        show	bool	False	Si Truemuestra las imágenes o vídeos anotados en una ventana. Resulta útil para obtener información visual inmediata durante el desarrollo o las pruebas.
        save	bool	False	Permite guardar las imágenes o vídeos anotados en un archivo. Útil para documentación, análisis posteriores o para compartir resultados.
        save_frames	bool	False	Al procesar vídeos, guarda fotogramas individuales como imágenes. Es útil para extraer fotogramas concretos o para un análisis detallado fotograma a fotograma.
        save_txt	bool	False	Guarda los resultados de la detección en un archivo de texto, siguiendo el formato [class] [x_center] [y_center] [width] [height] [confidence]. Útil para la integración con otras herramientas de análisis.
        save_conf	bool	False	Incluye puntuaciones de confianza en los archivos de texto guardados. Aumenta el detalle disponible para el postprocesado y el análisis.
        save_crop	bool	False	Guarda imágenes recortadas de las detecciones. Útil para aumentar el conjunto de datos, analizarlos o crear conjuntos de datos centrados en objetos concretos.
        show_labels	bool	True	Muestra etiquetas para cada detección en la salida visual. Proporciona una comprensión inmediata de los objetos detectados.
        show_conf	bool	True	Muestra la puntuación de confianza de cada detección junto a la etiqueta. Da una idea de la certeza del modelo para cada detección.
        show_boxes	bool	True	Dibuja recuadros delimitadores alrededor de los objetos detectados. Esencial para la identificación visual y la localización de objetos en imágenes o fotogramas de vídeo.
        line_width	None or int	None	Especifica la anchura de línea de los cuadros delimitadores. Si NoneEl ancho de línea se ajusta automáticamente en función del tamaño de la imagen. Proporciona personalización visual para mayor claridad.
        """
        results = model.track(frame, conf=0.7, iou=0.5, persist=True, verbose=False)

        # Visualizamos los resultados en el frame
        annotated_frame = results[0].plot()
        """
        Trabajo con los resultados

        """

        # Nombre ventana
        cv2.imshow("LPR GAO v.1.00", annotated_frame)

        # q para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Final del video
        break

# Suelta y destruye
cap.release()
cv2.destroyAllWindows()