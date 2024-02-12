import cv2
from ultralytics import YOLO

# Cargamos video o streaming
cap = cv2.VideoCapture("videos/traffic.mp4")
# Cargamos el modelo de YoloV8
"""
Modelo	    tamaño  acc1    acc5    Velocidad  VelocidadTRS    parámetros  FLOPs
YOLOv8n-cls	224	    69.0	88.3	12.9	    0.31	        2.7	        4.3
YOLOv8s-cls	224	    73.8	91.7	23.4	    0.35	        6.4	        13.5
YOLOv8m-cls	224	    76.8	93.5	85.4	    0.62	        17.0	    42.7
YOLOv8l-cls	224	    76.8	93.5	163.0	    0.87	        37.5	    99.7
YOLOv8x-cls	224	    79.0	94.6	232.0	    1.01	        57.4	    154.8
"""
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
        
        model.names
        {0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'}
        """
        results = model.track(frame, conf=0.7, iou=0.5, persist=True, verbose=False, classes=[2,3,5,7])

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