import os
import shutil
import cv2
from tkinter import *
from PIL import Image, ImageTk
import threading
import supervision as sv
from ultralytics import YOLO
from datetime import datetime
import uuid

LINE_START = (0, 400)
LINE_END = (1200, 400)
RTSP_CAPTURE_CONFIG = "videos/traffic.mp4"
#RTSP_CAPTURE_CONFIG = "videos/test.mp4"

def main():
    global anchura_pantalla
    global altura_pantalla
    altura_img = int(altura_pantalla / 2)
    anchura_img = int(anchura_pantalla / 2)
    cambio = 0

    model = YOLO("yolov8n.pt")
    
    results = model.track(source=RTSP_CAPTURE_CONFIG, vid_stride=2,conf=0.7, iou=0.5, show=False, stream=True, agnostic_nms=True, persist=True, verbose=False, classes=[2,3,5,7])
    
    line_zone = sv.LineZone(start=sv.Point(LINE_START[0],LINE_START[1]), end=sv.Point(LINE_END[0],LINE_END[1]))
    #line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness = 1,text_thickness = 1,text_scale = 0.5)
    
        
    for result in results:
        frame = result.orig_img
        frame_car = frame.copy()
        antes_estaban = line_zone.tracker_state.copy()
        detections = sv.Detections.from_yolov8(result)
        if result.boxes is not None and result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        cv2.line(frame,LINE_START,LINE_END,(255,0,0),5)   

        
        labels = [f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                  for _, confidence, class_id, tracker_id 
                  in detections]
        
        line_zone.trigger(detections=detections)
        ahora_esta = line_zone.tracker_state.copy
        #line_annotator.annotate(frame, line_zone)
        frame = box_annotator.annotate(scene = frame,detections = detections,labels = labels)
        
        # Mostramos en TK
        frame = cv2.resize(frame, (anchura_img, altura_img))
        b,g,r = cv2.split(frame) # Reasignamos colores
        img = cv2.merge((r,g,b))
        im = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        # Deteccion de corte de linea
        dentro = line_zone.in_count
        fuera = line_zone.out_count
        if cambio != (dentro + fuera):
            # Detectar quien ha cambiado
            ahora_esta = line_zone.tracker_state
            # Actualizamos variables
            cambio = dentro + fuera
            for xyxy, confidence, class_id, tracker_id in detections:
                if tracker_id in antes_estaban:
                    if tracker_id in ahora_esta:
                        if antes_estaban[tracker_id] != line_zone.tracker_state[tracker_id]:
                            today = datetime.today()
                            for_dtae = today.timetuple()
                            date_view = str(for_dtae[2]) + "/" + str(for_dtae[1]) + "/" + str(for_dtae[0]) + "-" + str(for_dtae[3]) + ":" + str(for_dtae[4]) + ":" + str(for_dtae[5])
                            
                            texto1.insert("1.0", date_view + "-" + "Detectado box " + str(tracker_id) + ".Han pasado " + str(cambio) +" vehículos.\n")
                            x1, y1, x2, y2 = xyxy
                            # Capturamos imagen
                            cropped_image = frame_car[int(y1):int(y2),int(x1):int(x2)] 
                            frameNew = cv2.resize(cropped_image, (anchura_img, altura_img))
                            b,g,r = cv2.split(frameNew) # Reasignamos colores
                            img = cv2.merge((r,g,b))
                            im = Image.fromarray(img)
                            img = ImageTk.PhotoImage(image=im)
                            lblCaptura.configure(image=img)
                            lblCaptura.image = img
                            # Salvamos imagen del vehiculo
                            numero_raro = str(uuid.uuid4())
                            nombre_file_coche = "./capturas/coche_" + numero_raro + ".jpg"
                            im.save(nombre_file_coche)
            antes_estaban = ahora_esta.copy()
           
       
def quit(self):
    self.destroy()
    exit()       

def hilos_de_trabajo(): 
    hilo_captura_de_video = threading.Thread(target=main)
    #hilo_deteccion = threading.Thread(target=deteccion_coches)
    hilo_captura_de_video.start()
    #hilo_deteccion.start()
    hilo_captura_de_video.join()
    #hilo_deteccion.join()

if __name__ == "__main__":
    try:
        if os.path.exists("./capturas/"):
            shutil.rmtree("./capturas/")
        os.mkdir("./capturas/")
    except: pass
    # Iniciamos entorno gráfico
    pantalla = Tk()
    # Capturamos resolución de la pantalla
    altura_pantalla = pantalla.winfo_screenheight()
    anchura_pantalla = pantalla.winfo_screenwidth()
    pantalla.title("ALPR by GAO.")
    #Asignamos dimensión de la ventana
    pantalla.geometry(str(anchura_pantalla) + "x" + str(altura_pantalla))
    texto1 = Text(pantalla, width=anchura_pantalla)
    texto1.place(x = 0, y = altura_pantalla/2)
    texto1.insert("1.0", "GAO LPR 1.0\r\n")
    # Video de entrada
    lblVideo = Label(pantalla, command=threading.Thread(target=hilos_de_trabajo).start())
    lblVideo.place(x = 0, y = 0)
    # Captura de vehiculo
    lblCaptura = Label(pantalla)
    lblCaptura.place(x = (anchura_pantalla/2), y = 0)
    pantalla.mainloop()