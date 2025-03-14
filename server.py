from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Función para calcular el ángulo entre tres puntos
def calcular_angulo(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(cos_angulo))
    
    return angulo

@app.route('/procesar', methods=['POST'])
def procesar():
    print("🔹 Recibiendo solicitud...")

    # 📌 Verifica si MIT App Inventor está enviando archivos
    if not request.files:
        print("⚠ No se recibió ningún archivo en request.files")
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    # 🔹 Obtiene el primer archivo recibido, sin importar el nombre de la clave
    file_key = list(request.files.keys())[0]  
    file = request.files[file_key]

    if file.filename == '':
        print("⚠ El archivo no tiene nombre")
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    print(f"✅ Imagen recibida con clave: {file_key}, nombre: {file.filename}")

    # Procesar la imagen
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convertir imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Obtener coordenadas de hombro, codo y muñeca DERECHO
        hombro_der = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        codo_der = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
        muñeca_der = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

        # Obtener coordenadas de hombro, codo y muñeca IZQUIERDO
        hombro_izq = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        codo_izq = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        muñeca_izq = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)

        # Calcular ángulos de ambos lados
        angulo_codo_der = calcular_angulo(hombro_der, codo_der, muñeca_der)
        angulo_codo_izq = calcular_angulo(hombro_izq, codo_izq, muñeca_izq)

        # Calcular diferencia de movimiento entre ambos lados
        diferencia = abs(angulo_codo_der - angulo_codo_izq)

        # Diagnóstico basado en diferencia de movilidad
        if diferencia > 10:  # Si la diferencia es mayor a 10 grados, se considera restricción
            if angulo_codo_der < angulo_codo_izq:
                mensaje = "Movilidad reducida en el HOMBRO DERECHO"
            else:
                mensaje = "Movilidad reducida en el HOMBRO IZQUIERDO"
        else:
            mensaje = "Rango de movimiento similar en ambos hombros"

        print("✅ Datos procesados correctamente")

        return jsonify({
            "angulo_codo_derecho": angulo_codo_der,
            "angulo_codo_izquierdo": angulo_codo_izq,
            "diferencia_movilidad": diferencia,
            "diagnostico": mensaje
        })

    print("⚠ No se detectó la postura en la imagen")
    return jsonify({"error": "No se detectó la postura"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


