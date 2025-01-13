
import pyrender
import trimesh
import numpy as np
import base64
import psycopg2
from psycopg2 import sql
from io import BytesIO
from PIL import Image
import argparse

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def center_mesh(mesh):
    """Centrar la malla en el origen de la escena."""
    if isinstance(mesh, trimesh.Scene):
        mesh.apply_translation(-mesh.bounding_box.centroid)
    elif isinstance(mesh, trimesh.Trimesh):
        mesh.apply_translation(-mesh.centroid)
    return mesh

def rotate_mesh(mesh, rotation=None, quaternion=None):
    """Rota la malla ANTES de crear el objeto pyrender.Mesh."""
    if rotation or quaternion:
        rotation_matrix = None
        if quaternion:
            rotation_matrix = trimesh.transformations.quaternion_matrix(quaternion)
        elif rotation:
            rotation_matrix = trimesh.transformations.euler_matrix(*rotation, axes='sxyz')

        if isinstance(mesh, trimesh.Scene):
            for name, geom in mesh.geometry.items():
                geom.apply_transform(rotation_matrix)
        elif isinstance(mesh, trimesh.Trimesh):
            mesh.apply_transform(rotation_matrix)
    return mesh

def create_pyrender_scene(mesh):
    """Crea una escena de pyrender a partir de una malla de trimesh."""
    pyrender_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

    if isinstance(mesh, trimesh.Scene):
        pyrender_scene = pyrender.Scene.from_trimesh_scene(mesh, bg_color=[0.0, 0.0, 0.0, 0.0])
    elif isinstance(mesh, trimesh.Trimesh):
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        pyrender_scene.add(pyrender_mesh)

    return pyrender_scene

# Carga un modelo 3D, ya sea GLTF o FBX, y lo convierte a una escena de pyrender
def load_model(file_path):
    if file_path.endswith('.gltf') or file_path.endswith('.glb'):
        # Cargar un archivo GLTF
        trimesh_scene = trimesh.load(file_path)        
        return center_mesh(trimesh_scene)

    else:
        raise ValueError("Formato de archivo no soportado")

# Convertir la imagen a formato base64
def image_to_base64(image_array):
    """Convierte una imagen numpy en base64."""
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

def add_lights_to_scene(scene):
    """
    Agrega iluminación general a la escena para una visualización adecuada.
    Incluye luz direccional, luz ambiental y luz puntual.
    """
    # Luz direccional simula la luz del sol
    directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5)
    scene.add(directional_light, pose=np.eye(4))  # Desde la dirección estándar

    # Luz puntual para resaltar detalles desde una posición elevada
    point_light_pose = np.eye(4)
    point_light_pose[:3, 3] = [0, 0, 10]  # Por encima del modelo
    point_light = pyrender.PointLight(color=[1.0, 1.0, 0.8], intensity=800)
    scene.add(point_light, pose=point_light_pose)

    # Luz hemisférica o similar (simulación de luz ambiental)
    ambient_light = pyrender.SpotLight(color=[0.6, 0.6, 0.6], intensity=0.8, innerConeAngle=1.0, outerConeAngle=np.pi/2.0)
    ambient_light_pose = np.eye(4)
    ambient_light_pose[:3, 3] = [5, 5, 5]  # Luz de un lado para más realismo
    scene.add(ambient_light, pose=ambient_light_pose)

    return scene

# Calcula las dimensiones para centrar y encuadrar el modelo
def calculate_camera_distance(scene, fov=60, margin_factor=1.5):
    """
    Calcula la distancia adecuada para una cámara de perspectiva para ver toda la escena.
    Args:
        scene: pyrender.Scene object.
        fov: Field of View (en grados) para la cámara de perspectiva (default 60 grados).
        margin_factor: Factor de margen para asegurar que el modelo cabe completamente.
    Returns:
        La distancia que la cámara debe estar para encuadrar el modelo.
    """
    bbox = scene.extents  # Dimensiones [x, y, z] de la caja de contorno
    max_extent = max(bbox)  # La extensión máxima es la base para calcular la distancia
    fov_rad = np.radians(fov)  # Convertir a radianes
    return (max_extent * margin_factor) / (2 * np.tan(fov_rad / 2))

# Agrega una cámara de perspectiva a la escena
def add_perspective_camera(scene, fov=60, margin_factor=1.5):
    """
    Agrega una cámara de perspectiva que centra y encuadra el modelo.
    Args:
        scene: pyrender.Scene object.
        fov: Campo de visión vertical de la cámara (default 60 grados).
        margin_factor: Factor de margen para ajustar el encuadre.
    Returns:
        cam_node: Nodo de la cámara en la escena.
    """
    distance = calculate_camera_distance(scene, fov, margin_factor)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov))
    
    # Posicionar la cámara centrada en la escena y alejada
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, distance]  # Mueve la cámara en el eje Z positivo
    cam_node = scene.add(camera, pose=camera_pose)
    
    return cam_node

def render_scene(scene, width, height):
    """Renderiza la escena rotada según ángulos de Euler."""
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.RGBA)
        return color
    finally:
        renderer.delete()

# Guarda imágenes base64 en una base de datos PostgreSQL
def save_to_db(db_config, table_name, image_base64, nombre_imagen, rotation=None, quaternion=None, control=None):
    try:
        # Conectar a la base de datos
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Crear la conexión y cursor de manera segura
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cursor:

                # Preparar la consulta base
                query = sql.SQL("""
                    INSERT INTO {table_name} 
                    (nombre, img, timestamp, rotacion_x, rotacion_y, rotacion_z, rotacion_w, control)
                    VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s)
                """)

                # Determinar si usar quaternion o rotation
                if quaternion:
                    values = (nombre_imagen, image_base64, quaternion[0], quaternion[1], quaternion[2], quaternion[3], control)
                elif rotation:
                    values = (nombre_imagen, image_base64, rotation[0], rotation[1], rotation[2], None, control)

                cursor.execute(query.format(table_name=sql.Identifier(table_name)), values)

                # Confirmar la transacción
                conn.commit()
                print("Imagen insertada correctamente en la base de datos.")
                
    except psycopg2.Error as db_error:
        print(f"Error en la base de datos: {db_error.pgcode} - {db_error.pgerror}")
    except ValueError as ve:
        print(f"Error de validación: {ve}")
    except Exception as e:
        print(f"Error inesperado: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():

    db_config = {
        "dbname": "test_inta",
        "user": "postgres",
        "password": "M4kenai",
        "host": "192.168.10.43",
        "port": "5432",
    }
    table_name = "images"
    output_width = 700 
    output_height = 700 

    parser = argparse.ArgumentParser(description="Renderizar un modelo GLTF y almacenarlo en una base de datos.")
    parser.add_argument("--file", type=str, required=True, help="Ruta al archivo GLTF.")
    parser.add_argument("--rotation", type=float, nargs=3, help="Ángulos de rotación (en radianes) como (x, y, z).")
    parser.add_argument("--quaternion", type=float, nargs=4, help="Cuaternión de rotación como (x, y, z, w).")

    args = parser.parse_args()
   
    if args.rotation and args.quaternion:
        parser.error("Solo se puede usar '--rotation' o '--quaternion', no ambos.")
    if not args.rotation and not args.quaternion:
        parser.error("Debe especificar '--rotation' o '--quaternion'.")

    # Cargar y transformar el modelo
    mesh = load_model(args.file)  
    centered_mesh = center_mesh(mesh.copy())
    rotated_mesh = rotate_mesh(centered_mesh.copy(), 
        rotation=tuple(args.rotation) if args.rotation else None,
        quaternion=tuple(args.quaternion) if args.quaternion else None)

    # Crear la escena, añadir luces y cámara
    scene = create_pyrender_scene(rotated_mesh)
    add_lights_to_scene(scene)
    add_perspective_camera(scene, fov=60)

    # Renderizar la escena
    rendered_image = render_scene(scene, output_width, output_height)

    # Convertir en Base64
    image_base64 = image_to_base64(rendered_image)

    # Cargar en BD
    save_to_db(
        db_config, 
        table_name, 
        image_base64, 
        os.path.basename(args.file),
        rotation=tuple(args.rotation) if args.rotation else None,
        quaternion=tuple(args.quaternion) if args.quaternion else None,
        control=args.rotation if args.rotation else args.quaternion
    )    

if __name__ == "__main__":
    main()
