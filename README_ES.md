# Guía para Crear Nodos Personalizados en ComfyUI

Este documento ofrece un recorrido paso a paso para quienes deseen crear sus propios nodos en ComfyUI. Está pensado para usuarios principiantes e intermedios y toma como referencia las utilidades incluidas en **ComfyUI-Custom-Scripts**.

## 1. Requisitos previos

1. Tener ComfyUI instalado. Puedes seguir las instrucciones en su repositorio oficial.
2. Disponer de Python 3.10 o superior.
3. Contar con un entorno virtual recomendado para gestionar dependencias.

## 2. Estructura básica de un nodo

Los nodos personalizados de este proyecto se guardan en la carpeta `py/`. Cada archivo Python contiene una o más clases que implementan un nodo. Los pasos generales son:

1. Crear un archivo, por ejemplo `mi_nodo.py`, dentro de `py/`.
2. Definir una clase con los métodos y atributos requeridos.
3. Registrar esa clase en `NODE_CLASS_MAPPINGS` para que ComfyUI la detecte.

```python
# py/mi_nodo.py
class MiNodo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"texto": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "saludar"
    CATEGORY = "tutorial"

    def saludar(self, texto):
        return (f"Hola, {texto}",)

NODE_CLASS_MAPPINGS = {"MiNodo|pysssss": MiNodo}
NODE_DISPLAY_NAME_MAPPINGS = {"MiNodo|pysssss": "Mi Nodo"}
```

## 3. Registro automático de nodos

El archivo `__init__.py` del proyecto carga todos los módulos en la carpeta `py/` y combina sus `NODE_CLASS_MAPPINGS`. Por ello, basta con colocar tu archivo en dicha carpeta para que se cargue al iniciar ComfyUI.

```python
# Fragmento de __init__.py
files = glob.glob(os.path.join(py, "*.py"))
for file in files:
    ...
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
```

## 4. Probando tu nodo

1. Copia o enlaza este repositorio dentro de la carpeta `custom_nodes` de ComfyUI.
2. Ejecuta ComfyUI y, en el buscador de nodos, localiza tu nuevo nodo por el nombre indicado en `NODE_DISPLAY_NAME_MAPPINGS`.
3. Arrástralo al lienzo y comprueba su funcionamiento.

## 5. Ejemplos y funcionalidades

Dentro de `py/` encontrarás múltiples ejemplos que puedes utilizar como guía:

- **play_sound.py**: reproduce un sonido al ejecutarse.
- **string_function.py**: realiza operaciones de texto.
- **repeater.py**: genera repeticiones de nodos o valores.
- **constrain_image.py**: ajusta dimensiones de una imagen.

Estas implementaciones muestran distintas posibilidades: manejo de entradas, salidas, widgets y categorización.

## 6. Consejos para continuar

- Revisa el archivo `README.md` para conocer todas las extensiones disponibles en este proyecto.
- Explora la documentación oficial de ComfyUI para profundizar en el sistema de nodos.
- Experimenta creando nodos sencillos antes de pasar a funcionalidades más avanzadas.

Con esta guía deberías poder comenzar a crear tus propios nodos personalizados y ampliar ComfyUI según tus necesidades.
