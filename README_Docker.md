# 🐳 Water Analysis Framework - Docker Setup

Este documento explica cómo ejecutar el Water Analysis Framework usando Docker para un entorno completamente reproducible.

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)

```bash
# Construir y ejecutar todo automáticamente
./build_and_run.sh

# O solo construir la imagen
./build_and_run.sh build

# O solo ejecutar el análisis
./build_and_run.sh run

# O iniciar Jupyter Lab
./build_and_run.sh jupyter
```

### Opción 2: Docker Compose Manual

```bash
# Construir la imagen
docker-compose build

# Ejecutar análisis
docker-compose up water-analysis

# Ejecutar Jupyter Lab
docker-compose --profile jupyter up jupyter
```

## 📋 Requisitos Previos

- **Docker** (versión 20.10+)
- **Docker Compose** (versión 2.0+)
- **Git** (para clonar el repositorio)

### Instalación de Docker

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

#### macOS:
```bash
# Instalar Docker Desktop desde https://www.docker.com/products/docker-desktop
```

#### Windows:
```bash
# Instalar Docker Desktop desde https://www.docker.com/products/docker-desktop
```

## 🏗️ Estructura del Proyecto

```
water-analysis-framework/
├── Dockerfile                 # Configuración de la imagen Docker
├── docker-compose.yml         # Configuración de servicios
├── build_and_run.sh          # Script de construcción y ejecución
├── requirements.txt           # Dependencias de Python
├── test.py                   # Script principal de análisis
├── water_pixels_analysis.py  # Análisis de píxeles por año
├── cloud_water_correlation.py # Análisis de correlación nube-agua
├── README.md                 # Documentación principal
├── README_Docker.md          # Esta documentación
├── data/                     # Datos de entrada (montado)
├── outputs/                  # Resultados (montado)
├── logs/                     # Logs (montado)
└── notebooks/                # Jupyter notebooks (montado)
```

## 🎯 Modos de Ejecución

### 1. Análisis Automático
```bash
./build_and_run.sh
```
- Construye la imagen Docker
- Ejecuta el análisis completo de agua
- Genera imágenes NDWI y análisis estadísticos

### 2. Modo Interactivo (Jupyter)
```bash
./build_and_run.sh jupyter
```
- Inicia Jupyter Lab en http://localhost:8888
- Permite análisis interactivo y desarrollo
- Acceso a todos los scripts y datos

### 3. Solo Construcción
```bash
./build_and_run.sh build
```
- Solo construye la imagen Docker
- Útil para verificar la configuración

### 4. Solo Ejecución
```bash
./build_and_run.sh run
```
- Ejecuta el análisis (requiere imagen construida)
- Más rápido que la opción completa

## 📊 Resultados

Los resultados se guardan en el directorio `outputs/`:

- `ndwi_images/` - Imágenes NDWI para cada fecha
- `water_pixels_by_year.png` - Análisis anual de píxeles
- `cloud_water_correlation.png` - Correlación nube-agua
- `cloud_masking_analysis.png` - Análisis de enmascaramiento
- `ndwi_data.nc` - Datos NDWI en formato NetCDF
- `*.log` - Logs detallados del análisis

## 🔧 Personalización

### Cambiar Configuración del Análisis

Edita `test.py` antes de construir la imagen:

```python
# Área de estudio
bbox = [-73.710, 4.600, -73.695, 4.615]  # Tu área de interés

# Período de tiempo
start_date = "2014-01-01"
end_date = "2024-12-31"

# Parámetros
max_cloud_cover = 20.0
max_items = 200
```

### Agregar Datos Personalizados

Coloca tus datos en el directorio `data/`:

```bash
mkdir -p data
# Copia tus archivos de datos aquí
```

### Modificar Dependencias

Edita `requirements.txt` y reconstruye:

```bash
# Editar requirements.txt
./build_and_run.sh build
```

## 🐛 Solución de Problemas

### Error de Permisos
```bash
# Si tienes problemas de permisos con Docker
sudo usermod -aG docker $USER
# Reinicia tu sesión
```

### Error de Memoria
```bash
# Aumentar memoria disponible para Docker
# En Docker Desktop: Settings > Resources > Memory
```

### Error de Red
```bash
# Verificar conectividad
docker run --rm alpine ping google.com
```

### Limpiar Recursos
```bash
# Limpiar contenedores e imágenes
./build_and_run.sh clean
```

## 📝 Logs y Debugging

Los logs se guardan en `logs/`:

```bash
# Ver logs en tiempo real
docker-compose logs -f water-analysis

# Ver logs específicos
tail -f logs/water_analysis.log
```

## 🔄 Actualizaciones

Para actualizar el framework:

```bash
# Obtener cambios del repositorio
git pull

# Reconstruir imagen
./build_and_run.sh build

# Ejecutar análisis actualizado
./build_and_run.sh run
```

## 🌐 Acceso Web

### Jupyter Lab
- URL: http://localhost:8888
- Sin contraseña (configurado para desarrollo)

### Archivos de Resultados
Los resultados están disponibles en el directorio `outputs/` del host.

## 📞 Soporte

Si encuentras problemas:

1. Verifica que Docker esté funcionando: `docker --version`
2. Revisa los logs: `docker-compose logs`
3. Limpia y reconstruye: `./build_and_run.sh clean && ./build_and_run.sh build`
4. Abre un issue en el repositorio

## 🎉 ¡Listo!

Tu entorno de análisis de agua está listo para usar. El framework es completamente reproducible y puede ejecutarse en cualquier sistema con Docker.

---

**Nota**: Este entorno Docker incluye todas las dependencias necesarias y está optimizado para análisis de datos satelitales a gran escala. 