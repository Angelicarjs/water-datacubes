"""
Water Analysis Framework using Microsoft Planetary Computer
==========================================================

"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import shapely.geometry
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import time

# Remote sensing libraries
import stackstac
import pystac_client
import planetary_computer
import rioxarray
from rasterio.features import geometry_mask

# Visualization
import folium

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('water_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class WaterAnalyzer:
    def __init__(self, collection="sentinel-2-l2a", assets=None, band_aliases=None,
                 chunksize=2048, resolution=100, epsg=32618):
        self.collection = collection
        self.assets = assets or ["B03", "B08", "SCL"]  # Include SCL for masking
        self.band_aliases = band_aliases or {"green": "B03", "nir": "B08"}
        self.chunksize = chunksize
        self.resolution = resolution
        self.epsg = epsg
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        self.data = None
        self.ndwi = None
        self.items = None
        logger.info(f"Initialized WaterAnalyzer with collection: {collection}")

    def check_connectivity(self):
        """Verifica la conectividad con Microsoft Planetary Computer"""
        try:
            logger.info("🔗 Verificando conectividad con Microsoft Planetary Computer...")
            # Intenta una búsqueda simple para verificar conectividad
            test_search = self.catalog.search(
                collections=[self.collection],
                bbox=[-73.705, 4.605, -73.700, 4.610],
                datetime="2024-01-01/2024-01-02",
                limit=1
            )
            test_items = list(test_search.item_collection())
            logger.info("✅ Conectividad verificada exitosamente")
            return True
        except Exception as e:
            logger.error(f"❌ Error de conectividad: {str(e)}")
            logger.info("💡 Sugerencias:")
            logger.info("   - Verifica tu conexión a internet")
            logger.info("   - Intenta ejecutar el script más tarde")
            logger.info("   - El servicio puede estar temporalmente no disponible")
            return False

    # 🔹 Helper to centralize threshold logic
    def get_threshold_for_baseline(self, baseline: float) -> float:
        if baseline >= 5.00:
            return 0.05
        elif baseline >= 4.00:
            return 0.07
        else:
            return 0.1

    def get_threshold_for_item(self, item_index: int) -> float:
        if self.items is None or item_index >= len(self.items):
            return 0.1
        baseline_str = self.items[item_index].properties.get('s2:processing_baseline', '0.0')
        try:
            return self.get_threshold_for_baseline(float(baseline_str))
        except (ValueError, TypeError):
            return 0.1

    # 🔹 Helper to count water pixels CHANGE!
    def count_water_pixels(self, ndwi_array: xr.DataArray, threshold: float):
        water_pixels = (ndwi_array > threshold).sum().values
        total_pixels = ndwi_array.size
        return water_pixels, (water_pixels / total_pixels) * 100

    def search_data(self, bbox: List[float], start_date="2014-01-01", end_date="2024-12-31",
                    max_cloud_cover=10.0, max_items=200) -> List:
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 Intento {attempt + 1}/{max_retries}: Buscando datos satelitales...")
                search = self.catalog.search(
                    collections=[self.collection],
                    bbox=bbox,
                    datetime=f"{start_date}/{end_date}",
                    query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                    limit=max_items
                )
                self.items = list(search.item_collection())
                if not self.items:
                    raise ValueError("No data found")
                logger.info(f"✅ Búsqueda exitosa: {len(self.items)} items encontrados")
                return self.items
                
            except Exception as e:
                logger.warning(f"⚠️ Intento {attempt + 1} falló: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponencial
                else:
                    logger.error(f"❌ Todos los intentos fallaron. Error final: {str(e)}")
                    raise

    def load_data(self) -> xr.DataArray:
        if not self.items:
            raise ValueError("Run search_data() first")
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"📥 Intento {attempt + 1}/{max_retries}: Cargando datos...")
                self.data = (
                    stackstac.stack(
                        items=self.items,
                        assets=self.assets,
                        chunksize=min(self.chunksize, 1024),
                        resolution=self.resolution,
                        epsg=self.epsg
                    )
                    .where(lambda x: x > 0, other=np.nan)
                    .chunk({'time': 1, 'band': -1, 'x': min(self.chunksize, 1024), 'y': min(self.chunksize, 1024)})
                )
                logger.info(f"✅ Carga exitosa - Shape: {self.data.shape}")
                return self.data
                
            except Exception as e:
                logger.warning(f"⚠️ Intento {attempt + 1} falló: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponencial
                else:
                    logger.error(f"❌ Todos los intentos fallaron. Error final: {str(e)}")
                    raise

    # 🔹 Main NDWI calculation (single version, always cloud-masked)
    def calculate_ndwi_only(self) -> xr.DataArray:
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")

        logger.info("🌊 Calculating NDWI with cloud masking...")
        start_time = time.time()

        green = self.data.sel(band="B03")
        nir = self.data.sel(band="B08")
        scl = self.data.sel(band="SCL")

        valid_mask = ~((scl == 1) | (scl == 2) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10))
        green = green.where(valid_mask)
        nir = nir.where(valid_mask)

        self.ndwi = (green - nir) / (green + nir)
        self.ndwi = self.ndwi.compute()

        if 'band' in self.ndwi.dims:
            self.ndwi = self.ndwi.squeeze('band')
        for dim in list(self.ndwi.dims):
            if dim not in ['time', 'y', 'x']:
                self.ndwi = self.ndwi.squeeze(dim)

        logger.info(f"✅ NDWI calculated - Shape: {self.ndwi.shape} | Time: {time.time() - start_time:.2f}s")

        # Water stats
        total_water_pixels = 0
        total_pixels = self.ndwi.size
        baseline_summary = {}

        for i, time_val in enumerate(self.ndwi.time.values):
            threshold = self.get_threshold_for_item(i)
            ndwi_slice = self.ndwi.sel(time=time_val)
            water_pixels, _ = self.count_water_pixels(ndwi_slice, threshold)
            total_water_pixels += water_pixels

            baseline_str = self.items[i].properties.get('s2:processing_baseline', '0.0') if i < len(self.items) else "0.0"
            baseline_key = f"{float(baseline_str):.2f}"
            baseline_summary.setdefault(baseline_key, {'count': 0, 'threshold': threshold})
            baseline_summary[baseline_key]['count'] += 1

        water_percentage = (total_water_pixels / total_pixels) * 100
        logger.info("📊 Baseline usage summary:")
        for baseline, info in baseline_summary.items():
            logger.info(f"   Baseline {baseline}: {info['count']} images, threshold {info['threshold']}")

        return self.ndwi

    def save_all_ndwi_images(self, save_dir: str = "ndwi_images"):
        """
        Save NDWI images for all time steps using centralized threshold and water pixel counting.
        """
        if self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_ndwi_only() first.")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"📊 Saving NDWI images for all {len(self.ndwi.time)} time steps...")
        
        for i, time_val in enumerate(self.ndwi.time.values):
            # Get NDWI data for this time step
            ndwi_slice = self.ndwi.sel(time=time_val)
            
            # Ensure the slice is 2D for plotting
            if len(ndwi_slice.dims) > 2:
                extra_dims = [dim for dim in ndwi_slice.dims if dim not in ['y', 'x', 'lat', 'lon']]
                for dim in extra_dims:
                    ndwi_slice = ndwi_slice.isel({dim: 0})
            
            # Convert to numpy array and ensure it's 2D
            ndwi_array = ndwi_slice.values
            if ndwi_array.ndim > 2:
                ndwi_array = ndwi_array[0] if ndwi_array.shape[0] == 1 else ndwi_array.squeeze()
            
            # Create filename with date
            date_str = pd.to_datetime(time_val).strftime("%Y%m%d")
            filename = f"ndwi_{date_str}.png"
            save_path = os.path.join(save_dir, filename)
            
            # Get threshold using centralized method
            threshold = self.get_threshold_for_item(i)
            
            # Get baseline for this item
            baseline_str = self.items[i].properties.get('s2:processing_baseline', '0.0') if i < len(self.items) else "0.0"
            baseline = float(baseline_str)
            
            # Print date, baseline and threshold
            print(f"📅 {pd.to_datetime(time_val).strftime('%Y-%m-%d')} | Baseline: {baseline:.2f} | Threshold: {threshold}")
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Main NDWI plot
            plt.subplot(1, 2, 1)
            im1 = plt.imshow(ndwi_array, cmap='RdYlBu_r', vmin=-1, vmax=1)
            plt.colorbar(im1, label='NDWI')
            plt.title(f'NDWI - {pd.to_datetime(time_val).strftime("%Y-%m-%d")}')
            plt.axis('off')
            
            # Water mask plot
            plt.subplot(1, 2, 2)
            water_mask = ndwi_array > threshold
            im2 = plt.imshow(water_mask, cmap='Blues')
            plt.colorbar(im2, label='Water (1) / Non-water (0)')
            plt.title(f'Water Mask (NDWI > {threshold})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Count water pixels using centralized method
            water_pixels, water_percentage = self.count_water_pixels(ndwi_slice, threshold)
            
            logger.info(f"✅ Saved {filename} - Water pixels: {water_pixels:,} ({water_percentage:.2f}%)")
        
        logger.info(f"✅ All NDWI images saved to '{save_dir}' directory")

    def extract_cloud_water_data(self):
        """
        Extract cloud cover and water pixels data using centralized water pixel counting.
        """
        if self.data is None or self.ndwi is None:
            raise ValueError("Both satellite data and NDWI must be loaded. Run load_data() and calculate_ndwi_only() first.")
        
        logger.info("📊 Extracting cloud cover and water pixels data...")
        
        analysis_data = []
        
        for i, time_val in enumerate(self.ndwi.time.values):
            date = pd.to_datetime(time_val)
            
            # Get NDWI data for this time step
            ndwi_slice = self.ndwi.sel(time=time_val)
            
            # Ensure the slice is 2D for analysis
            if len(ndwi_slice.dims) > 2:
                extra_dims = [dim for dim in ndwi_slice.dims if dim not in ['y', 'x', 'lat', 'lon']]
                for dim in extra_dims:
                    ndwi_slice = ndwi_slice.isel({dim: 0})
            
            # Convert to numpy array and ensure it's 2D
            ndwi_array = ndwi_slice.values
            if ndwi_array.ndim > 2:
                ndwi_array = ndwi_array[0] if ndwi_array.shape[0] == 1 else ndwi_array.squeeze()
            
            # Get threshold using centralized method
            threshold = self.get_threshold_for_item(i)
            
            # Count water pixels using centralized method
            water_pixels, water_percentage = self.count_water_pixels(ndwi_slice, threshold)
            total_pixels = ndwi_slice.size
            
            # Get cloud cover from SCL band
            cloud_cover = None
            try:
                # Get SCL data for this time step
                scl_slice = self.data.sel(band="SCL", time=time_val)
                scl_array = scl_slice.values
                if scl_array.ndim > 2:
                    scl_array = scl_array.squeeze()
                
                # Calculate cloud cover from SCL (classes 1, 2, 3, 8, 9, 10 are clouds/shadows)
                cloud_classes = [1, 2, 3, 8, 9, 10]
                cloud_pixels = np.isin(scl_array, cloud_classes).sum()
                cloud_cover = (cloud_pixels / total_pixels) * 100
                
            except Exception as e:
                logger.warning(f"⚠️  Could not extract cloud cover for {date}: {e}")
                cloud_cover = 0
            
            # Store data
            analysis_data.append({
                'date': date,
                'year': date.year,
                'month': date.month,
                'water_pixels': water_pixels,
                'water_percentage': water_percentage,
                'cloud_cover': cloud_cover,
                'total_pixels': total_pixels
            })
        
        df = pd.DataFrame(analysis_data)
        logger.info(f"✅ Extracted data for {len(df)} time steps")
        
        return df

    def calculate_water_area_per_image(self, threshold: float = None) -> pd.DataFrame:
        """
        Calculate water area for each image using centralized water pixel counting.
        """
        if self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_ndwi_only() first.")
        
        # Get threshold using centralized method if not provided
        if threshold is None:
            threshold = self.get_threshold_for_item(0) if self.items else 0.1
            logger.info(f"🔧 Using centralized threshold: {threshold}")
        
        logger.info(f"🌊 Calculating water area per image with threshold {threshold}...")
        
        # Get pixel resolution in meters
        pixel_resolution = self.resolution  # meters per pixel
        pixel_area = pixel_resolution * pixel_resolution  # square meters per pixel
        
        water_area_data = []
        
        for i, time_val in enumerate(self.ndwi.time.values):
            date = pd.to_datetime(time_val)
            
            # Get NDWI data for this time step
            ndwi_slice = self.ndwi.sel(time=time_val)
            
            # Ensure the slice is 2D for analysis
            if len(ndwi_slice.dims) > 2:
                extra_dims = [dim for dim in ndwi_slice.dims if dim not in ['y', 'x', 'lat', 'lon']]
                for dim in extra_dims:
                    ndwi_slice = ndwi_slice.isel({dim: 0})
            
            # Convert to numpy array and ensure it's 2D
            ndwi_array = ndwi_slice.values
            if ndwi_array.ndim > 2:
                ndwi_array = ndwi_array[0] if ndwi_array.shape[0] == 1 else ndwi_array.squeeze()
            
            # Count water pixels using centralized method
            water_pixels, water_percentage = self.count_water_pixels(ndwi_slice, threshold)
            total_pixels = ndwi_slice.size
            
            # Calculate areas
            water_area_sqm = water_pixels * pixel_area  # square meters
            water_area_ha = water_area_sqm / 10000  # hectares
            water_area_km2 = water_area_sqm / 1000000  # square kilometers
            
            total_area_sqm = total_pixels * pixel_area
            total_area_ha = total_area_sqm / 10000
            total_area_km2 = total_area_sqm / 1000000
            
            # Store data
            water_area_data.append({
                'date': date,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'water_pixels': water_pixels,
                'total_pixels': total_pixels,
                'water_percentage': water_percentage,
                'water_area_sqm': water_area_sqm,
                'water_area_ha': water_area_ha,
                'water_area_km2': water_area_km2,
                'total_area_sqm': total_area_sqm,
                'total_area_ha': total_area_ha,
                'total_area_km2': total_area_km2,
                'pixel_resolution_m': pixel_resolution
            })
        
        df = pd.DataFrame(water_area_data)
        
        # Log summary statistics
        logger.info(f"✅ Water area calculation completed for {len(df)} images")
        logger.info(f"📊 Summary statistics:")
        logger.info(f"   Total area per image: {df['total_area_km2'].iloc[0]:.4f} km²")
        logger.info(f"   Average water area: {df['water_area_km2'].mean():.4f} km²")
        logger.info(f"   Min water area: {df['water_area_km2'].min():.4f} km²")
        logger.info(f"   Max water area: {df['water_area_km2'].max():.4f} km²")
        logger.info(f"   Average water percentage: {df['water_percentage'].mean():.2f}%")
        
        return df

    def run_analysis(self, bbox: List[float], start_date: str = "2014-01-01", 
                    end_date: str = "2024-12-31", max_cloud_cover: float = 10.0,
                    max_items: int = 200, output_dir: str = "water_analysis_output"):
        """
        Ejecuta el análisis completo de agua: búsqueda, carga, cálculo NDWI, guardado y análisis.
        """
        logger.info("🚀 Iniciando análisis de agua...")
        start_time = time.time()
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Verificar conectividad antes de empezar
            if not self.check_connectivity():
                raise ConnectionError("No se pudo conectar con Microsoft Planetary Computer")
            
            # Paso 1: Búsqueda de datos
            logger.info("🔍 Buscando datos satelitales...")
            items = self.search_data(bbox, start_date, end_date, max_cloud_cover, max_items)
            logger.info(f"✅ Encontrados {len(items)} items satelitales")
            
            # Paso 2: Carga de datos
            logger.info("📥 Cargando datos...")
            data = self.load_data()
            logger.info(f"✅ Datos cargados - Shape: {data.shape}")
            
            # Paso 3: Cálculo de NDWI
            logger.info("🌊 Calculando NDWI...")
            ndwi = self.calculate_ndwi_only()
            logger.info(f"✅ NDWI calculado - Shape: {ndwi.shape}")
            
            # Paso 4: Guardado de imágenes
            logger.info("🖼️ Guardando imágenes NDWI...")
            images_dir = os.path.join(output_dir, "ndwi_images")
            self.save_all_ndwi_images(images_dir)
            
           
            
            return {
                'bbox': bbox,
                'start_date': start_date,
                'end_date': end_date,
                'items_found': len(items)
            }
            
        except Exception as e:
            logger.error(f"❌ Error durante el análisis: {str(e)}")
            raise


if __name__ == "__main__":
    # Configuración para análisis de 10 años (2014-2024)
    bbox = [-73.705, 4.605, -73.700, 4.610]  # Área pequeña para pruebas
    start_date = "2014-01-01"  # Inicio del análisis de 10 años
    end_date = "2024-12-31"    # Fin del análisis de 10 años
    
    # Crear instancia del analizador
    analyzer = WaterAnalyzer()
    
    try:
        # Ejecutar análisis completo
        results = analyzer.run_analysis(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=10.0,
            max_items=200,
            output_dir="water_analysis_output"
        )
        
        print(f"✅ Análisis completado exitosamente!")
        print(f"📊 Items encontrados: {results['items_found']}")
        print(f"⏱️ Tiempo de ejecución: {results['execution_time']:.2f} segundos")
        print(f"📁 Resultados guardados en: {results['output_dir']}")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()

