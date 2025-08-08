
"""
Water Analysis Framework using Microsoft Planetary Computer
==========================================================

This script provides a generic framework for analyzing water pixels over time
using Sentinel-2 data from Microsoft Planetary Computer.

Features:
- Data retrieval from Planetary Computer
- Water index calculation (NDWI)
- Temporal analysis of water extent
- Visualization and statistics
- Configurable parameters for different study areas
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
import dask
from dask.distributed import progress
import multiprocessing
import time
from tqdm import tqdm

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def configure_dask_for_performance():
    """
    Configure Dask for optimal performance without distributed processing.
    """
    # Get number of CPU cores
    n_cores = multiprocessing.cpu_count()
    n_workers = min(n_cores, 8)  # Limit to 8 workers to avoid memory issues
    
    logger.info(f"🖥️  Detected {n_cores} CPU cores, using {n_workers} workers")
    
    # Configure Dask for local processing
    dask.config.set({
        'array.chunk-size': '128MB',
        'num_workers': n_workers,
        'threads_per_worker': 2,
    })
    
    logger.info(f"🚀 Dask configured for local processing with {n_workers} workers")
    
    return None  # No client needed for local processing

def cleanup_dask_client(client):
    """
    Clean up Dask client and cluster.
    """
    if client is not None:
        logger.info("🧹 Dask client cleanup (not needed for local processing)")
    else:
        logger.info("🧹 No Dask client to clean up")

def print_performance_summary(start_time, end_time, data_shape):
    """
    Print a summary of performance metrics.
    """
    total_time = end_time - start_time
    total_pixels = np.prod(data_shape)
    pixels_per_second = total_pixels / total_time if total_time > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("📊 PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"📊 Total pixels processed: {total_pixels:,}")
    logger.info(f"⚡ Processing speed: {pixels_per_second:,.0f} pixels/second")
    logger.info(f"📈 Data shape: {data_shape}")
    logger.info("="*60)


class WaterAnalyzer:
    """
    A class for analyzing water extent using satellite imagery from Planetary Computer.
    """
    
    def __init__(self, 
                 collection: str = "sentinel-2-l2a",
                 assets: List[str] = None,
                 band_aliases: Dict[str, str] = None,
                 chunksize: int = 2048,
                 resolution: int = 100,
                 epsg: int = 32618,
                 enable_parallel: bool = True,
                 n_workers: int = None):
        """
        Initialize the WaterAnalyzer.
        
        Parameters:
        -----------
        collection : str
            STAC collection to use (default: sentinel-2-l2a)
        assets : List[str]
            Band assets to load (default: ["B03", "B08", "B11"])
        chunksize : int
            Chunk size for dask arrays
        resolution : int
            Resolution in meters
        epsg : int
            EPSG code for projection
        """
        self.collection = collection
        self.assets = assets or ["B03", "B08"]  # Green, NIR
        self.band_aliases = band_aliases or {"green": "B03", "nir": "B08"}
        self.chunksize = chunksize
        self.resolution = resolution
        self.epsg = epsg
        self.enable_parallel = enable_parallel
        self.n_workers = n_workers or min(multiprocessing.cpu_count(), 8)
        
        # Initialize catalog
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        # Data storage
        self.data = None
        self.ndwi = None
        self.items = None
        self.dask_client = None
        
        logger.info(f"Initialized WaterAnalyzer with collection: {collection}")
        if self.enable_parallel:
            logger.info(f"🚀 Parallel processing enabled with {self.n_workers} workers")
    
    def start_dask_client(self):
        """
        Start Dask client for parallel processing.
        """
        if self.enable_parallel and self.dask_client is None:
            logger.info("🚀 Configuring Dask for local parallel processing...")
            self.dask_client = configure_dask_for_performance()
            return self.dask_client
        return None
    
    def stop_dask_client(self):
        """
        Stop Dask client.
        """
        if self.dask_client is not None:
            cleanup_dask_client(self.dask_client)
            self.dask_client = None
    
    def search_data(self,
                   bbox: List[float],
                   start_date: str = "2014-01-01",
                   end_date: str = "2024-12-31",
                   max_cloud_cover: float = 10.0,
                   max_items: int = 200) -> List:
        """
        Search for satellite data in the specified area and time period.
        
        Parameters:
        -----------
        bbox : List[float]
            Bounding box [lon_min, lat_min, lon_max, lat_max]
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        max_cloud_cover : float
            Maximum cloud cover percentage
        max_items : int
            Maximum number of items to retrieve
            
        Returns:
        --------
        List of STAC items
        """
        logger.info(f"🔍 Searching for data from {start_date} to {end_date}")
        logger.info(f"📍 Bounding box: {bbox}")
        logger.info(f"☁️  Maximum cloud cover: {max_cloud_cover}%")
        logger.info(f"📊 Maximum items to retrieve: {max_items}")
        
        # Perform search
        logger.info("🔎 Executing STAC search query...")
        search = self.catalog.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            limit=max_items
        )
        
        self.items = list(search.item_collection())
        logger.info(f"✅ Found {len(self.items)} items")
        
        if len(self.items) > 0:
            # Log some details about the found items
            first_item = self.items[0]
            last_item = self.items[-1]
            logger.info(f"📅 Date range: {first_item.datetime.strftime('%Y-%m-%d')} to {last_item.datetime.strftime('%Y-%m-%d')}")
            
            # Calculate average cloud cover
            cloud_covers = [item.properties.get('eo:cloud_cover', 0) for item in self.items]
            avg_cloud_cover = sum(cloud_covers) / len(cloud_covers)
            logger.info(f"☁️  Average cloud cover: {avg_cloud_cover:.1f}%")
        
        if len(self.items) == 0:
            raise ValueError("No data found for the specified parameters")
        
        return self.items
    
    def load_data(self) -> xr.DataArray:
        """
        Load and stack the satellite data.
        
        Returns:
        --------
        xarray.DataArray with the stacked data
        """
        if self.items is None or len(self.items) == 0:
            raise ValueError("No items available. Run search_data() first.")
        
        logger.info(f"📥 Loading data for {len(self.items)} items")
        logger.info(f"🔧 Configuration: chunksize={self.chunksize}, resolution={self.resolution}m, EPSG={self.epsg}")
        logger.info(f"🎯 Target assets: {self.assets}")
        
        # Start Dask client if parallel processing is enabled
        if self.enable_parallel:
            self.start_dask_client()
        
        # Optimize chunking for parallel processing
        optimal_chunksize = min(self.chunksize, 1024)  # Smaller chunks for better parallelization
        logger.info(f"🔧 Using optimized chunk size: {optimal_chunksize}")
        
        # Stack the data with optimized settings
        logger.info("🔄 Stacking satellite data with parallel optimization...")
        start_time = time.time()
        
        # Prepare stackstac parameters
        stack_params = {
            'items': self.items,
            'assets': self.assets,
            'chunksize': optimal_chunksize,
            'resolution': self.resolution,
            'properties': True,  # Include properties for band names
        }
        
        # Only add epsg if specified
        if self.epsg is not None:
            stack_params['epsg'] = self.epsg
        
        self.data = (
            stackstac.stack(**stack_params)
            .where(lambda x: x > 0, other=np.nan)  # Sentinel-2 uses 0 as nodata
            .chunk({
                'time': 1,  # Process one time step at a time
                'band': -1,  # Keep all bands together
                'x': optimal_chunksize,
                'y': optimal_chunksize
            })
        )
        
        load_time = time.time() - start_time
        logger.info(f"⚡ Data loading completed in {load_time:.2f} seconds")
        
        logger.info(f"✅ Data loaded successfully!")
        logger.info(f"📊 Data shape: {self.data.shape}")
        logger.info(f"🎨 Available bands: {list(self.data.band.values)}")
        
        # Log data size information
        total_pixels = np.prod(self.data.shape)
        data_size_gb = self.data.nbytes / (1024**3)
        logger.info(f"💾 Total pixels: {total_pixels:,}")
        logger.info(f"💾 Estimated data size: {data_size_gb:.2f} GB")
        
        return self.data
    
    def calculate_water_indices(self, show_progress: bool = True) -> Dict[str, xr.DataArray]:
        """
        Calculate water indices (NDWI) with detailed progress tracking.
        
        Parameters:
        -----------
        show_progress : bool
            Whether to show progress bars and detailed progress information
            
        Returns:
        --------
        Dictionary containing NDWI array
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        logger.info("🌊 Calculating water indices...")
        logger.info("📐 Using bands: Green (B03), NIR (B08)")
        
        # Calculate water indices with detailed progress tracking
        start_time = time.time()
        
        if show_progress:
            logger.info("📊 Step 1/5: Extracting individual bands...")
        
        # Get bands with progress tracking
        green = self.data.sel(band="green")
        nir = self.data.sel(band="nir")
        
        if show_progress:
            logger.info(f"✅ Step 1/5 completed: Band shapes - Green: {green.shape}, NIR: {nir.shape}")
            logger.info("📊 Step 2/5: Creating NDWI calculation graph...")
        
        # Calculate NDWI with step-by-step progress
        logger.info("🧮 Calculating NDWI: (Green - NIR) / (Green + NIR)")
        
        if show_progress:
            logger.info("📊 Step 3/5: Computing Green - NIR...")
            green_minus_nir = green - nir
            logger.info("✅ Step 3/5 completed: Green - NIR")
            
            logger.info("📊 Step 4/5: Computing Green + NIR...")
            green_plus_nir = green + nir
            logger.info("✅ Step 4/5 completed: Green + NIR")
            
            logger.info("📊 Step 5/5: Computing NDWI division...")
            self.ndwi = green_minus_nir / green_plus_nir
            logger.info("✅ Step 5/5 completed: NDWI calculation")
        else:
            # Direct calculation without progress tracking
            self.ndwi = (green - nir) / (green + nir)
        
        # Compute indices with progress tracking
        if self.enable_parallel and self.dask_client is not None:
            logger.info("🚀 Computing water indices in parallel with progress tracking...")
            
            if show_progress:
                logger.info("📊 Computing NDWI with parallel processing...")
                # Use Dask's progress function for detailed progress tracking
                with progress(self.ndwi) as pbar:
                    self.ndwi = self.ndwi.compute()
                logger.info("✅ NDWI computation completed!")
            else:
                # Simple computation without progress bars
                self.ndwi = self.ndwi.compute()
                
        else:
            logger.info("🔄 Computing water indices sequentially...")
            
            if show_progress:
                logger.info("📊 Computing NDWI sequentially...")
                self.ndwi = self.ndwi.compute()
                logger.info("✅ NDWI computation completed!")
            else:
                self.ndwi = self.ndwi.compute()
        
        indices_time = time.time() - start_time
        logger.info(f"⚡ Water indices calculation completed in {indices_time:.2f} seconds")
        
        # Log statistics about the calculated indices
        logger.info("📊 Water indices statistics:")
        logger.info(f"   NDWI range: {float(self.ndwi.min()):.3f} to {float(self.ndwi.max()):.3f}")
        logger.info(f"   NDWI mean: {float(self.ndwi.mean()):.3f}")
        
        logger.info("✅ Water indices calculated successfully")
        
        result = {"ndwi": self.ndwi}
        return result
    
    def get_computation_status(self) -> Dict[str, Any]:
        """
        Get detailed status of Dask computation including task progress and worker information.
        
        Returns:
        --------
        Dictionary with computation status information
        """
        status = {
            "dask_client_available": self.dask_client is not None,
            "parallel_enabled": self.enable_parallel,
            "n_workers": self.n_workers
        }
        
        if self.dask_client is not None:
            try:
                # Get worker information
                workers = self.dask_client.scheduler_info()['workers']
                status.update({
                    "active_workers": len(workers),
                    "total_memory": sum(w['memory'] for w in workers.values()),
                    "available_memory": sum(w['memory'] for w in workers.values()),
                    "cpu_percent": sum(w.get('cpu', 0) for w in workers.values()) / len(workers) if workers else 0
                })
                
                # Get task information if available
                if hasattr(self.dask_client, 'scheduler'):
                    tasks = self.dask_client.scheduler_info().get('tasks', {})
                    status.update({
                        "total_tasks": len(tasks),
                        "running_tasks": len([t for t in tasks.values() if t.get('state') == 'running']),
                        "waiting_tasks": len([t for t in tasks.values() if t.get('state') == 'waiting']),
                        "completed_tasks": len([t for t in tasks.values() if t.get('state') == 'memory'])
                    })
                    
            except Exception as e:
                status["error"] = str(e)
        
        return status
    
    def calculate_ndwi_only(self, show_progress: bool = True) -> xr.DataArray:
        """
        Calculate only NDWI with detailed step-by-step progress tracking.
        
        Parameters:
        -----------
        show_progress : bool
            Whether to show progress bars and detailed progress information
            
        Returns:
        --------
        NDWI DataArray
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        logger.info("🌊 Calculating NDWI only with detailed progress tracking...")
        logger.info("📐 Formula: NDWI = (Green - NIR) / (Green + NIR)")
        
        start_time = time.time()
        
        if show_progress:
            logger.info("📊 Step 1/5: Extracting Green band (B03)...")
        
        # Extract bands
        green = self.data.sel(band="B03")
        nir = self.data.sel(band="B08")
        scl = self.data.sel(band="SCL")
        
        if show_progress:
            logger.info(f"✅ Step 1/6 completed: Green band shape {green.shape}")
            logger.info("📊 Step 2/6: Extracting NIR band (B08)...")
            logger.info(f"✅ Step 2/6 completed: NIR band shape {nir.shape}")
            logger.info("📊 Step 3/6: Creating cloud mask...")
        
        # Create cloud mask (SCL values: 1=saturated, 2=dark, 3=shadow, 8,9=cloud, 10=cirrus)
        valid_mask = ~((scl == 1) | (scl == 2) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10))
        
        # Apply mask to bands
        green = green.where(valid_mask)
        nir = nir.where(valid_mask)
        
        if show_progress:
            logger.info(f"✅ Step 3/6 completed: Cloud mask applied")
            logger.info("📊 Step 4/6: Computing Green - NIR...")
        
        # Calculate numerator: Green - NIR
        green_minus_nir = green - nir
        
        if show_progress:
            logger.info("✅ Step 4/6 completed: Green - NIR")
            logger.info("📊 Step 5/6: Computing Green + NIR...")
        
        # Calculate denominator: Green + NIR
        green_plus_nir = green + nir
        
        if show_progress:
            logger.info("✅ Step 5/6 completed: Green + NIR")
            logger.info("📊 Step 6/6: Computing NDWI division...")
        
        # Calculate NDWI
        self.ndwi = green_minus_nir / green_plus_nir
        
        if show_progress:
            logger.info("✅ Step 6/6 completed: NDWI calculation")
            logger.info("📊 Computing NDWI with parallel processing...")
        
        # Compute the result
        if self.enable_parallel and self.dask_client is not None:
            with progress(self.ndwi) as pbar:
                self.ndwi = self.ndwi.compute()
        else:
            self.ndwi = self.ndwi.compute()
        
        # Ensure NDWI is 2D by removing any extra dimensions if they exist
        if 'band' in self.ndwi.dims:
            self.ndwi = self.ndwi.squeeze('band')
        
        # Remove any other extra dimensions that might exist
        expected_dims = ['time', 'y', 'x']  # or ['time', 'lat', 'lon']
        extra_dims = [dim for dim in self.ndwi.dims if dim not in expected_dims]
        for dim in extra_dims:
            self.ndwi = self.ndwi.squeeze(dim)
        
        # Ensure we have exactly 3 dimensions: time + 2 spatial dimensions
        if len(self.ndwi.dims) != 3:
            logger.warning(f"Unexpected NDWI dimensions: {self.ndwi.dims}. Shape: {self.ndwi.shape}")
            # If we have more than 3 dimensions, remove extras
            if len(self.ndwi.dims) > 3:
                spatial_dims = [dim for dim in self.ndwi.dims if dim != 'time']
                if len(spatial_dims) > 2:
                    # Keep only the first two spatial dimensions
                    for dim in spatial_dims[2:]:
                        self.ndwi = self.ndwi.isel({dim: 0})
        
        # Log final NDWI structure
        logger.info(f"✅ Final NDWI structure - Dimensions: {self.ndwi.dims}, Shape: {self.ndwi.shape}")
        
        calculation_time = time.time() - start_time
        logger.info(f"⚡ NDWI calculation completed in {calculation_time:.2f} seconds")
        
        # Count water pixels (NDWI > 0.05 is typically water)
        water_pixels = (self.ndwi > 0.05).sum()
        total_pixels = self.ndwi.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        logger.info(f"💧 Water pixels (NDWI > 0.05): {water_pixels:,} ({water_percentage:.2f}%)")
        
        return self.ndwi
    
    def calculate_ndwi_only_with_tqdm(self) -> xr.DataArray:
        """
        Calculate only NDWI with visual tqdm progress bars.
        
        Returns:
        --------
        NDWI DataArray
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        logger.info("🌊 Calculating NDWI only with visual progress bars...")
        
        with tqdm(total=5, desc="🌊 NDWI Calculation", unit="step") as pbar:
            
            # Step 1: Extract bands
            pbar.set_description("📊 Extracting bands")
            green = self.data.sel(band="green")
            nir = self.data.sel(band="nir")
            pbar.update(1)
            
            # Step 2: Calculate Green - NIR
            pbar.set_description("🧮 Computing Green - NIR")
            green_minus_nir = green - nir
            pbar.update(1)
            
            # Step 3: Calculate Green + NIR
            pbar.set_description("🧮 Computing Green + NIR")
            green_plus_nir = green + nir
            pbar.update(1)
            
            # Step 4: Calculate NDWI
            pbar.set_description("🌊 Computing NDWI division")
            self.ndwi = green_minus_nir / green_plus_nir
            pbar.update(1)
            
            # Step 5: Compute result
            pbar.set_description("⚡ Computing NDWI (parallel)")
            if self.enable_parallel and self.dask_client is not None:
                with progress(self.ndwi) as dask_pbar:
                    self.ndwi = self.ndwi.compute()
            else:
                self.ndwi = self.ndwi.compute()
            pbar.update(1)
        
        logger.info("✅ NDWI calculation completed with visual progress!")
        
        # Count water pixels
        water_pixels = (self.ndwi > 0.05).sum()
        total_pixels = self.ndwi.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        logger.info(f"💧 Water pixels (NDWI > 0.05): {water_pixels:,} ({water_percentage:.2f}%)")
        
        return self.ndwi
    
    def save_all_ndwi_images(self, save_dir: str = "ndwi_images"):
        """
        Save NDWI images for all time steps.
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
                # Remove extra dimensions by selecting first index
                extra_dims = [dim for dim in ndwi_slice.dims if dim not in ['y', 'x', 'lat', 'lon']]
                for dim in extra_dims:
                    ndwi_slice = ndwi_slice.isel({dim: 0})
            
            # Convert to numpy array and ensure it's 2D
            ndwi_array = ndwi_slice.values
            if ndwi_array.ndim > 2:
                # Take the first slice if it's still 3D
                ndwi_array = ndwi_array[0] if ndwi_array.shape[0] == 1 else ndwi_array.squeeze()
            
            # Create filename with date
            date_str = pd.to_datetime(time_val).strftime("%Y%m%d")
            filename = f"ndwi_{date_str}.png"
            save_path = os.path.join(save_dir, filename)
            
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
            water_mask = ndwi_array > 0.05
            im2 = plt.imshow(water_mask, cmap='Blues')
            plt.colorbar(im2, label='Water (1) / Non-water (0)')
            plt.title('Water Mask (NDWI > 0.05)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Count water pixels for this time step
            water_pixels = (ndwi_array > 0.05).sum()
            total_pixels = ndwi_array.size
            water_percentage = (water_pixels / total_pixels) * 100
            
            logger.info(f"✅ Saved {filename} - Water pixels: {water_pixels:,} ({water_percentage:.2f}%)")
        
        logger.info(f"✅ All NDWI images saved to '{save_dir}' directory")
    
    def save_ndwi_data(self, save_path: str = "ndwi_data.nc"):
        """
        Save NDWI data as NetCDF file for further analysis.
        """
        if self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_ndwi_only() first.")
        
        logger.info(f"💾 Saving NDWI data to {save_path}...")
        
        # Add metadata
        self.ndwi.attrs['description'] = 'Normalized Difference Water Index'
        self.ndwi.attrs['formula'] = 'NDWI = (Green - NIR) / (Green + NIR)'
        self.ndwi.attrs['water_threshold'] = 0.05
        self.ndwi.attrs['calculation_date'] = datetime.now().isoformat()
        
        # Try to save as NetCDF first, fallback to other formats if needed
        try:
            self.ndwi.to_netcdf(save_path)
            logger.info(f"✅ NDWI data saved to {save_path} (NetCDF format)")
        except (ImportError, ValueError) as e:
            logger.warning(f"⚠️  NetCDF save failed: {e}")
            logger.info("🔄 Trying alternative formats...")
            
            # Try HDF5 format
            try:
                hdf5_path = save_path.replace('.nc', '.h5')
                self.ndwi.to_netcdf(hdf5_path, engine='h5netcdf')
                logger.info(f"✅ NDWI data saved to {hdf5_path} (HDF5 format)")
                save_path = hdf5_path
            except ImportError:
                logger.warning("⚠️  HDF5 save failed, trying pickle format...")
                
                # Fallback to pickle format
                pickle_path = save_path.replace('.nc', '.pkl')
                import pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.ndwi, f)
                logger.info(f"✅ NDWI data saved to {pickle_path} (Pickle format)")
                save_path = pickle_path
        
        # Print summary
        water_pixels = (self.ndwi > 0.05).sum()
        total_pixels = self.ndwi.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        logger.info(f"📊 Summary:")
        logger.info(f"   Total time steps: {len(self.ndwi.time)}")
        logger.info(f"   Total pixels per image: {total_pixels:,}")
        logger.info(f"   Total water pixels: {water_pixels:,} ({water_percentage:.2f}%)")
    
    def extract_cloud_water_data(self):
        """
        Extract cloud cover and water pixels data for correlation analysis.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with cloud cover and water pixels data
        """
        if self.data is None or self.ndwi is None:
            raise ValueError("Both satellite data and NDWI must be loaded. Run load_data() and calculate_ndwi_only() first.")
        
        logger.info("📊 Extracting cloud cover and water pixels data...")
        
        analysis_data = []
        
        for time_val in self.ndwi.time.values:
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
            
            # Count water pixels using current threshold
            water_mask = ndwi_array > 0.05  # Using current threshold
            water_pixels = water_mask.sum()
            total_pixels = ndwi_array.size
            water_percentage = (water_pixels / total_pixels) * 100
            
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
    
    def calculate_correlation_statistics(self, df):
        """
        Calculate correlation statistics between cloud cover and water pixels.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with cloud cover and water pixels data
            
        Returns:
        --------
        dict
            Dictionary with correlation statistics
        """
        from scipy import stats
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df['cloud_cover'], df['water_pixels'])
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(df['cloud_cover'], df['water_pixels'])
        
        # Linear regression
        X = df['cloud_cover'].values.reshape(-1, 1)
        y = df['water_pixels'].values
        reg = LinearRegression().fit(X, y)
        r_squared = r2_score(y, reg.predict(X))
        
        # Calculate confidence intervals
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept,
            'n_observations': len(df)
        }
    
    def analyze_cloud_water_correlation(self, save_path: str = "cloud_water_correlation.png"):
        """
        Analyze correlation between cloud cover and water pixels.
        
        Parameters:
        -----------
        save_path : str
            Path to save the analysis plots
        """
        logger.info("📊 Analyzing cloud-water correlation...")
        
        # Extract data
        df = self.extract_cloud_water_data()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cloud Cover vs Water Pixels Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Filter out invalid data
        valid_data = df[(df['cloud_cover'] >= 0) & (df['water_pixels'] >= 0)].copy()
        
        if len(valid_data) == 0:
            logger.error("❌ No valid data for correlation analysis")
            return
        
        # Plot 1: Scatter plot - Cloud cover vs Water pixels
        ax1 = axes[0, 0]
        ax1.scatter(valid_data['cloud_cover'], valid_data['water_pixels'], 
                   alpha=0.6, color='blue', s=50)
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['cloud_cover'], valid_data['water_pixels'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data['cloud_cover'], p(valid_data['cloud_cover']), 
                    "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Cloud Cover (%)')
        ax1.set_ylabel('Water Pixels')
        ax1.set_title('Cloud Cover vs Water Pixels')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot - Cloud cover vs Water percentage
        ax2 = axes[0, 1]
        ax2.scatter(valid_data['cloud_cover'], valid_data['water_percentage'], 
                   alpha=0.6, color='green', s=50)
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['cloud_cover'], valid_data['water_percentage'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data['cloud_cover'], p(valid_data['cloud_cover']), 
                    "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Cloud Cover (%)')
        ax2.set_ylabel('Water Percentage (%)')
        ax2.set_title('Cloud Cover vs Water Percentage')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correlation heatmap
        ax3 = axes[0, 2]
        correlation_matrix = valid_data[['cloud_cover', 'water_pixels', 'water_percentage']].corr()
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, ax=ax3, cbar_kws={'label': 'Correlation Coefficient'})
        ax3.set_title('Correlation Matrix')
        
        # Plot 4: Cloud cover distribution
        ax4 = axes[1, 0]
        ax4.hist(valid_data['cloud_cover'], bins=20, alpha=0.7, color='lightblue', 
                 edgecolor='black', density=True)
        ax4.set_xlabel('Cloud Cover (%)')
        ax4.set_ylabel('Density')
        ax4.set_title('Cloud Cover Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Water pixels by cloud cover categories
        ax5 = axes[1, 1]
        # Create cloud cover categories
        valid_data['cloud_category'] = pd.cut(valid_data['cloud_cover'], 
                                             bins=[0, 10, 25, 50, 100], 
                                             labels=['Clear (0-10%)', 'Low (10-25%)', 'Medium (25-50%)', 'High (50-100%)'])
        
        cloud_water_stats = valid_data.groupby('cloud_category')['water_pixels'].agg(['mean', 'std', 'count']).dropna()
        
        if len(cloud_water_stats) > 0:
            x_pos = range(len(cloud_water_stats))
            ax5.bar(x_pos, cloud_water_stats['mean'], yerr=cloud_water_stats['std'], 
                   capsize=5, alpha=0.7, color='skyblue')
            ax5.set_xlabel('Cloud Cover Category')
            ax5.set_ylabel('Average Water Pixels')
            ax5.set_title('Water Pixels by Cloud Cover Category')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(cloud_water_stats.index, rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Time series of cloud cover and water pixels
        ax6 = axes[1, 2]
        ax6_twin = ax6.twinx()
        
        # Sort by date
        valid_data_sorted = valid_data.sort_values('date')
        
        # Plot cloud cover on primary y-axis
        line1 = ax6.plot(valid_data_sorted['date'], valid_data_sorted['cloud_cover'], 
                         'o-', color='gray', linewidth=2, markersize=4, label='Cloud Cover')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Cloud Cover (%)', color='gray')
        ax6.tick_params(axis='y', labelcolor='gray')
        ax6.grid(True, alpha=0.3)
        
        # Plot water pixels on secondary y-axis
        line2 = ax6_twin.plot(valid_data_sorted['date'], valid_data_sorted['water_pixels'], 
                              's-', color='blue', linewidth=2, markersize=4, label='Water Pixels')
        ax6_twin.set_ylabel('Water Pixels', color='blue')
        ax6_twin.tick_params(axis='y', labelcolor='blue')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        ax6.set_title('Cloud Cover and Water Pixels Over Time')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate correlation statistics
        correlation_stats = self.calculate_correlation_statistics(valid_data)
        
        # Print summary
        logger.info("📊 Cloud-Water Correlation Analysis Summary:")
        logger.info(f"   Total observations: {len(valid_data)}")
        logger.info(f"   Cloud cover range: {valid_data['cloud_cover'].min():.1f}% - {valid_data['cloud_cover'].max():.1f}%")
        logger.info(f"   Water pixels range: {valid_data['water_pixels'].min():.0f} - {valid_data['water_pixels'].max():.0f}")
        logger.info(f"   Correlation coefficient (cloud vs water pixels): {correlation_stats['pearson_r']:.3f}")
        logger.info(f"   P-value: {correlation_stats['pearson_p']:.3f}")
        logger.info(f"   R-squared: {correlation_stats['r_squared']:.3f}")
        
        logger.info(f"✅ Cloud-water correlation analysis saved to {save_path}")
        
        return correlation_stats
    
    def print_computation_progress(self):
        """
        Print current computation progress and status.
        """
        status = self.get_computation_status()
        
        logger.info("📊 Computation Status:")
        logger.info(f"   🚀 Parallel processing: {'✅ Enabled' if status['parallel_enabled'] else '❌ Disabled'}")
        logger.info(f"   🔧 Dask client: {'✅ Available' if status['dask_client_available'] else '❌ Not available'}")
        logger.info(f"   👥 Workers: {status['n_workers']}")
        
        if status['dask_client_available']:
            logger.info(f"   🖥️  Active workers: {status.get('active_workers', 'Unknown')}")
            logger.info(f"   💾 Total memory: {status.get('total_memory', 0) / (1024**3):.2f} GB")
            logger.info(f"   💾 Available memory: {status.get('available_memory', 0) / (1024**3):.2f} GB")
            logger.info(f"   ⚡ CPU usage: {status.get('cpu_percent', 0):.1f}%")
            
            if 'total_tasks' in status:
                logger.info(f"   📋 Total tasks: {status['total_tasks']}")
                logger.info(f"   🔄 Running tasks: {status['running_tasks']}")
                logger.info(f"   ⏳ Waiting tasks: {status['waiting_tasks']}")
                logger.info(f"   ✅ Completed tasks: {status['completed_tasks']}")
                
                if status['total_tasks'] > 0:
                    completion_rate = (status['completed_tasks'] / status['total_tasks']) * 100
                    logger.info(f"   📈 Completion rate: {completion_rate:.1f}%")
    
    def monitor_computation_progress(self, interval: float = 5.0, max_updates: int = 10):
        """
        Monitor computation progress at regular intervals.
        
        Parameters:
        -----------
        interval : float
            Time interval between progress updates in seconds
        max_updates : int
            Maximum number of progress updates to show
        """
        logger.info(f"📊 Starting progress monitoring (updates every {interval}s, max {max_updates} updates)")
        
        update_count = 0
        start_time = time.time()
        
        while update_count < max_updates:
            time.sleep(interval)
            update_count += 1
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n⏱️  Progress Update #{update_count} (elapsed: {elapsed_time:.1f}s):")
            self.print_computation_progress()
            
            # Check if computation is complete
            status = self.get_computation_status()
            if status.get('total_tasks', 0) > 0:
                completion_rate = (status.get('completed_tasks', 0) / status['total_tasks']) * 100
                if completion_rate >= 100:
                    logger.info("🎉 Computation appears to be complete!")
                    break
    
    def start_progress_monitoring(self, interval: float = 5.0, max_updates: int = 10):
        """
        Start progress monitoring in a separate thread.
        
        Parameters:
        -----------
        interval : float
            Time interval between progress updates in seconds
        max_updates : int
            Maximum number of progress updates to show
            
        Returns:
        --------
        Thread object for the monitoring process
        """
        import threading
        
        def monitor_thread():
            self.monitor_computation_progress(interval, max_updates)
        
        monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
        monitor_thread.start()
        
        logger.info(f"📊 Progress monitoring started in background thread")
        return monitor_thread
    
    def calculate_water_indices_with_tqdm(self) -> Dict[str, xr.DataArray]:
        """
        Calculate water indices with tqdm progress bars for visual progress tracking.
        
        Returns:
        --------
        Dictionary containing NDWI array
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        logger.info("🌊 Calculating water indices with visual progress bars...")
        
        # Create progress bar for overall process
        with tqdm(total=4, desc="🌊 Water Indices Calculation", unit="step") as pbar:
            
            # Step 1: Extract bands
            pbar.set_description("📊 Extracting bands")
            green = self.data.sel(band="green")
            nir = self.data.sel(band="nir")
            pbar.update(1)
            
            # Step 2: Calculate NDWI components
            pbar.set_description("🧮 Calculating NDWI components")
            green_minus_nir = green - nir
            green_plus_nir = green + nir
            pbar.update(1)
            
            # Step 3: Calculate NDWI
            pbar.set_description("🌊 Computing NDWI")
            self.ndwi = green_minus_nir / green_plus_nir
            pbar.update(1)
            
            # Step 4: Compute NDWI
            pbar.set_description("⚡ Computing NDWI (parallel)")
            if self.enable_parallel and self.dask_client is not None:
                with progress(self.ndwi) as dask_pbar:
                    self.ndwi = self.ndwi.compute()
            else:
                self.ndwi = self.ndwi.compute()
            pbar.update(1)
        
        logger.info("✅ Water indices calculation completed with visual progress!")
        
        # Log statistics
        logger.info("📊 Water indices statistics:")
        logger.info(f"   NDWI range: {float(self.ndwi.min()):.3f} to {float(self.ndwi.max()):.3f}")
        logger.info(f"   NDWI mean: {float(self.ndwi.mean()):.3f}")
        
        result = {"ndwi": self.ndwi}
        return result
    

    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the loaded data.
        
        Returns:
        --------
        Dictionary with data statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        stats = {
            "shape": self.data.shape,
            "dimensions": dict(self.data.sizes),
            "bands": list(self.data.band.values),
            "time_range": {
                "start": str(self.data.time.min().values),
                "end": str(self.data.time.max().values),
                "total_dates": len(self.data.time)
            },
            "spatial_extent": {
                "x_min": float(self.data.x.min()),
                "x_max": float(self.data.x.max()),
                "y_min": float(self.data.y.min()),
                "y_max": float(self.data.y.max())
            },
            "crs": str(self.data.rio.crs) if hasattr(self.data, 'rio') else "Unknown"
        }
        
        return stats
    
    def analyze_water_extent(self, 
                           water_index: str = "ndwi",
                           threshold: float = 0.05,
                           mask_geometry: Optional[shapely.geometry.Polygon] = None) -> pd.DataFrame:
        """
        Analyze water extent over time.
        
        Parameters:
        -----------
        water_index : str
            Which water index to use ("ndwi")
        threshold : float
            Threshold for water classification
        mask_geometry : Optional[shapely.geometry.Polygon]
            Optional geometry to mask the analysis area
            
        Returns:
        --------
        DataFrame with temporal water extent data
        """
        if water_index == "ndwi" and self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_water_indices() first.")
        
        water_data = self.ndwi
        
        logger.info(f"🌊 Analyzing water extent using {water_index.upper()} with threshold {threshold}")
        logger.info(f"📊 Data shape: {water_data.shape}")
        logger.info(f"📅 Time steps to process: {len(water_data.time)}")
        
        # Apply mask if provided
        if mask_geometry is not None:
            logger.info("🎭 Applying geometry mask...")
            water_data = self._apply_geometry_mask(water_data, mask_geometry)
        
        # Calculate water pixels for each time step using parallel processing
        logger.info("🔄 Processing time steps with parallel optimization...")
        start_time = time.time()
        
        # Use sequential processing since distributed is not available
        logger.info("🔄 Using sequential processing...")
        results = self._analyze_water_extent_sequential(water_data, threshold)
        
        analysis_time = time.time() - start_time
        logger.info(f"⚡ Water extent analysis completed in {analysis_time:.2f} seconds")
        
        return pd.DataFrame(results)
    

    
    def _analyze_water_extent_sequential(self, water_data, threshold):
        """
        Analyze water extent using sequential processing (fallback).
        """
        logger.info("🔄 Using sequential processing...")
        results = []
        
        for i, t in enumerate(water_data.time.values):
            try:
                # Log progress every 5 time steps or at the end
                if i % 5 == 0 or i == len(water_data.time.values) - 1:
                    progress = (i + 1) / len(water_data.time.values) * 100
                    logger.info(f"📈 Progress: {progress:.1f}% ({i+1}/{len(water_data.time.values)})")
                
                # Select time slice and compute
                water_slice = water_data.sel(time=t)
                if water_slice is None:
                    logger.warning(f"⚠️  Empty data for time {t}")
                    continue
                water_slice = water_slice.compute()
                
                # Create water mask
                water_mask = water_slice > threshold
                
                # Count water pixels
                water_pixels = water_mask.sum().item()
                total_pixels = (~np.isnan(water_slice)).sum().item()
                
                # Calculate percentage
                water_percentage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0
                
                results.append({
                    'date': pd.to_datetime(t),
                    'water_pixels': water_pixels,
                    'total_pixels': total_pixels,
                    'water_percentage': water_percentage,
                    'water_index_mean': float(water_slice.mean()),
                    'water_index_std': float(water_slice.std())
                })
                
            except Exception as e:
                logger.warning(f"Error processing date {t}: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"Analysis complete. Processed {len(df)} time steps")
        
        return df
    
    def _apply_geometry_mask(self, data: xr.DataArray, geometry: shapely.geometry.Polygon) -> xr.DataArray:
        """
        Apply a geometry mask to the data.
        
        Parameters:
        -----------
        data : xarray.DataArray
            Data to mask
        geometry : shapely.geometry.Polygon
            Geometry to use as mask
            
        Returns:
        --------
        Masked xarray.DataArray
        """
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")
        
        # Ensure data has CRS
        if data.rio.crs is None:
            data.rio.write_crs("EPSG:4326", inplace=True)
        
        # Reproject geometry to data CRS
        gdf_proj = gdf.to_crs(data.rio.crs)
        
        # Create mask
        mask = geometry_mask(
            geometries=gdf_proj.geometry,
            out_shape=(data.sizes['y'], data.sizes['x']),
            transform=data.rio.transform(),
            invert=True
        )
        
        # Apply mask
        return data.where(mask)
    
    def plot_temporal_analysis(self, 
                             df: pd.DataFrame,
                             save_path: Optional[str] = "images/temporal_analysis.png") -> None:
        """
        Plot temporal analysis results.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame from analyze_water_extent()
        save_path : Optional[str]
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Water pixels over time
        ax1.plot(df['date'], df['water_pixels'], marker='o', linestyle='-', linewidth=2, markersize=4)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Water Pixels')
        ax1.set_title('Water Extent Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Water percentage over time
        ax2.plot(df['date'], df['water_percentage'], marker='s', linestyle='-', 
                linewidth=2, markersize=4, color='orange')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Water Percentage (%)')
        ax2.set_title('Water Percentage Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_water_index_map(self, 
                           water_index: str = "ndwi",
                           time_index: int = 0,
                           save_path: Optional[str] = "images/water_index_map.png") -> None:
        """
        Plot a water index map for a specific time.
        
        Parameters:
        -----------
        water_index : str
            Which water index to plot ("ndwi")
        time_index : int
            Time index to plot
        save_path : Optional[str]
            Path to save the plot
        """
        if water_index == "ndwi" and self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_water_indices() first.")
        
        water_data = self.ndwi
        
        # Select time slice
        data_slice = water_data.isel(time=time_index)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = data_slice.plot(
            ax=ax,
            cmap='BrBG',
            vmin=-1,
            vmax=1,
            add_colorbar=True,
            cbar_kwargs={'label': f'{water_index.upper()} Value'}
        )
        
        ax.set_title(f'{water_index.upper()} - {pd.to_datetime(data_slice.time.values).strftime("%Y-%m-%d")}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Map saved to {save_path}")
        
        plt.show()
    
    def create_interactive_map(self, bbox: List[float]) -> folium.Map:
        """
        Create an interactive map showing the study area.
        
        Parameters:
        -----------
        bbox : List[float]
            Bounding box [lon_min, lat_min, lon_max, lat_max]
            
        Returns:
        --------
        folium.Map object
        """
        # Calculate center
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="CartoDB Positron"
        )
        
        # Add bounding box
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="blue",
            fill=True,
            fill_opacity=0.2,
            tooltip="Study Area"
        ).add_to(m)
        
        return m

    def create_data_cube_visualizations(self, save_dir: str = "images") -> None:
        """
        Create comprehensive visualizations of the data cube.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save the visualizations
        """
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        logger.info(f"Creating comprehensive data cube visualizations in {save_dir}")
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Data cube overview
        logger.info("Creating data cube overview visualization...")
        self._plot_data_cube_overview(save_dir)
        
        # 2. Band statistics
        logger.info("Creating band statistics visualization...")
        self._plot_band_statistics(save_dir)
        
        # 3. Time series overview
        logger.info("Creating time series overview...")
        self._plot_time_series_overview(save_dir)
        
        # 4. Spatial coverage
        logger.info("Creating spatial coverage visualization...")
        self._plot_spatial_coverage(save_dir)
        
        # 5. Water indices visualization
        if self.ndwi is not None:
            logger.info("Creating water indices visualization...")
            self._plot_ndwi_only(save_dir)
        
        logger.info(f"All visualizations saved to {save_dir}/")

    def _plot_data_cube_overview(self, save_dir: str) -> None:
        """Create an overview of the data cube structure."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Cube Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Data shape and dimensions
        ax1 = axes[0, 0]
        dims = list(self.data.sizes.items())
        dim_names = [d[0] for d in dims]
        dim_values = [d[1] for d in dims]
        
        bars = ax1.bar(dim_names, dim_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Data Cube Dimensions')
        ax1.set_ylabel('Size')
        ax1.set_xlabel('Dimension')
        
        # Add value labels on bars
        for bar, value in zip(bars, dim_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(dim_values),
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Time coverage
        ax2 = axes[0, 1]
        dates = pd.to_datetime(self.data.time.values)
        ax2.plot(dates, range(len(dates)), 'o-', linewidth=2, markersize=4)
        ax2.set_title('Temporal Coverage')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Time Index')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Band information
        ax3 = axes[1, 0]
        bands = list(self.data.band.values)
        band_colors = ['green', 'red']
        ax3.bar(bands, [1]*len(bands), color=band_colors[:len(bands)])
        ax3.set_title('Available Bands')
        ax3.set_ylabel('Available')
        ax3.set_ylim(0, 1.2)
        
        # Add band descriptions
        band_descriptions = {
            'green': 'Green (B03)',
            'nir': 'Near-Infrared (B08)'
        }
        for i, band in enumerate(bands):
            desc = band_descriptions.get(band, band)
            ax3.text(i, 0.6, desc, ha='center', va='center', fontweight='bold')
        
        # Plot 4: Data statistics
        ax4 = axes[1, 1]
        stats_data = []
        stats_labels = []
        
        for band in bands:
            band_data = self.data.sel(band=band)
            stats_data.extend([
                float(band_data.min()),
                float(band_data.max()),
                float(band_data.mean()),
                float(band_data.std())
            ])
            stats_labels.extend([f'{band}_min', f'{band}_max', f'{band}_mean', f'{band}_std'])
        
        ax4.bar(range(len(stats_data)), stats_data, alpha=0.7)
        ax4.set_title('Band Statistics')
        ax4.set_ylabel('Value')
        ax4.set_xticks(range(len(stats_labels)))
        ax4.set_xticklabels(stats_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/01_data_cube_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Data cube overview saved")

    def _plot_band_statistics(self, save_dir: str) -> None:
        """Create detailed band statistics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Band Statistics', fontsize=16, fontweight='bold')
        
        bands = list(self.data.band.values)
        colors = ['green', 'red']
        
        for i, band in enumerate(bands):
            band_data = self.data.sel(band=band)
            
            # Histogram
            ax = axes[i//2, i%2]
            band_data.plot.hist(ax=ax, bins=50, alpha=0.7, color=colors[i])
            ax.set_title(f'{band.upper()} Band Distribution')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = float(band_data.mean())
            std_val = float(band_data.std())
            min_val = float(band_data.min())
            max_val = float(band_data.max())
            
            stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/02_band_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Band statistics saved")

    def _plot_time_series_overview(self, save_dir: str) -> None:
        """Create time series overview visualization."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Time Series Overview', fontsize=16, fontweight='bold')
        
        bands = list(self.data.band.values)
        colors = ['green', 'red']
        
        for i, band in enumerate(bands):
            band_data = self.data.sel(band=band)
            
            # Calculate statistics over time
            time_mean = band_data.mean(dim=['x', 'y'])
            time_std = band_data.std(dim=['x', 'y'])
            
            dates = pd.to_datetime(time_mean.time.values)
            
            ax = axes[i]
            ax.plot(dates, time_mean, 'o-', color=colors[i], linewidth=2, markersize=4, label='Mean')
            ax.fill_between(dates, time_mean - time_std, time_mean + time_std, 
                          alpha=0.3, color=colors[i], label='±1 Std')
            
            ax.set_title(f'{band.upper()} Band - Temporal Statistics')
            ax.set_xlabel('Date')
            ax.set_ylabel('Pixel Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/03_time_series_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Time series overview saved")

    def _plot_spatial_coverage(self, save_dir: str) -> None:
        """Create spatial coverage visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spatial Coverage Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: First time step RGB-like composite
        ax1 = axes[0, 0]
        first_slice = self.data.isel(time=0)
        
        # Create RGB composite (normalize each band)
        rgb_data = np.zeros((first_slice.sizes['y'], first_slice.sizes['x'], 3))
        for i, band in enumerate(['green', 'nir']):
            if band in first_slice.band.values:
                band_data = first_slice.sel(band=band).values
                # Normalize to 0-1
                band_norm = (band_data - band_data.min()) / (band_data.max() - band_data.min())
                rgb_data[:, :, i] = band_norm
        
        ax1.imshow(rgb_data)
        ax1.set_title('RGB Composite (First Time Step)')
        ax1.set_xlabel('X Pixels')
        ax1.set_ylabel('Y Pixels')
        ax1.axis('off')
        
        # Plot 2: Spatial extent
        ax2 = axes[0, 1]
        x_coords = self.data.x.values
        y_coords = self.data.y.values
        
        ax2.plot(x_coords, [y_coords[0]]*len(x_coords), 'b-', linewidth=2, label='Top')
        ax2.plot(x_coords, [y_coords[-1]]*len(x_coords), 'r-', linewidth=2, label='Bottom')
        ax2.plot([x_coords[0]]*len(y_coords), y_coords, 'g-', linewidth=2, label='Left')
        ax2.plot([x_coords[-1]]*len(y_coords), y_coords, 'orange', linewidth=2, label='Right')
        
        ax2.set_title('Spatial Extent')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coverage area
        ax3 = axes[1, 0]
        area_km2 = (x_coords[-1] - x_coords[0]) * (y_coords[-1] - y_coords[0]) / 1e6
        ax3.text(0.5, 0.5, f'Coverage Area:\n{area_km2:.2f} km²', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax3.transAxes)
        ax3.set_title('Study Area Coverage')
        ax3.axis('off')
        
        # Plot 4: Resolution information
        ax4 = axes[1, 1]
        x_res = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
        y_res = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)
        
        info_text = f'Resolution:\nX: {x_res:.1f} m\nY: {y_res:.1f} m\n\nPixel Count:\nX: {len(x_coords)}\nY: {len(y_coords)}'
        ax4.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Spatial Resolution')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/04_spatial_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Spatial coverage saved")



    def _plot_ndwi_only(self, save_dir: str) -> None:
        """Create NDWI-only visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NDWI Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: NDWI histogram
        ax1 = axes[0, 0]
        if self.ndwi is not None:
            ndwi_values = self.ndwi.values.flatten()
            ax1.hist(ndwi_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Water Threshold (0.05)')
            ax1.set_title('NDWI Distribution')
            ax1.set_xlabel('NDWI Value')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No NDWI data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('NDWI Distribution - No Data')
        
        # Plot 2: NDWI statistics over time
        ax2 = axes[0, 1]
        if self.ndwi is not None:
            time_mean = self.ndwi.mean(dim=['x', 'y'])
            time_std = self.ndwi.std(dim=['x', 'y'])
            dates = pd.to_datetime(time_mean.time.values)
            
            ax2.plot(dates, time_mean, 'o-', color='blue', linewidth=2, markersize=4, label='Mean')
            ax2.fill_between(dates, time_mean - time_std, time_mean + time_std, 
                            alpha=0.3, color='blue', label='±1 Std')
            ax2.set_title('NDWI Temporal Statistics')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('NDWI Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No NDWI data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('NDWI Temporal Statistics - No Data')
        
        # Plot 3: NDWI map (first time step)
        ax3 = axes[1, 0]
        if self.ndwi is not None:
            ndwi_slice = self.ndwi.isel(time=0)
            im3 = ndwi_slice.plot(ax=ax3, cmap='BrBG', vmin=-1, vmax=1, add_colorbar=True)
            ax3.set_title(f'NDWI - {pd.to_datetime(ndwi_slice.time.values).strftime("%Y-%m-%d")}')
        else:
            ax3.text(0.5, 0.5, 'No NDWI data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('NDWI Map - No Data')
        
        # Plot 4: Water pixels over time
        ax4 = axes[1, 1]
        if self.ndwi is not None:
            water_pixels = []
            dates = []
            
            for t in self.ndwi.time.values:
                try:
                    ndwi_t = self.ndwi.sel(time=t).compute()
                    water_mask = ndwi_t > 0.05
                    count = water_mask.sum().item()
                    water_pixels.append(count)
                    dates.append(pd.to_datetime(t))
                except:
                    continue
            
            ax4.plot(dates, water_pixels, 'o-', color='green', linewidth=2, markersize=4)
            ax4.set_title('Water Pixels Over Time (NDWI > 0.05)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Water Pixels')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No NDWI data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Water Pixels Over Time - No Data')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/05_ndwi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("NDWI analysis saved")
    
    def plot_water_pixels_by_year(self, save_path: str = "water_pixels_by_year.png"):
        """
        Create a comprehensive analysis of water pixels by year.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_ndwi_only() first.")
        
        logger.info("📊 Creating water pixels by year analysis...")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Water Pixels Analysis by Year (2014-2024)', fontsize=16, fontweight='bold')
        
        # Collect data by year
        yearly_data = {}
        monthly_data = {}
        
        for time_val in self.ndwi.time.values:
            date = pd.to_datetime(time_val)
            year = date.year
            month = date.month
            
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
            
            # Count water pixels
            water_mask = ndwi_array > 0.05
            water_pixels = water_mask.sum()
            total_pixels = ndwi_array.size
            water_percentage = (water_pixels / total_pixels) * 100
            
            # Store data
            if year not in yearly_data:
                yearly_data[year] = {'pixels': [], 'percentages': [], 'dates': []}
            yearly_data[year]['pixels'].append(water_pixels)
            yearly_data[year]['percentages'].append(water_percentage)
            yearly_data[year]['dates'].append(date)
            
            # Monthly data
            month_key = (year, month)
            if month_key not in monthly_data:
                monthly_data[month_key] = {'pixels': [], 'percentages': []}
            monthly_data[month_key]['pixels'].append(water_pixels)
            monthly_data[month_key]['percentages'].append(water_percentage)
        
        # Plot 1: Water pixels by year (box plot)
        ax1 = axes[0, 0]
        years = sorted(yearly_data.keys())
        water_pixels_by_year = [yearly_data[year]['pixels'] for year in years]
        
        bp1 = ax1.boxplot(water_pixels_by_year, labels=years, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax1.set_title('Water Pixels Distribution by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Water Pixels')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Water percentage by year (box plot)
        ax2 = axes[0, 1]
        water_percentages_by_year = [yearly_data[year]['percentages'] for year in years]
        
        bp2 = ax2.boxplot(water_percentages_by_year, labels=years, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        
        ax2.set_title('Water Percentage Distribution by Year')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Water Percentage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Monthly averages
        ax3 = axes[1, 0]
        months = range(1, 13)
        monthly_avg_pixels = []
        monthly_avg_percentages = []
        
        for month in months:
            month_pixels = []
            month_percentages = []
            for year in years:
                month_key = (year, month)
                if month_key in monthly_data:
                    month_pixels.extend(monthly_data[month_key]['pixels'])
                    month_percentages.extend(monthly_data[month_key]['percentages'])
            
            if month_pixels:
                monthly_avg_pixels.append(np.mean(month_pixels))
                monthly_avg_percentages.append(np.mean(month_percentages))
            else:
                monthly_avg_pixels.append(0)
                monthly_avg_percentages.append(0)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax3.bar(month_names, monthly_avg_pixels, color='skyblue', alpha=0.8)
        ax3.set_title('Average Water Pixels by Month')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Water Pixels')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Time series trend
        ax4 = axes[1, 1]
        all_dates = []
        all_pixels = []
        all_percentages = []
        
        for year in years:
            all_dates.extend(yearly_data[year]['dates'])
            all_pixels.extend(yearly_data[year]['pixels'])
            all_percentages.extend(yearly_data[year]['percentages'])
        
        # Sort by date
        sorted_data = sorted(zip(all_dates, all_pixels, all_percentages))
        dates, pixels, percentages = zip(*sorted_data)
        
        # Plot time series
        ax4.plot(dates, pixels, 'o-', color='blue', linewidth=2, markersize=4, label='Water Pixels')
        ax4.set_title('Water Pixels Time Series')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Water Pixels')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        # Add trend line
        if len(pixels) > 1:
            z = np.polyfit(range(len(pixels)), pixels, 1)
            p = np.poly1d(z)
            ax4.plot(dates, p(range(len(pixels))), "r--", alpha=0.8, label='Trend')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        logger.info("📊 Water Pixels by Year Summary:")
        for year in years:
            year_pixels = yearly_data[year]['pixels']
            year_percentages = yearly_data[year]['percentages']
            logger.info(f"   {year}: {len(year_pixels)} observations")
            logger.info(f"      Avg pixels: {np.mean(year_pixels):.0f} ± {np.std(year_pixels):.0f}")
            logger.info(f"      Avg percentage: {np.mean(year_percentages):.2f}% ± {np.std(year_percentages):.2f}%")
        
        logger.info(f"✅ Water pixels by year analysis saved to {save_path}")
        
        return yearly_data


def main():
    """
    Main function demonstrating the WaterAnalyzer usage.
    """
    start_time = time.time()
    logger.info("🚀 Starting Water Analysis Framework")
    logger.info("=" * 50)
    
    # Configuration for 10-year analysis (2014-2024)
    bbox = [-73.705, 4.605, -73.700, 4.610]  # Very small area for testing
    start_date = "2014-01-01"  # Start of 10-year analysis window
    end_date = "2024-12-31"  # End of 10-year analysis window
    
    logger.info("📅 10-year analysis: 2014-2024")
    logger.info("📊 This will provide comprehensive water dynamics analysis")
    
    logger.info("⚙️  Configuration:")
    logger.info(f"   📍 Study area: {bbox}")
    logger.info(f"   📅 Time period: {start_date} to {end_date}")
    logger.info(f"   🛰️  Collection: sentinel-2-l2a")
    logger.info(f"   🎨 Bands: B03 (Green), B08 (NIR), B11 (SWIR)")
    logger.info(f"   ☁️  Cloud cover: < 20%")
    logger.info(f"   📊 Max items: 200")
    
    # Initialize analyzer with optimized settings (2 bands for NDWI)
    logger.info("🔧 Initializing WaterAnalyzer with optimized settings...")
    analyzer = WaterAnalyzer(
        collection="sentinel-2-l2a",
        assets=["B03", "B08", "SCL"],  # Green, NIR, and Scene Classification for cloud masking
        band_aliases={"green": "B03", "nir": "B08", "scl": "SCL"},  # Band name mappings
        chunksize=256,  # Much smaller chunks to match tile size
        resolution=500,  # Much lower resolution to reduce data size significantly
        epsg=32618,  # UTM Zone 18N for Colombia region
        enable_parallel=False,  # Disable parallel processing for testing
        n_workers=2  # Reduced workers to avoid too many concurrent connections
    )
    
    try:
        # Step 1: Search for data
        logger.info("\n" + "="*50)
        logger.info("📡 STEP 1: SEARCHING FOR SATELLITE DATA")
        logger.info("="*50)
        items = analyzer.search_data(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=10.0,  # Allow up to 20% cloud cover
            max_items=5  # Minimal number of images for testing
        )
        
        # Step 2: Load data
        logger.info("\n" + "="*50)
        logger.info("📥 STEP 2: LOADING AND STACKING DATA")
        logger.info("="*50)
        data = analyzer.load_data()
        
        # Step 3: Get statistics
        logger.info("\n" + "="*50)
        logger.info("📊 STEP 3: DATA STATISTICS")
        logger.info("="*50)
        stats = analyzer.get_data_statistics()
        print("\n📈 Data Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Step 4: Calculate water indices with progress tracking
        logger.info("\n" + "="*50)
        logger.info("🌊 STEP 4: CALCULATING WATER INDICES")
        logger.info("="*50)
        
        # Show initial computation status
        logger.info("📊 Initial computation status:")
        analyzer.print_computation_progress()
        
        # Calculate only NDWI with detailed progress tracking
        logger.info("🔧 Using detailed step-by-step progress tracking for NDWI...")
        ndwi = analyzer.calculate_ndwi_only(show_progress=True)
        
        # Alternative: Use visual progress bars (uncomment to try)
        # logger.info("🔧 Using visual progress bars...")
        # ndwi = analyzer.calculate_ndwi_only_with_tqdm()
        
        # Step 5: Save all NDWI images
        logger.info("\n" + "="*50)
        logger.info("📊 STEP 5: SAVING ALL NDWI IMAGES")
        logger.info("="*50)
        analyzer.save_all_ndwi_images(save_dir="ndwi_images")
        
        # Step 6: Save NDWI data
        logger.info("\n" + "="*50)
        logger.info("💾 STEP 6: SAVING NDWI DATA")
        logger.info("="*50)
        analyzer.save_ndwi_data(save_path="ndwi_data.nc")
        
        # Step 7: Create water pixels by year analysis
        logger.info("\n" + "="*50)
        logger.info("📊 STEP 7: ANALYZING WATER PIXELS BY YEAR")
        logger.info("="*50)
        analyzer.plot_water_pixels_by_year(save_path="water_pixels_by_year.png")
        
        # Step 8: Cloud-water correlation analysis
        logger.info("\n" + "="*50)
        logger.info("☁️ STEP 8: CLOUD-WATER CORRELATION ANALYSIS")
        logger.info("="*50)
        correlation_stats = analyzer.analyze_cloud_water_correlation(save_path="cloud_water_correlation.png")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("🎉 NDWI ANALYSIS COMPLETED!")
        logger.info("="*50)
        # Performance summary
        end_time = time.time()
        print_performance_summary(start_time, end_time, data.shape)
        
        logger.info("📁 Generated files:")
        logger.info("   📁 ndwi_images/ (all NDWI images)")
        logger.info("   📄 ndwi_data.nc (NDWI data)")
        logger.info("   📊 water_pixels_by_year.png (yearly analysis)")
        logger.info("   📊 cloud_water_correlation.png (correlation analysis)")
        logger.info("   📄 water_analysis.log")
        logger.info("📊 Water pixel counts shown for each image")
        
        return analyzer, None
        
    except Exception as e:
        logger.error(f"❌ Error in main analysis: {e}")
        logger.error("🔍 Check the log file for detailed error information")
        raise
    
    finally:
        # Clean up Dask client
        if 'analyzer' in locals():
            analyzer.stop_dask_client()
            logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    # Run the analysis
    print("🚀 Starting Water Indices Analysis Framework...")
    print("📝 Detailed logs will be saved to 'water_analysis.log'")
    print("🖼️  Images will be saved to 'images/' folder")
    print("=" * 60)
    
    try:
        analyzer, results = main()
        
        print("\n" + "=" * 60)
        print("🎉 NDWI ANALYSIS COMPLETED!")
        print("=" * 60)
        print("📁 Generated files:")
        print("   📁 ndwi_images/ - NDWI maps for all images")
        print("   📄 ndwi_data.nc - NDWI data")
        print("   📊 water_pixels_by_year.png - Yearly analysis")
        print("   📄 water_analysis.log - Detailed execution log")
        print("\n💡 NDWI values and water pixel counts shown above!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("🔍 Check the log file for detailed error information")
        raise 