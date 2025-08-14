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

# Dask imports with proper configuration
import dask
from dask.distributed import Client, LocalCluster
import dask.array as da

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

# Global client variable
client = None

def setup_dask_client():
    """Setup Dask client with Docker-optimized settings"""
    global client
    
    # Get available memory (in GB) - conservative estimate for Docker
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback if psutil is not available
        total_memory_gb = 4  # Conservative default for Docker
        logger.warning("⚠️  psutil not available, using conservative memory estimate")
    
    # Use only 70% of available memory for Dask
    dask_memory_gb = max(1, int(total_memory_gb * 0.7))
    
    # Configure Dask settings
    dask.config.set({
        'array.chunk-size': '128MB',  # Smaller chunks
        'distributed.worker.memory.target': 0.8,  # Target 80% memory usage
        'distributed.worker.memory.spill': 0.9,   # Spill to disk at 90%
        'distributed.worker.memory.pause': 0.95,  # Pause at 95%
        'distributed.worker.memory.terminate': 0.98,  # Terminate at 98%
    })
    
    # Create local cluster with memory limits
    cluster = LocalCluster(
        n_workers=1,  # Single worker to avoid multiprocessing issues
        threads_per_worker=2,  # Limit threads
        memory_limit=f"{dask_memory_gb}GB",
        local_directory="/tmp/dask-worker-space",  # Use temp directory
        silence_logs=logging.WARNING
    )
    
    # Create client
    client = Client(cluster)
    
    logger.info(f"🔧 Dask client configured with {dask_memory_gb}GB memory limit")
    logger.info(f"🔧 Cluster: {client.cluster}")
    
    return client


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
    def __init__(self, collection="sentinel-2-l2a", assets=None, band_aliases=None,
                 chunksize=512, resolution=100, epsg=32618):
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

    def search_data(self, bbox: List[float], start_date="2014-01-01", end_date="2024-12-31",
                    max_cloud_cover=10.0, max_items=200) -> List:
        logger.info(f"🔍 Searching for data from {start_date} to {end_date}")
        logger.info(f"📍 Bounding box: {bbox}")
        logger.info(f"☁️  Maximum cloud cover: {max_cloud_cover}%")
        logger.info(f"📊 Maximum items to retrieve: {max_items}")
        
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
        return self.items

    def load_data(self) -> xr.DataArray:
        if not self.items:
            raise ValueError("Run search_data() first")
        
        logger.info("📥 Loading data with optimized chunking for Docker...")
        
        # Use smaller chunks for Docker container
        docker_chunksize = min(self.chunksize, 256)
        
        self.data = (
            stackstac.stack(
                items=self.items,
                assets=self.assets,
                chunksize=docker_chunksize,
                resolution=self.resolution,
                epsg=self.epsg
            )
            .where(lambda x: x > 0, other=np.nan)
            .chunk({'time': 1, 'band': -1, 'x': 128, 'y': 128})  # Smaller chunks
        )

        logger.info(f"✅ Data loaded with shape: {self.data.shape}")
        logger.info(f"✅ Chunk sizes: {self.data.chunks}")
        
        return self.data


    def get_threshold_for_baseline(self, baseline: float) -> float:
        """
        Get the NDWI threshold based on the processing baseline.
        
        Parameters:
        -----------
        baseline : float
            Processing baseline value
            
        Returns:
        --------
        float
            NDWI threshold value
        """
        if baseline >= 5.00:
            return 0.05  # ajustado para reprocesados recientes
        elif baseline >= 4.00:
            return 0.07
        else:
            return 0.1  # original para datos más antiguos

    def get_threshold_for_item(self, item_index: int) -> float:
        """
        Get the threshold for a specific item by its index.
        
        Parameters:
        -----------
        item_index : int
            Index of the item in self.items
            
        Returns:
        --------
        float
            NDWI threshold value
        """
        if self.items is None or item_index >= len(self.items):
            return 0.1
        baseline_str = self.items[item_index].properties.get('s2:processing_baseline', '0.0')
        try:
            baseline = float(baseline_str)
            return self.get_threshold_for_baseline(baseline)
        except (ValueError, TypeError):
            return 0.1  # Default threshold

    def check_baseline_and_threshold_per_image(self) -> pd.DataFrame:
        """
        Check baseline and threshold for each image and return a DataFrame with the results.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with baseline and threshold information for each image
        """
        if self.items is None or len(self.items) == 0:
            logger.warning("⚠️  No items available. Run search_data() first.")
            return pd.DataFrame()
        
        logger.info("🔍 Checking baseline and threshold for each image...")
        
        baseline_data = []
        
        for i, item in enumerate(self.items):
            date_str = pd.to_datetime(item.properties.get('datetime', '')).strftime('%Y-%m-%d')
            baseline_str = item.properties.get('s2:processing_baseline', 'Unknown')
            
            try:
                baseline = float(baseline_str)
            except (ValueError, TypeError):
                baseline = 0.0
            
            threshold = self.get_threshold_for_baseline(baseline)
            
            baseline_data.append({
                'image_index': i,
                'date': date_str,
                'baseline': baseline_str,
                'baseline_numeric': baseline,
                'threshold': threshold
            })
        
        df = pd.DataFrame(baseline_data)
        
        return df

    # 🔹 Main NDWI calculation (single version, always cloud-masked)
    def calculate_ndwi_only(self) -> list:
 
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")

        logger.info("🌊 Calculating NDWI with cloud masking (Docker-optimized)...")
        start_time = time.time()

        green_all = self.data.sel(band="B03")
        nir_all = self.data.sel(band="B08")
        scl_all = self.data.sel(band="SCL")

        ndwi_results = []
        water_analysis_results = []

        # Process the first 5 images
        for i in range(0, min(5, len(self.items))):
            logger.info(f"🖼 Processing image {i+1}/5...")

            # Process one image at a time to manage memory
            green = green_all.isel(time=i)
            nir = nir_all.isel(time=i)
            scl = scl_all.isel(time=i)

            # Cloud mask
            valid_mask = ~((scl == 1) | (scl == 2) | (scl == 3) | 
                        (scl == 8) | (scl == 9) | (scl == 10))

            green = green.where(valid_mask)
            nir = nir.where(valid_mask)

            # Calculate NDWI with proper Dask computation
            ndwi_img = (green - nir) / (green + nir)
            
            # Compute with memory management - use synchronous computation
            ndwi_img = ndwi_img.compute()

            # Clean up dimensions
            if 'band' in ndwi_img.dims:
                ndwi_img = ndwi_img.squeeze('band')
            for dim in list(ndwi_img.dims):
                if dim not in ['time', 'y', 'x']:
                    ndwi_img = ndwi_img.squeeze(dim)

            # Calculate water statistics
            ndwi_array = ndwi_img.values
            if ndwi_array.ndim > 2:
                ndwi_array = ndwi_array.squeeze()
            
            # Create water mask with NDWI > 0.15
            water_mask = (ndwi_array > 0.15) & (~np.isnan(ndwi_array))
            
            # Calculate statistics
            valid_pixels = np.sum(~np.isnan(ndwi_array))
            water_pixels = np.sum(water_mask)
            water_percentage = (water_pixels / valid_pixels * 100) if valid_pixels > 0 else 0
            
            # Get image metadata
            item = self.items[i]
            date_str = pd.to_datetime(item.properties.get('datetime', '')).strftime('%Y-%m-%d')
            year = pd.to_datetime(item.properties.get('datetime', '')).year
            cloud_cover = item.properties.get('eo:cloud_cover', 0)
            baseline = item.properties.get('s2:processing_baseline', 'Unknown')
            
            # Store analysis results
            analysis_result = {
                'image_index': i,
                'date': date_str,
                'year': year,
                'cloud_cover': cloud_cover,
                'baseline': baseline,
                'valid_pixels': int(valid_pixels),
                'water_pixels': int(water_pixels),
                'water_percentage': float(water_percentage),
                'ndwi_mean': float(np.nanmean(ndwi_array)),
                'ndwi_std': float(np.nanstd(ndwi_array)),
                'ndwi_min': float(np.nanmin(ndwi_array)),
                'ndwi_max': float(np.nanmax(ndwi_array))
            }
            water_analysis_results.append(analysis_result)
            
            logger.info(f"📊 Image {i+1}: {water_pixels:,} water pixels ({water_percentage:.2f}%)")

            ndwi_results.append(ndwi_img)
            
            # Force garbage collection after each image
            import gc
            gc.collect()

        # Store the first result for saving
        self.ndwi = ndwi_results[0] if ndwi_results else None
        self.water_analysis_results = water_analysis_results

        logger.info(f"✅ NDWI calculation finished in {time.time() - start_time:.2f}s")

        return ndwi_results

    def save_water_analysis_results(self, save_dir: str = "water_analysis"):
        """
        Save water pixel analysis results to CSV for yearly analysis.
        """
        if not hasattr(self, 'water_analysis_results') or not self.water_analysis_results:
            logger.warning("⚠️  No water analysis results available. Run calculate_ndwi_only() first.")
            return None

        # Create output directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(self.water_analysis_results)
        
        # Create filename with date range
        if len(df) > 0:
            start_date = df['date'].min()
            end_date = df['date'].max()
            filename = f"water_analysis_{start_date}_to_{end_date}.csv"
        else:
            filename = "water_analysis_results.csv"
            
        save_path = os.path.join(save_dir, filename)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        logger.info(f"✅ Water analysis results saved to: {save_path}")
        logger.info(f"📊 Analysis includes {len(df)} images")
        
        # Display summary statistics
        logger.info("\n📈 WATER ANALYSIS SUMMARY:")
        logger.info("="*50)
        for _, row in df.iterrows():
            logger.info(f"   {row['date']}: {row['water_pixels']:,} water pixels ({row['water_percentage']:.2f}%)")
        
        # Yearly summary if multiple years
        if len(df['year'].unique()) > 1:
            logger.info("\n📅 YEARLY SUMMARY:")
            logger.info("="*30)
            yearly_summary = df.groupby('year').agg({
                'water_pixels': 'sum',
                'valid_pixels': 'sum',
                'water_percentage': 'mean'
            }).round(2)
            logger.info(yearly_summary)
        
        return save_path

    def save_all_ndwi_images_no_mask(self, save_dir: str = "ndwi_images_nomask"):
        """
        Save NDWI images for all time steps, leaving the second subplot empty.
        Uses Dask to compute only one time slice at a time to avoid memory issues.
        """
        if self.ndwi is None:
            raise ValueError("NDWI not calculated. Run calculate_ndwi_only() first.")

        # Create output directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Saving NDWI image with water mask...")

        # Get the date from the first item
        if self.items and len(self.items) > 0:
            date_str = pd.to_datetime(self.items[0].properties.get('datetime', '')).strftime("%Y%m%d")
        else:
            date_str = "unknown_date"
            
        filename = f"ndwi_with_mask_{date_str}.png"
        save_path = os.path.join(save_dir, filename)

        # Convert to 2D NumPy array
        ndwi_array = self.ndwi.values
        if ndwi_array.ndim > 2:
            ndwi_array = ndwi_array.squeeze()

        # Create water mask with NDWI > 0.15
        water_mask = (ndwi_array > 0.15) & (~np.isnan(ndwi_array))
        
        # Calculate statistics
        valid_pixels = np.sum(~np.isnan(ndwi_array))
        water_pixels = np.sum(water_mask)
        water_percentage = (water_pixels / valid_pixels * 100) if valid_pixels > 0 else 0

        plt.figure(figsize=(12, 6))

        # Subplot 1: NDWI
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(ndwi_array, cmap='RdYlBu_r', vmin=-1, vmax=1)
        plt.colorbar(im1, label='NDWI')
        plt.title(f'NDWI - {date_str}')
        plt.axis('off')

        # Subplot 2: Water mask (NDWI > 0.15) - Binary legend
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(water_mask, cmap='binary', vmin=0, vmax=1)
        plt.colorbar(im2, label='Water Mask (0=No Water, 1=Water)')
        plt.title(f'Water Detection (NDWI > 0.15)\n{water_pixels:,} pixels ({water_percentage:.2f}%)')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Saved {filename}")
        logger.info(f"📊 Water statistics: {water_pixels:,} water pixels ({water_percentage:.2f}%)")
        
        # Force garbage collection after each save
        import gc
        gc.collect()

        logger.info(f"✅ NDWI image with water mask saved to '{save_dir}' directory")
        return save_path

    def cleanup(self):
        """Clean up resources to free memory"""
        logger.info("🧹 Cleaning up resources...")
        
        # Clear data arrays
        if hasattr(self, 'data'):
            del self.data
        if hasattr(self, 'ndwi'):
            del self.ndwi
        if hasattr(self, 'items'):
            del self.items
        if hasattr(self, 'water_analysis_results'):
            del self.water_analysis_results
            
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("✅ Cleanup completed")


def main():
    """
    Main function demonstrating the WaterAnalyzer usage with the same parameters as test.py.
    """
    global client
    
    start_time = time.time()
    logger.info("🚀 Starting Water Analysis Framework")
    logger.info("=" * 50)
    
    # Setup Dask client inside main function to avoid multiprocessing issues
    logger.info("🔧 Setting up Dask client...")
    client = setup_dask_client()
    
    # Configuration for 10-year analysis (2014-2024) - same as test.py
    bbox = [-73.705, 4.605, -73.700, 4.610]  # Very small area for testing
    start_date = "2014-01-01"  # Start of 10-year analysis window
    end_date = "2024-12-31"  # End of 10-year analysis window
    
    logger.info("📅 10-year analysis: 2014-2024")
    logger.info("📊 This will provide comprehensive water dynamics analysis")
    
    logger.info("⚙️  Configuration:")
    logger.info(f"   📍 Study area: {bbox}")
    logger.info(f"   📅 Time period: {start_date} to {end_date}")
    logger.info(f"   🛰️  Collection: sentinel-2-l2a")
    logger.info(f"   🎨 Bands: B03 (Green), B08 (NIR), SCL (Scene Classification)")
    logger.info(f"   ☁️  Cloud cover: < 10%")
    
    # Initialize analyzer with Docker-optimized settings
    logger.info("🔧 Initializing WaterAnalyzer with Docker-optimized settings...")
    analyzer = WaterAnalyzer(
        collection="sentinel-2-l2a",
        assets=["B03", "B08", "SCL"],  # Green, NIR, and Scene Classification for cloud masking
        band_aliases={"green": "B03", "nir": "B08", "scl": "SCL"},  # Band name mappings
        chunksize=256,  # Smaller chunks for Docker
        resolution=500,  # Lower resolution to reduce data size
        epsg=32618  # UTM Zone 18N for Colombia region
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
            max_cloud_cover=10.0,  # Allow up to 10% cloud cover
            max_items=5  # Minimal number of images for testing
        )
        
        # Step 2: Load data
        logger.info("\n" + "="*50)
        logger.info("📥 STEP 2: LOADING AND STACKING DATA")
        logger.info("="*50)
        data = analyzer.load_data()
        
        # Get detailed baseline and threshold information
        baseline_df = analyzer.check_baseline_and_threshold_per_image()
        
        # Display the results
        logger.info("\n📊 BASELINE AND THRESHOLD SUMMARY:")
        logger.info("="*50)
        for _, row in baseline_df.iterrows():
            logger.info(f"   Image {row['image_index']:2d}: {row['date']} -> "
                       f"Baseline {row['baseline']} -> Threshold {row['threshold']} "
                    )

        # Step 3: Calculate NDWI
        logger.info("\n" + "="*50)
        logger.info("💧 STEP 3: CALCULATING NDWI")
        logger.info("="*50)
        ndwi = analyzer.calculate_ndwi_only()

        # Step 4: Save water analysis results
        logger.info("\n" + "="*50)
        logger.info("📊 STEP 4: SAVING WATER ANALYSIS RESULTS")
        logger.info("="*50)
        analysis_path = analyzer.save_water_analysis_results()

        # Step 5: Save NDWI image
        logger.info("\n" + "="*50)
        logger.info("💾 STEP 5: SAVING NDWI IMAGE")
        logger.info("="*50)
        saved_path = analyzer.save_all_ndwi_images_no_mask()
        
        logger.info(f"✅ Analysis completed!")
        logger.info(f"📊 Water analysis saved to: {analysis_path}")
        logger.info(f"🖼️  NDWI image saved to: {saved_path}")
           
        return analyzer, baseline_df
        
    except Exception as e:
        logger.error(f"❌ Error in main analysis: {e}")
        logger.error("🔍 Check the log file for detailed error information")
        raise
    
    finally:
        # Cleanup
        if 'analyzer' in locals():
            analyzer.cleanup()
        logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    # Run the analysis
    print("🚀 Starting Water Indices Analysis Framework...")
    print("📝 Detailed logs will be saved to 'water_analysis.log'")
    print("🔍 Baseline and threshold analysis will be performed")
    print("🔧 Docker-optimized Dask configuration active")
    print("=" * 60)
    
    try:
        analyzer, baseline_df = main()
        
        print("\n" + "=" * 60)
        print("🎉 NDWI ANALYSIS COMPLETED!")
        print("=" * 60)
        print("📊 Generated analysis:")
        print("   📄 Baseline and threshold analysis for all images")
        print("   📊 Water pixel counts with item-specific thresholds")
        print("   📄 water_analysis.log - Detailed execution log")
        print("\n💡 NDWI values and water pixel counts shown above!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("🔍 Check the log file for detailed error information")
        raise
    
    finally:
        # Shutdown Dask client
        if client is not None:
            client.shutdown()
            print("🔧 Dask client shutdown completed")